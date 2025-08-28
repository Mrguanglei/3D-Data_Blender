import argparse
import math
import os
import random
import shutil
import sys
import time
import urllib.request
from mathutils import Vector
import numpy as np
import bpy
parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the object file",
)
parser.add_argument("--output_dir", type=str, default="./rendering_random_32views")
parser.add_argument(
    "--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"]
)
parser.add_argument("--num_images", type=int, default=32)
parser.add_argument("--resolution", type=int, default=1024)
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--camera_sampling", type=str, default="random", 
                    choices=["random", "uniform"], 
                    help="相机采样策略: random=随机分布, uniform=均匀分布")
parser.add_argument("--enhanced_normals", action="store_true", default=True,
                    help="启用增强的法线图渲染配置以获得更高质量的法线图")

if "--" in sys.argv:
    argv = sys.argv[sys.argv.index("--") + 1:]
else:
    argv = sys.argv[1:]

args = parser.parse_args(argv)

context = bpy.context
scene = context.scene
render = scene.render

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = args.resolution
render.resolution_y = args.resolution
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 128
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True

# 法线图渲染优化设置
scene.cycles.max_bounces = 2           # 限制光线弹射次数，对法线图不重要
scene.cycles.preview_samples = 32      # 预览采样数
scene.cycles.aa_samples = 16           # 抗锯齿采样数
scene.cycles.use_square_samples = False # 使用线性采样计数

# 优化法线pass的质量
scene.view_layers[0].cycles.use_denoising = False  # 法线图不需要降噪，保持原始精度

# Set the device_type
cycles_preferences = bpy.context.preferences.addons["cycles"].preferences
cycles_preferences.compute_device_type = "CUDA"
cuda_devices = cycles_preferences.get_devices_for_type("CUDA")
for device in cuda_devices:
    device.use = False

def compose_RT(R, T):
    return np.hstack((R, T.reshape(-1, 1)))

def sample_spherical(radius_min=1.5, radius_max=2.0, maxz=1.6, minz=-0.75):
    correct = False
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        radius = np.random.uniform(radius_min, radius_max, 1)
        vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
        if maxz > vec[2] > minz:
            correct = True
    return vec

def set_camera_location(camera, option: str):
    assert option in ['fixed', 'random', 'front']

    if option == 'fixed':
        x, y, z = 0, -2.25, 0
    elif option == 'random':
        x, y, z = sample_spherical(radius_min=2.5, radius_max=3.5, maxz=1.60, minz=-0.75)
    elif option == 'front':
        x, y, z = 0, -np.random.uniform(1.9, 2.6, 1)[0], 0

    camera.location = x, y, z
    direction = - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    return camera

def set_camera_location_with_angles(camera, azimuth_deg: float, elevation_deg: float, radius: float = 3.0):
    """
    根据方位角和仰角设置相机位置
    :param azimuth_deg: 方位角（度）, 0度为-Y方向
    :param elevation_deg: 仰角（度）, 正值向上
    :param radius: 相机到原点的距离
    """
    azimuth_rad = np.deg2rad(azimuth_deg)
    elevation_rad = np.deg2rad(elevation_deg)
    
    # 球面坐标转笛卡尔坐标
    x = radius * np.cos(elevation_rad) * np.sin(azimuth_rad)
    y = -radius * np.cos(elevation_rad) * np.cos(azimuth_rad)  # Blender中-Y为前方
    z = radius * np.sin(elevation_rad)
    
    camera.location = x, y, z
    direction = - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    return camera



def add_lighting(option: str) -> None:
    assert option in ['fixed', 'random']

    bpy.data.objects["Light"].select_set(True)
    bpy.ops.object.delete()

    bpy.ops.object.light_add(type="AREA")
    light = bpy.data.lights["Area"]

    if option == 'fixed':
        light.energy = 30000
        bpy.data.objects["Area"].location[0] = 0
        bpy.data.objects["Area"].location[1] = 1
        bpy.data.objects["Area"].location[2] = 0.5

    elif option == 'random':
        light.energy = random.uniform(80000, 120000)
        bpy.data.objects["Area"].location[0] = random.uniform(-2., 2.)
        bpy.data.objects["Area"].location[1] = random.uniform(-2., 2.)
        bpy.data.objects["Area"].location[2] = random.uniform(1.0, 3.0)

    bpy.data.objects["Area"].scale[0] = 200
    bpy.data.objects["Area"].scale[1] = 200
    bpy.data.objects["Area"].scale[2] = 200

def reset_scene() -> None:
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

def load_object(object_path: str) -> None:
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif object_path.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")

def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj

def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def optimize_materials_for_normals():
    """
    优化场景中的材质以获得更好的法线图渲染效果
    """
    for obj in scene_meshes():
        if obj.data.materials:
            for material in obj.data.materials:
                if material and material.use_nodes:
                    # 确保材质有法线输出
                    nodes = material.node_tree.nodes
                    
                    # 查找输出节点
                    output_node = None
                    for node in nodes:
                        if node.type == 'OUTPUT_MATERIAL':
                            output_node = node
                            break
                    
                    if output_node:
                        # 确保法线连接正确 - 添加法线贴图支持
                        bsdf_node = None
                        for node in nodes:
                            if node.type == 'BSDF_PRINCIPLED':
                                bsdf_node = node
                                break
                        
                        if bsdf_node:
                            # 如果没有法线贴图，确保使用几何法线
                            if not bsdf_node.inputs['Normal'].is_linked:
                                # 材质法线将使用几何法线，这是正确的默认行为
                                pass
                            
                            # 优化材质属性以获得更清晰的法线渲染
                            bsdf_node.inputs['Roughness'].default_value = 0.5  # 适中的粗糙度
                            bsdf_node.inputs['Metallic'].default_value = 0.0   # 非金属材质更适合法线可视化
                            
                else:
                    # 为没有节点的材质创建基础节点设置
                    material.use_nodes = True
                    nodes = material.node_tree.nodes
                    nodes.clear()
                    
                    # 添加基础的BSDF节点
                    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
                    output = nodes.new(type='ShaderNodeOutputMaterial')
                    material.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
                    
                    # 设置适合法线渲染的材质属性
                    bsdf.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1.0)
                    bsdf.inputs['Roughness'].default_value = 0.5
                    bsdf.inputs['Metallic'].default_value = 0.0

def normalize_scene(box_scale: float):
    bbox_min, bbox_max = scene_bbox()
    scale = box_scale / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    
    # 根据参数决定是否优化材质设置
    if args.enhanced_normals:
        optimize_materials_for_normals()

def setup_camera():
    cam = scene.objects["Camera"]
    # 增加相机距离，让物体完全显示在视野内
    cam.location = (0, 2.0, 0)  # 从1.2增加到2.0
    # 设置50度FOV (InstantMesh要求)
    cam.data.angle = math.radians(50)
    # 设置传感器尺寸为正方形
    cam.data.sensor_width = 32
    cam.data.sensor_height = 32
    cam.data.sensor_fit = 'HORIZONTAL'  # 确保FOV计算基准
    # 裁剪平面
    cam.data.clip_start = 0.01
    cam.data.clip_end = 10.0  # 从6.0增加到10.0
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    return cam, cam_constraint

def save_images(object_file: str) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    reset_scene()
    load_object(object_file)
    object_uid = os.path.basename(object_file).split(".")[0]
    normalize_scene(box_scale=1.5)  # 减小物体缩放，确保完全在视野内
    add_lighting(option='random')
    camera, cam_constraint = setup_camera()

    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    img_dir = os.path.join(args.output_dir, object_uid)
    os.makedirs(img_dir, exist_ok=True)

    # Prepare to save camera parameters
    # InstantMesh数据加载器期望的格式:
    # - cam_poses: 3x4 W2C矩阵数组 (N, 3, 4)
    # - intrinsics: 相机内参信息
    cam_params = {
        "intrinsics": get_calibration_matrix_K_from_blender(camera.data, return_principles=True),
        "cam_poses": [],  # W2C格式的3x4矩阵，符合数据加载器期望
    }

    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    for n in tree.nodes:
        tree.nodes.remove(n)

    bpy.context.view_layer.use_pass_normal = True
    bpy.context.view_layer.use_pass_z = True
    rl = tree.nodes.new(type='CompositorNodeRLayers')

    # Save images
    image_save = tree.nodes.new(type='CompositorNodeOutputFile')
    links.new(rl.outputs['Image'], image_save.inputs[0])

    # Save depth maps as png
    depth_map = tree.nodes.new(type="CompositorNodeMapRange")
    depth_map.inputs['From Min'].default_value = 0.01
    depth_map.inputs['From Max'].default_value = 10.0  # 匹配新的clip_end
    depth_map.inputs['To Min'].default_value = 0.0
    depth_map.inputs['To Max'].default_value = 1.0
    links.new(rl.outputs['Depth'], depth_map.inputs['Value'])
    depth_save = tree.nodes.new(type="CompositorNodeOutputFile")
    links.new(depth_map.outputs[0], depth_save.inputs[0])

    # FlexiCubes风格的法线图处理
    # 分离法线的XYZ分量
    sep_xyz = tree.nodes.new(type="CompositorNodeSeparateXYZ")
    links.new(rl.outputs['Normal'], sep_xyz.inputs[0])
    
    # === FlexiCubes精确的数值范围映射: (img + 1) * 0.5 ===
    # X分量处理
    add_x = tree.nodes.new('CompositorNodeMath')
    add_x.operation = 'ADD'
    add_x.inputs[1].default_value = 1.0
    links.new(sep_xyz.outputs[0], add_x.inputs[0])
    
    mult_x = tree.nodes.new('CompositorNodeMath')
    mult_x.operation = 'MULTIPLY'
    mult_x.inputs[1].default_value = 0.5
    links.new(add_x.outputs[0], mult_x.inputs[0])
    
    # Y分量处理
    add_y = tree.nodes.new('CompositorNodeMath')
    add_y.operation = 'ADD'
    add_y.inputs[1].default_value = 1.0
    links.new(sep_xyz.outputs[1], add_y.inputs[0])
    
    mult_y = tree.nodes.new('CompositorNodeMath')
    mult_y.operation = 'MULTIPLY'
    mult_y.inputs[1].default_value = 0.5
    links.new(add_y.outputs[0], mult_y.inputs[0])
    
    # Z分量处理
    add_z = tree.nodes.new('CompositorNodeMath')
    add_z.operation = 'ADD'
    add_z.inputs[1].default_value = 1.0
    links.new(sep_xyz.outputs[2], add_z.inputs[0])
    
    mult_z = tree.nodes.new('CompositorNodeMath')
    mult_z.operation = 'MULTIPLY'
    mult_z.inputs[1].default_value = 0.5
    links.new(add_z.outputs[0], mult_z.inputs[0])
    
    # 重新组合XYZ
    comb_xyz = tree.nodes.new('CompositorNodeCombineXYZ')
    links.new(mult_x.outputs[0], comb_xyz.inputs[0])
    links.new(mult_y.outputs[0], comb_xyz.inputs[1])
    links.new(mult_z.outputs[0], comb_xyz.inputs[2])
    
    # FlexiCubes风格的高质量抗锯齿滤波器 (使用模糊节点替代)
    blur_node = tree.nodes.new('CompositorNodeBlur')
    blur_node.size_x = 1  # 轻微模糊，模拟抗锯齿效果 (Blender 4.3.2需要整数)
    blur_node.size_y = 1
    links.new(comb_xyz.outputs[0], blur_node.inputs[0])
    
    # 白色背景混合
    constant_color = tree.nodes.new('CompositorNodeRGB')
    constant_color.outputs[0].default_value = (1.0, 1.0, 1.0, 1.0)  # 白色背景
    
    mix_normal = tree.nodes.new('CompositorNodeMixRGB')
    mix_normal.blend_type = 'MIX'
    links.new(rl.outputs['Alpha'], mix_normal.inputs[0])  # Alpha as factor
    links.new(constant_color.outputs[0], mix_normal.inputs[1])  # Background
    links.new(blur_node.outputs['Image'], mix_normal.inputs[2])  # Foreground
    
    normal_save = tree.nodes.new(type="CompositorNodeOutputFile")
    links.new(mix_normal.outputs[0], normal_save.inputs[0])

# #添加角度列表
#     num_views=args.num_images
#     views = [(0,30)]
#     azimuths = np.linspace(0, 360, num_views, endpoint=False)  # 均匀分布方位角
#     elevations = [30] * num_views  # 仰角固定为 30 度
#     for i in range(1, num_views):
#         views.append((azimuths[i], elevations[i]))  # 将计算出的视角添加到视角列表
    


    for i in range(args.num_images):
        # RGB图像设置
        image_save.base_path = ""  # 使用完整路径，不设置base_path
        image_save.file_slots[0].use_node_format = False  # 禁用自动编号
        image_save.file_slots[0].path = os.path.join(img_dir, f"{i:03d}")
        image_save.format.file_format = 'PNG'
        image_save.format.color_mode = 'RGBA'

        # 深度图设置
        depth_save.base_path = ""
        depth_save.file_slots[0].use_node_format = False
        depth_save.file_slots[0].path = os.path.join(img_dir, f"{i:03d}_depth")
        depth_save.format.file_format = 'PNG'
        depth_save.format.color_mode = 'BW'
        depth_save.format.color_depth = '8'

        # 法线图设置 - 根据enhanced_normals参数优化
        normal_save.base_path = ""
        normal_save.file_slots[0].use_node_format = False
        normal_save.file_slots[0].path = os.path.join(img_dir, f"{i:03d}_normal")
        normal_save.format.file_format = 'PNG'
        normal_save.format.color_mode = 'RGBA'
        
        if args.enhanced_normals:
            # 增强法线图质量设置
            normal_save.format.color_depth = '16'   # 使用16位色深提高法线图精度
            normal_save.format.compression = 15     # PNG压缩级别，平衡文件大小和质量
        else:
            # 标准设置
            normal_save.format.color_depth = '8'
            normal_save.format.compression = 50
        # # Set the camera position
        # azimuth, elevation=views[i]
        # camera = set_camera_location(camera, 4.0, azimuth, elevation, option='random')
        # bpy.ops.render.render(write_still=True)


        # Set the camera position based on sampling strategy
        if args.camera_sampling == "uniform":
            # 均匀分布：第一个视图是正面，其余按均匀角度分布
            if i == 0:
                camera_option = 'front'
            else:
                # 创建均匀分布的相机位置
                azimuth = (i - 1) * 360.0 / (args.num_images - 1)  # 0到360度均匀分布
                elevation = 30.0  # 固定仰角30度
                camera = set_camera_location_with_angles(camera, azimuth, elevation)
                bpy.ops.render.render(write_still=True)
                RT_w2c = get_3x4_RT_matrix_from_blender(camera)
                cam_params["cam_poses"].append(RT_w2c)
                continue
        else:
            # 随机分布：第一个视图是正面，其余随机
            camera_option = 'random' if i > 0 else 'front'
            
        if args.camera_sampling != "uniform" or i == 0:
            camera = set_camera_location(camera, option=camera_option)
            bpy.ops.render.render(write_still=True)

        # Save camera RT matrix (W2C) - 符合InstantMesh数据加载器格式
        RT_w2c = get_3x4_RT_matrix_from_blender(camera)
        cam_params["cam_poses"].append(RT_w2c)

        # cam_params["eular"].append(camera.rotation_euler)
        # cam_params["location"].append(camera.location)
    # 清理Blender自动添加的文件名后缀
    import re
    for file_name in os.listdir(img_dir):
        file_path = os.path.join(img_dir, file_name)
        if os.path.isfile(file_path):
            name, extension = os.path.splitext(file_name)
            
            # Blender OutputFile节点会自动添加"0001"等后缀，需要清理
            # 处理各种情况:
            # "000_depth0001.png" -> "000_depth.png"  
            # "000_normal0001.png" -> "000_normal.png"
            # "0000001.png" -> "000.png"
            
            original_name = name
            
            # 情况1: 处理 "000_depth0001" -> "000_depth" 格式
            if re.search(r'_(depth|normal)\d{4}$', name):
                clean_name = re.sub(r'(\d{4})$', '', name)  # 移除末尾的4位数字
                
            # 情况2: 处理 "0000001" -> "000" 格式 (纯数字文件名)
            elif re.match(r'^\d{7}$', name):  # 7位数字
                # 提取前3位作为视图编号
                view_num = name[:3]
                clean_name = view_num
                
            # 情况3: 其他可能的格式
            else:
                # 尝试移除末尾的4位数字
                clean_name = re.sub(r'\d{4}$', '', name)
            
            # 如果名称发生了变化，进行重命名
            if clean_name != original_name:
                new_filename = clean_name + extension
                new_filepath = os.path.join(img_dir, new_filename)
                
                print(f"重命名: {file_name} -> {new_filename}")
                
                # 如果目标文件已存在，先删除
                if os.path.exists(new_filepath):
                    os.remove(new_filepath)
                    
                shutil.move(file_path, new_filepath)

    # Save camera intrinsics and poses
    np.savez(os.path.join(img_dir, 'cameras.npz'), **cam_params)
    
    # 验证输出数据的完整性和格式
    validation_result = validate_output_data(img_dir, args.num_images)
    if validation_result["success"]:
        print(f"✅ 数据验证通过: {validation_result['message']}")
    else:
        print(f"❌ 数据验证失败: {validation_result['message']}")
        return False
    
    return True



def validate_output_data(img_dir: str, num_images: int) -> dict:
    """
    验证渲染输出数据的完整性和格式
    """
    try:
        # 检查相机参数文件
        cameras_file = os.path.join(img_dir, 'cameras.npz')
        if not os.path.exists(cameras_file):
            return {"success": False, "message": "缺少相机参数文件 cameras.npz"}
        
        # 加载并验证相机参数
        cam_data = np.load(cameras_file)
        if 'cam_poses' not in cam_data:
            return {"success": False, "message": "相机参数文件缺少 cam_poses 数据"}
        if 'intrinsics' not in cam_data:
            return {"success": False, "message": "相机参数文件缺少 intrinsics 数据"}
        
        cam_poses = cam_data['cam_poses']
        if cam_poses.shape != (num_images, 3, 4):
            return {"success": False, "message": f"相机pose矩阵形状错误: 期望({num_images}, 3, 4), 实际{cam_poses.shape}"}
        
        # 验证图像文件
        missing_files = []
        file_types = ['', '_depth', '_normal']  # RGB、深度、法线图
        
        for i in range(num_images):
            for file_type in file_types:
                png_file = os.path.join(img_dir, f"{i:03d}{file_type}.png")
                if not os.path.exists(png_file):
                    missing_files.append(f"{i:03d}{file_type}.png")
        
        if missing_files:
            return {"success": False, "message": f"缺少图像文件: {missing_files[:5]}{'...' if len(missing_files) > 5 else ''}"}
        
        # 验证第一个图像的尺寸和格式
        first_image = os.path.join(img_dir, "000.png")
        try:
            from PIL import Image
            with Image.open(first_image) as img:
                if img.size != (args.resolution, args.resolution):
                    return {"success": False, "message": f"图像尺寸错误: 期望{args.resolution}x{args.resolution}, 实际{img.size}"}
                if img.mode != 'RGBA':
                    return {"success": False, "message": f"图像格式错误: 期望RGBA, 实际{img.mode}"}
        except Exception as e:
            return {"success": False, "message": f"无法读取图像文件: {e}"}
        
        return {
            "success": True, 
            "message": f"所有{num_images}个视图的数据完整，格式正确"
        }
        
    except Exception as e:
        return {"success": False, "message": f"验证过程出错: {str(e)}"}

def download_object(object_url: str) -> str:
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(object_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)
    local_path = os.path.abspath(local_path)
    return local_path

def get_3x4_RT_matrix_from_blender(cam):
    bpy.context.view_layer.update()
    location, rotation = cam.matrix_world.decompose()[0:2]
    R = np.asarray(rotation.to_matrix())
    t = np.asarray(location)

    cam_rec = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32)
    R = R.T
    t = -R @ t
    R_world2cv = cam_rec @ R
    t_world2cv = cam_rec @ t

    RT = np.concatenate([R_world2cv,t_world2cv[:,None]],1)
    return RT

def get_calibration_matrix_K_from_blender(camera, return_principles=False):
    """
    计算归一化的相机内参矩阵，符合InstantMesh项目标准
    返回归一化内参矩阵，焦距和光心都是相对于图像尺寸的比例
    """
    render = bpy.context.scene.render
    width = render.resolution_x * render.pixel_aspect_x
    height = render.resolution_y * render.pixel_aspect_y
    
    # 使用FOV角度计算归一化焦距（符合项目标准）
    fov_degrees = math.degrees(camera.angle)
    focal_length_normalized = 0.5 / np.tan(np.deg2rad(fov_degrees) * 0.5)
    
    # 归一化内参矩阵（符合InstantMesh项目标准）
    K = np.array([[focal_length_normalized, 0, 0.5],
                  [0, focal_length_normalized, 0.5],
                  [0, 0, 1]])
    
    if return_principles:
        # 返回格式：[[fx, fy], [cx, cy], [width, height]]
        return np.array([
            [focal_length_normalized, focal_length_normalized],
            [0.5, 0.5],  # 归一化光心
            [width, height],
        ])
    else:
        return K

def main(args):
    try:
        start_i = time.time()
        local_path = args.object_path
        success = save_images(local_path)
        end_i = time.time()
        
        if success:
            print(f"✅ 成功完成渲染: {local_path}, 耗时: {end_i - start_i:.2f}秒")
        else:
            print(f"❌ 渲染失败: {local_path}")
            return False
            
    except Exception as e:
        print(f"❌ 渲染过程出错: {args.object_path}")
        print(f"错误详情: {e}")
        return False
    
    return True


if __name__ == "__main__":
    main(args)
