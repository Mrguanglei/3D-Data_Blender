import argparse
import math
import os
import random
import sys
import time
import urllib.request
from typing import Tuple
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
parser.add_argument("--output_dir", type=str, default="./views")
parser.add_argument(
    "--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"]
)
parser.add_argument("--num_images", type=int, default=32)
parser.add_argument("--resolution", type=int, default=1024)
    
argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

print('===================', args.engine, '===================')

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

# Set the device_type
cycles_preferences = bpy.context.preferences.addons["cycles"].preferences
cycles_preferences.compute_device_type = "CUDA"  # or "OPENCL"
cuda_devices = cycles_preferences.get_devices_for_type("CUDA")
for device in cuda_devices:
    device.use = True

def compose_RT(R, T):
    return np.hstack((R, T.reshape(-1, 1)))

def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )

def sample_spherical(radius=3.0, maxz=3.0, minz=0.):
    correct = False
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        vec[2] = np.abs(vec[2])
        vec = vec / np.linalg.norm(vec, axis=0) * radius
        if maxz > vec[2] > minz:
            correct = True
    return vec

def sample_spherical(radius_min=2.0, radius_max=3.5, maxz=1.6, minz=-0.75):
    correct = False
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
#         vec[2] = np.abs(vec[2])
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
        # from https://blender.stackexchange.com/questions/18530/
        x, y, z = sample_spherical(radius_min=2.0, radius_max=3.5, maxz=1.60, minz=-0.75)
    elif option == 'front':
        x, y, z = 0, -np.random.uniform(2.0, 3.5, 1)[0], 0

    camera.location = x, y, z

    # adjust orientation
    direction = - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    return camera

def add_lighting(option: str) -> None:
    assert option in ['fixed', 'random']
    
    # delete the default light
    bpy.data.objects["Light"].select_set(True)
    bpy.ops.object.delete()
    
    # add a new light
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

    # set light scale
    bpy.data.objects["Area"].scale[0] = 200
    bpy.data.objects["Area"].scale[1] = 200
    bpy.data.objects["Area"].scale[2] = 200


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
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


def normalize_scene(box_scale: float):
    bbox_min, bbox_max = scene_bbox()
    scale = box_scale / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")


def setup_camera():
    cam = scene.objects["Camera"]
    cam.location = (0, 1.2, 0)
    cam.data.angle = math.radians(50)  # Set FOV to 50 degrees
    cam.data.sensor_width = 32
    cam.data.sensor_height = 32  # affects intrinsics calculation, should be set explicitly
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    return cam, cam_constraint


def save_images(object_file: str) -> None:
    """Saves rendered images of the object in the scene."""
    os.makedirs(args.output_dir, exist_ok=True)
    reset_scene()

    # load the object
    load_object(object_file)
    object_uid = os.path.basename(object_file).split(".")[0]
    normalize_scene(box_scale=1.5)  # Adjusted from 2 to prevent clipping while maintaining project compatibility
    add_lighting(option='random')
    camera, cam_constraint = setup_camera()

    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    # prepare to save
    base_dir = os.path.join(args.output_dir, 'rendering_random_32views', object_uid)
    os.makedirs(base_dir, exist_ok=True)

    # Enable passes for normal and depth
    view_layer = context.view_layer
    view_layer.use_pass_normal = True
    view_layer.use_pass_z = True

    # Set camera clip end to match depth mapping
    camera.data.clip_end = 6.0

    # Set up Compositor nodes
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    # Render Layers node
    rl = tree.nodes.new('CompositorNodeRLayers')

    # Process Depth: Map Range from [0, clip_end] to [0,1] (near black, far white), clamp for inf background to white
    # Process Normal: Separate XYZ, Add 0.5 to each, Combine XYZ for [0,1]
    sep_xyz = tree.nodes.new('CompositorNodeSeparateXYZ')
    tree.links.new(rl.outputs['Normal'], sep_xyz.inputs[0])

    add_x = tree.nodes.new('CompositorNodeMath')
    add_x.operation = 'ADD'
    add_x.inputs[1].default_value = 0.5
    tree.links.new(sep_xyz.outputs[0], add_x.inputs[0])

    add_y = tree.nodes.new('CompositorNodeMath')
    add_y.operation = 'ADD'
    add_y.inputs[1].default_value = 0.5
    tree.links.new(sep_xyz.outputs[1], add_y.inputs[0])

    add_z = tree.nodes.new('CompositorNodeMath')
    add_z.operation = 'ADD'
    add_z.inputs[1].default_value = 0.5
    tree.links.new(sep_xyz.outputs[2], add_z.inputs[0])

    comb_xyz = tree.nodes.new('CompositorNodeCombineXYZ')
    tree.links.new(add_x.outputs[0], comb_xyz.inputs[0])
    tree.links.new(add_y.outputs[0], comb_xyz.inputs[1])
    tree.links.new(add_z.outputs[0], comb_xyz.inputs[2])

    # Mix for Normal background (blue-purple: 0.5,0.5,1.0)
    constant_color = tree.nodes.new('CompositorNodeRGB')
    constant_color.outputs[0].default_value = (0.5, 0.5, 1.0, 1.0)

    mix_normal = tree.nodes.new('CompositorNodeMixRGB')
    mix_normal.blend_type = 'MIX'
    tree.links.new(rl.outputs['Alpha'], mix_normal.inputs[0])  # Fac (alpha)
    tree.links.new(constant_color.outputs[0], mix_normal.inputs[1])  # Background
    tree.links.new(comb_xyz.outputs[0], mix_normal.inputs[2])  # Foreground

    # Output for Combined (RGBA)
    file_output_combined = tree.nodes.new('CompositorNodeOutputFile')
    file_output_combined.base_path = base_dir
    file_output_combined.format.file_format = 'PNG'
    file_output_combined.format.color_mode = 'RGBA'
    file_output_combined.file_slots[0].path = ''

    # Output for Normal
    file_output_normal = tree.nodes.new('CompositorNodeOutputFile')
    file_output_normal.base_path = base_dir
    file_output_normal.format.file_format = 'PNG'
    file_output_normal.format.color_mode = 'RGB'  # Opaque with background color
    file_output_normal.file_slots[0].path = '_normal'

    # Save depth maps as png and exr
    depth_map = tree.nodes.new(type="CompositorNodeMapRange")
    depth_map.inputs['From Min'].default_value = 0.01
    depth_map.inputs['From Max'].default_value = 6.0
    depth_map.inputs['To Min'].default_value = 0.0
    depth_map.inputs['To Max'].default_value = 1.0
    tree.links.new(rl.outputs['Depth'], depth_map.inputs['Value'])

    depth_save = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_save.base_path = base_dir
    depth_save.format.file_format = 'PNG'
    depth_save.format.color_mode = 'BW'
    depth_save.format.color_depth = '8'
    depth_save.file_slots[0].path = '_depth'
    tree.links.new(depth_map.outputs[0], depth_save.inputs['Image'])

    # Link nodes
    tree.links.new(rl.outputs['Image'], file_output_combined.inputs[0])
    tree.links.new(mix_normal.outputs[0], file_output_normal.inputs[0])

    # List to collect C2W matrices
    c2w_list = []

    for i in range(args.num_images):
        # set the camera position
        camera_option = 'random' if i > 0 else 'front'
        camera = set_camera_location(camera, option=camera_option)

        # Set frame to 0 to standardize suffix '0000'
        scene.frame_current = 0

        # Set file names for this frame
        frame_str = f"{i:03d}"
        file_output_combined.file_slots[0].path = frame_str
        file_output_normal.file_slots[0].path = frame_str + '_normal'
        depth_save.file_slots[0].path = frame_str + '_depth'

        # render the image
        bpy.ops.render.render(write_still=True)

        # Rename files to remove '0000' suffix
        for suffix, ext in [('', '.png'), ('_normal', '.png'), ('_depth', '.png')]:
            old_suffix = f"{frame_str}{suffix}0000{ext}"
            new_suffix = f"{frame_str}{suffix}{ext}"
            old_path = os.path.join(base_dir, old_suffix)
            new_path = os.path.join(base_dir, new_suffix)
            if os.path.exists(old_path):
                os.rename(old_path, new_path)

        # save camera RT matrix (C2W)
        location, rotation = camera.matrix_world.decompose()[0:2]
        R = rotation.to_matrix()
        T = np.array(location)
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = T
        c2w_list.append(c2w)

    # Save poses as W2C in npz
    w2c_list = [np.linalg.inv(c2w) for c2w in c2w_list]
    np.savez(os.path.join(base_dir, 'cameras.npz'), cam_poses=np.array(w2c_list))


def download_object(object_url: str) -> str:
    """Download the object and return the path."""
    # uid = uuid.uuid4()
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(object_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)
    # get the absolute path
    local_path = os.path.abspath(local_path)
    return local_path


def get_calibration_matrix_K_from_blender(camera, return_principles=False):
    """
        Get the camera intrinsic matrix from Blender camera.
        Return also numpy array of principle parameters if specified.
        
        Intrinsic matrix K has the following structure in pixels:
            [fx  0 cx]
            [0  fy cy]
            [0   0  1]
        
        Specified principle parameters are:
            [fx, fy] - focal lengths in pixels
            [cx, cy] - optical centers in pixels
            [width, height] - image resolution in pixels
        
    """
    # Render resolution
    render = bpy.context.scene.render
    width = render.resolution_x * render.pixel_aspect_x
    height = render.resolution_y * render.pixel_aspect_y

    # Camera parameters
    focal_length = camera.lens  # Focal length in millimeters
    sensor_width = camera.sensor_width  # Sensor width in millimeters
    sensor_height = camera.sensor_height  # Sensor height in millimeters

    # Calculate the focal length in pixel units
    focal_length_x = width * (focal_length / sensor_width)
    focal_length_y = height * (focal_length / sensor_height)

    # Assuming the optical center is at the center of the sensor
    optical_center_x = width / 2
    optical_center_y = height / 2

    # Constructing the intrinsic matrix
    K = np.array([[focal_length_x, 0, optical_center_x],
                [0, focal_length_y, optical_center_y],
                [0, 0, 1]])
    
    if return_principles:
        return np.array([
            [focal_length_x, focal_length_y],
            [optical_center_x, optical_center_y],
            [width, height],
        ])
    else:
        return K


if __name__ == "__main__":
    try:
        start_i = time.time()
        if args.object_path.startswith("http"):
            local_path = download_object(args.object_path)
        else:
            local_path = args.object_path
        save_images(local_path)
        end_i = time.time()
        print("Finished", local_path, "in", end_i - start_i, "seconds")
        # delete the object if it was downloaded
        if args.object_path.startswith("http"):
            os.remove(local_path)
    except Exception as e:
        print("Failed to render", args.object_path)
        print(e)
