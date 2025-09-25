#  train the E-Mesh

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-yellow.svg)](LICENSE)
[![Weight License](https://img.shields.io/badge/Weight%20License-CC%20By%20NC%204.0-red)](LICENSE_WEIGHT)
[![LRM](https://img.shields.io/badge/LRM-Arxiv%20Link-green)](https://arxiv.org/abs/2311.04400)

[![HF Models](https://img.shields.io/badge/Models-Huggingface%20Models-bron)](https://huggingface.co/zxhezexin)
[![HF Demo](https://img.shields.io/badge/Demo-Huggingface%20Demo-blue)](https://huggingface.co/spaces/zxhezexin/OpenLRM)



## News

- [2025.09.15] Update [training  E-Mesh  code](scripts/data/objaverse/blender_script.py) and release [OpenLRM v1.1.1](https://github.com/3DTopia/OpenLRM/releases/tag/v1.1.1).



## Setup

### Installation
```
git clone https://github.com/Mrguanglei/Instantmesh_scriptData.git
cd OpenLRM
```

### Environment
- Install requirements for OpenLRM first.
  ```
  conda create --name Openlrm  python=3.9 -y
  pip install -r requirements.txt
  ```
- Please then follow the [xFormers installation guide](https://github.com/facebookresearch/xformers?tab=readme-ov-file#installing-xformers) to enable memory efficient attention inside [DINOv2 encoder](openlrm/models/encoders/dinov2/layers/attention.py).

## Quick Start

### Dataset format 

```Bash
|--rendering_random_32views
|--- 1 # object id
| |--- 000.png
| |--- 000_normal.png
| |--- 000_depth.png
| |--- ...
| |--- 031.png
| |--- 031_normal.png
| |--- 031_depth.png
| |--- camera.npz # 
|--- 2 # object id
| |--- 000.png
| |--- 000_normal.png
| |--- 000_depth.png
| |--- ...
| |--- 031.png
| |--- 031_normal.png
| |--- 031_depth.png
| |--- camera.npz #
|--- 3 # object id
| |--- 000.png
| |--- 000_normal.png
| |--- 000_depth.png
| |--- ...
| |--- 031.png
| |--- 031_normal.png
| |--- 031_depth.png
| |--- camera.npz # 
|-- valid_paths.json
|-- ...
```

valid_paths.json fomerat:
```Bash
{
  "good_objs":[
      "1",
      "2",
      ...
  ]

}
```


### Downloading the dataset 

**I took over 200 glb files from the Objaverse dataset and used the glb to render the dataset we needed**

**1.Downloading the dataset:[Dataset address](https://drive.google.com/drive/folders/1_s1W8Pq_1D5xouvefREFHSRgawHqefJR).**

**2.Place the glb file in the data folder**



### Modifying the script
- Find the script that we need to modify `scripts/data/objaverse/blender.sh`, 

  ```py	
  DIRECTORY="/your/path/OpenLRM/data"    #Put the path to the dataset file you downloaded here
  ```


### Blender:

**Run `blender.sh` .**

**This will automatically render the dataset we need above.**

### Tips
- The recommended PyTorch version is `>=2.1`. Code is developed and tested under PyTorch `2.1.2`.
- If you encounter CUDA OOM issues, please try to reduce the `frame_size` in the inference configs.
- You should be able to see `UserWarning: xFormers is available` if `xFormers` is actually working.
- **If there is no module in bpy and mathutils, please look up the information yourself.**



## Citation

If you find this work useful for your research, please consider citing:
```
@article{hong2023lrm,
  title={Lrm: Large reconstruction model for single image to 3d},
  author={Hong, Yicong and Zhang, Kai and Gu, Jiuxiang and Bi, Sai and Zhou, Yang and Liu, Difan and Liu, Feng and Sunkavalli, Kalyan and Bui, Trung and Tan, Hao},
  journal={arXiv preprint arXiv:2311.04400},
  year={2023}
}
```



## License

- OpenLRM as a whole is licensed under the [Apache License, Version 2.0](LICENSE), while certain components are covered by [NVIDIA's proprietary license](LICENSE_NVIDIA). Users are responsible for complying with the respective licensing terms of each component.
- Model weights are licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE_WEIGHT). They are provided for research purposes only, and CANNOT be used commercially.
