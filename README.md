# CenterLineDet
This is the official repo of paper **CenterLineDet: Road Lane CenterLine Graph Detection With Vehicle-Mounted Sensors by Transformer for High-definition Map Creation** by Zhenhua Xu, Yuxuan Liu, Yuxiang Sun, Ming Liu and Lujia Wang.

## Update 
Feb/17/2023: Add raw outputs of CenterLineDet

Jan/18/2023: Release the training code

Jan/17/2023: Accepted by ICRA 2023

Oct/15/2022: Release the inference code

## Platform info
Hardware:
```
GPU: 4 RTX3090
CPU: Intel(R) Xeon(R) Gold 6230 CPU @ 2.10GHz
RAM: 256G
SSD: 4T
```
Software:
```
Ubuntu 20.04.3 LTS
CUDA 11.1
Docker 20.10.7
Nvidia-driver 495.29.05
```
## Docker 
This repo is implemented in the docker container. Make sure you have docker installed. Please refer to [install Docker](https://docs.docker.com/engine/install/ubuntu/) and [Docker beginner tutorial](https://docker-curriculum.com/) for more information.

### Docker image
For train and inference:
```
cd docker
./build_image.bash
```

For evaluation:
```
cd docker_py2
./build_image.bash
```
### Docker container
In ```./build_continer.bash``` and ```./build_continer_py2.bash```, set ```home_dir``` as the directory of this repo, and set ```dataset_dir``` as the directory of the downloaded nuscenes dataset. 

For train and inference, run
```
./build_continer.bash
```

For evaluation, run 
```
./build_continer_py2.bash
```


## Data preparation and pretrained checkpoints
Check ```./data``` for data preparation and pretrained checkpoints.


## Implementation
For baseline models (i.e., segmentation based approachs including HDMapNet and our proposed FusionNet), please refer to ```./segmentation_baselines```.

For CenterLineDet with different perspective transformation models, please refer to ```./CenterLineDet```.

## Contact
For any questions, please open an issue.

## Ackonwledgement
We thank the following open-sourced projects:

[HDMapNet](https://github.com/Tsinghua-MARS-Lab/HDMapNet)

[STSU](https://github.com/ybarancan/STSU)

[LaneExtraction](https://github.com/songtaohe/LaneExtraction)

[SAT2GRAPH](https://github.com/songtaohe/Sat2Graph)

[TopoRoad](https://github.com/ybarancan/TopologicalLaneGraph)

[DETR](https://github.com/facebookresearch/detr)

## Citation
```
@INPROCEEDINGS{10161508,
  author={Xu, Zhenhua and Liu, Yuxuan and Sun, Yuxiang and Liu, Ming and Wang, Lujia},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={CenterLineDet: CenterLine Graph Detection for Road Lanes with Vehicle-mounted Sensors by Transformer for HD Map Generation}, 
  year={2023},
  volume={},
  number={},
  pages={3553-3559},
  keywords={Laser radar;Network topology;Roads;Transformers;Topology;Sensors;Planning},
  doi={10.1109/ICRA48891.2023.10161508}}
```
## License
GNU General Public License v3.0
