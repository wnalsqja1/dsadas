# TSDF Voxel Mapper (HYU & LGE Project)

## Authors

    Computer Vision Lab @ HYU

    - Changhee Won (chwon@hanyang.ac.kr)
    - Changho Sung (oknkc8@gmail.com)
    - Jongwoo Lim (jongwoo.lim@gmail.com)
    - Sangheon Yang (yangsh.official@gmail.com)
    - Minbum Joo (wnalsqja@gmail.com)


## Install Dependencies

### Linux (Ubuntu)

    * Install necessary packages

    ```
    sudo apt-get install cmake libeigen3-dev libgflags-dev libgoogle-glog-dev libpng-dev libjpeg-dev libglew-dev freeglut3-dev libyaml-cpp-dev
    ```

    * Install pangolin library: https://github.com/stevenlovegrove/Pangolin

## Dataset

### Download Example Data


    Set path as `[project_dir]/data/dataset_name/...`

    #### Multi-FoV

        - Modified from Multi-FoV Dataset (http://rpg.ifi.uzh.ch/fov.html)
        - Download here(Urban): https://drive.google.com/file/d/1f-SvukbBIAlLCmtPK6jw5HICIDUozl-P/view?usp=sharing
        - Download here(VFR):

    #### TUM RGB-D SLAM

        * Modified from TUM RGB-D SLAM Dataset (https://vision.in.tum.de/data/datasets/rgbd-dataset)
        * Download here(freiburg2_pioneer2_360): 
        * Download here(freiburg2_pioneer2_slam): 
        * Download here(freiburg3_long_office_household): 

### Rosbag Data Parsing ( LG seocho etc ...)

#### Environments

Ros melodic : http://wiki.ros.org/melodic/Installation/Ubuntu <br>
OpenCV2 <br>
Eigen3 <br>
cv_bridge <br>

#### Settings 

first, you have to make catkin_workspace in your computer

```shell
$ mkdir -p catkin_ws/src
$ cd catkin_make/src
$ catkin_init_workspace
$ catkin_make
```

second, move directory`[rosbag_synchronizer]` to `[your catkin_ws]/src`

```shell
$ cd catkin_ws
$ catkin_make 
```

#### Run Rosbag_synchronizer
To subscribe after synchronizing rosbag message with timestamp header, Run this code.  
This creates RGB_image, aligned_depth, raw_depth and Pose of rosbag.  
(Since the direction IMU pose of dataset is the direction of the camera up vector, Change to camera direction vector)

```shell
$ roscore
$ rosrun rosbag_synchronizer rosbag_synchronizer
$ rosbag play [-bag file name]
# for checking Intrinsic parameter
$ rostopic echo /camera/color/camera_info
```

outputs are as follows

```shell
[your catkin_ws]/img/aligned_depth/frame%06d.png  # 
[your catkin_ws]/img/color/frame%06d.png
[your catkin_ws]/img/depth/frame%06d.png
[your catkin_ws]/groudtruth.txt
```

#### Trouble shooting
If the 'package [package name] not found' message is output after rosrun, it can be solved through the command below

```shell
$ source ~/catkin_ws/devel/setup.bash
$ rospack profile
```

### Dataset Structure & Format

```shell
.
└── [project_dir]/data/dataset_name/
    ├── data
    │   ├── depth
    │   │   ├── img0001_0.depth
    │   │   ├── img0002_0.depth
    │   │   ├── img0003_0.depth
    │   │   └── ...
    │   └── img
    │       ├── img0001_0.png
    │       ├── img0002_0.png
    │       ├── img0003_0.png
    │       └── ...
    ├── info
    │   └── groundtruth.txt
    └── intrinsic.yaml
```

### Convert 3DOF Pose Data to 6DOF Pose Data

Execute `parse_lg.m`, change the path of dataset and set extrinsic.

```matlab
% example(SCRND_F19)
...
% set dataset path
dir_path = './SCRND_F19/Test_1/image_pose';
pose_path = fullfile(dir_path, 'optimized_pose.txt')
...
% set extrinsic
cam2body = [0.28431189; 1.45908165; -1.6146073; -71.1137085; -1018.271; 220.124268];
...
```

### Convert Filename Format

You should change `dir_path` according to dataset.

```shell
python convert_LGE_fmt.py
```

## Build & Run

```shell
mkdir build && cd build
cmake ..
make -j
cd ..
sh run_voxel_mapper.sh
```

## Running Options 

```shell
# run_voxel_mapper.sh
./build/voxel_mapper \
 -img_file_fmt /DATASET_PATH/data/img/img%04d_0.png \
 -depth_file_fmt /DATASET_PATH/data/depth/img%04d_0.depth \
 -trajectory_txt_path /DATASET_PATH/info/groundtruth.txt \
 -intrinsic_yaml_path /DATASET_PATH/intrinsic.yaml \
 --depth_is_z=true \		# false if depth values are length of ray, almost true, false in synthetic dataset(ex. Multi-FoV)
 --save_pc=false \		# save point cloud as .ply file
 --save_mesh=true \		# save mesh as .ply file
 --evaluate_mesh=false \	# evaluate mesh using KD-Tree
 --visualize_pc=false \		# visualize point cloud in pangolin viewer (for live)
 --visualize_mesh=false \	# visualize point cloud in pangolin viewer (for live)
 -n_voxels_per_side 8 \		# number of voxels per side of block
 -voxel_size 0.02 \		# voxel size (m)
 -truncated_ratio 3 \		# trucated distance = voxel size * truncated ratio
 -max_update_distance 5 \	# max update distance when integrating voxel grid map
 -raycast_mode simple \		# simple(naive) / grouped
 --carving_on=false \		# carving option
 --count_updated_voxels=false \ # counting updated voxel when integrating voxel grid map
 -logtostderr -colorlogtostderr

```



## Evaluation

### 1. Synthetic Dataset ( urban, vfr ) Mesh accuracy evaluation

#### settings

change directory in `[project_dir]/src/eval_mesh_accuracy_2022.cc`

```shell
const string urban_prefix = "/home/sangheonyang/HDDdata/LG22/results/result_plys/05_Multi-fov_urban_1216_mesh_eval/";
const string vfr_prefix = "/home/sangheonyang/HDDdata/LG22/results/result_plys/06_Multi-fov_vfr_1216_mesh_eval/";
```

#### run evaluation

```shell
$ cd [project_dir]/build/
$ build
$ cd [project_dir]/build/./evaluator
```

### 2. All Dataset Mesh accuracy (RMSE) evaluation

#### settings

change directory in `[project_dir]/src/eval_plane_accuracy_mesh.cc`

```shell
const string prefix = "/home/sangheonyang/HDDdata/LG22/results/result_plys/1216_results_all_SH/";
```

#### run evaluation

```shell
# after cmake and build
$ cd [project_dir]/build/
$ build
$ cd [project_dir]/build/./evaluator_plane_mesh
```

### 3. Scannet_evaluation (per-Plane-Recall)
 
#### reference
Planenet : https://github.com/art-programmer/PlaneNet  
PlaneRCNN : https://github.com/NVlabs/planercnn

#### Download Scannet evaluation Dataset and result of project
consist of 5000 dataset  
Download Dataset into : `[project_dir]/scannet_evaluation_mb/`

#### Scannet evaluation Dataset Structure & Format
```shell
.
└── [project_dir]/scannet_evaluation_mb
    ├── depth
    │   ├── depth
    │   │   ├── img0000_0.depth
    │   │   ├── img0001_0.depth
    │   │   └── ...
    │   └── depth_img
    │       ├── img0000_0.png
    │       ├── img0001_0.png
    │       └── ...
    ├── info
    │   └── intrinsic0000.yaml
    │   └── intrinsic0001.yaml
    │   └── ...
    ├── mask
    │   ├── gtSegmentation
    │   │   ├── geSeg_npy0000.npy
    │   │   ├── geSeg_npy0001.npy
    │   │   └── ...
    │   └── predSegmentation
    │       ├── res0000.txt
    │       ├── res0001.txt
    │       └── ...
    ├── image
    │   └── img0000_0.png
    │   └── img0001_0.png
    │   └── ...
    └── evaluation_plan_recall.py
```

#### Run Evaluation
```shell
$ python `[project_dir]/scannet_evaluation_mb/evaluate_plane_recall.py`
```  

### 4. create plane label for plane_recall evaluation  

#### settings

change directory in `[project_dir]/src/eval_plane_recall.cc`

```shell
// ---- NOTE : please assign appropriate path to files below -------------------------------//////////////////////
std::string yaml_path= "/[project_dir]/scannet_evaluation_mb/info/intrinsic%04d.yaml";
std::string depth_path= "/[project_dir]/scannet_evaluation_mb/depth/img%04d_0.depth";
std::string rgb_path= "/[project_dir]/scannet_evaluation_mb/scannet_evaluation_mb/image/img%04d_0.png";
std::string output_path = "/[project_dir]/scannet_evaluation_mb/mask/new_predSegmentation/res%04d.txt";
std::string outimg_path = "/[project_dir]/make_dir/result_img/";
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
```

#### run creating plane label 

```shell
# after cmake and build
$ cd [project_dir]/build/
$ build
$ cd [project_dir]/build/./evaluator_plane_recall
```

and rename `new_predSegmentation` as `pred_Segmentation`


## Example
 

