# High Speed Mapping with DVS

## Abstract

Mapping is an essential component of a navigation stack in an autonomous robot, more so in the context of autonomous road vehicles. The growing scale of autonomy necessitates exploration of lightweight techniques to map surroundings, with emphasis on reliable feature detection and robustness against ambient conditions. Given these constraints, we find Dynamic Vision Sensors (DVS) of particular interest as they offer higher dynamic range and virtually no motion blur in contrast to standard image camera.

Building upon it's precursor Hough²Map, we intend to take a geometric approach towards mapping by identifying vertical markers in the environment like poles and edges of structures to and mapping on to 2D plane. This pipeline is capable of identifying vertical markers, tracking them over time, triangulating these vertical features. We also evaluate the pipeline for reliability metrics across different lighting scenarios and also for performance across different sensor sizes. The current pipeline achieves feature re-association percentage of up to 46%.

## Install

### Setup ROS, catkin workspace and system dependencies

Install additional system dependency:

```bash
sudo apt install libomp-dev --yes
```

### Clone and build

Pay special attention to the install instructions of the [rpg_dvs_ros](https://github.com/uzh-rpg/rpg_dvs_ros) package, especially regarding setting the catkin build type:

```bash
cd ~/catkin_ws/src/
catkin config --merge-devel --cmake-args -DCMAKE_BUILD_TYPE=Release
git clone git@github.com:ethz-asl/Hough2Map.git --recursive
catkin build hough2map
```

### Run

Run using:

```bash
roslaunch hough2map hough2map.launch bag:="path/to/bag/file(s)" config:="config profiles found in config/" camera:="camera profiles found in config/cam/"
# For example
roslaunch hough2map hough2map.launch bag:="~/zurich_006.bag" config:="zurich_1" camera:="zurich_240_180"
```
