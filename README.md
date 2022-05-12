# Hough²Map
Iterative Event-based Hough Transform for High-Speed Railway Mapping

Ubuntu 20.04+ROS noetic [![Build Status](https://jenkins.asl.ethz.ch/buildStatus/icon?job=Hough2Map)](https://jenkins.asl.ethz.ch/job/Hough2Map/)

## Abstract

To cope with the growing demand for transportation on the railway system, accurate, robust, and high-frequency positioning is required to enable a safe and efficient utilization of the existing railway infrastructure.
As a basis for a localization system we propose a complete on-board mapping pipeline able to map robust meaningful landmarks, such as poles from power lines, in the vicinity of the vehicle.
Such poles are good candidates for reliable and long term landmarks even through difficult weather conditions or seasonal changes.
To address the challenges of motion blur and illumination changes in railway scenarios we employ a DVS, a novel event-based camera.
Using a sideways oriented on-board camera, poles appear as vertical lines.
To map such lines in a real-time event stream, we introduce **Hough²Map**, a novel consecutive iterative event-based Hough transform framework capable of detecting, tracking, and triangulating close-by structures.
We demonstrate the mapping reliability and accuracy of **Hough²Map** on real-world data in typical usage scenarios and evaluate using surveyed infrastructure ground truth maps.
**Hough²Map** achieves a detection reliability of up to 92% and a mapping root mean square error accuracy of 1.1518m.




## Paper and Video

The **Hough²Map** pipeline is described in the following publication:

- Florian Tschopp, Cornelius von Einem, Andrei Cramariuc, David Hug, Andrew William Palmer, Roland Siegwart, Margarita Chli, Juan Nieto, **Hough²Map – Iterative Event-based Hough Transform for High-Speed Railway Mapping**, in _IEEE Robotics and Automation Letters_, April 2021. [[PDF](https://arxiv.org/pdf/2102.08145.pdf)] [[Video](https://www.youtube.com/watch?v=YPSiODVzD-I)]


```bibtex
@ARTICLE{Tschopp2021Hough2Map,  
  author={Tschopp, Florian and von Einem, Cornelius and Cramariuc, Andrei and Hug, David and Palmer, Andrew William and Siegwart, Roland and Chli, Margarita and Nieto, Juan},  
  journal={IEEE Robotics and Automation Letters},   
  title={Hough$^2$Map – Iterative Event-Based Hough Transform for High-Speed Railway Mapping},   
  year={2021},  
  volume={6},  
  number={2},  
  pages={2745-2752},  
  doi={10.1109/LRA.2021.3061404}
}
```

Please also have a look at our video:

[![Hough²Map Youtube Video](http://img.youtube.com/vi/YPSiODVzD-I/0.jpg)](http://www.youtube.com/watch?v=YPSiODVzD-I)

## Install

### Setup ROS, catkin workspace and system dependencies
Refer to the [install instructions](https://maplab.asl.ethz.ch/docs/develop/pages/installation/A_Installation-Ubuntu.html#manual-installation) of maplab up to the cloning and building of maplab itself.

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
```
roslaunch hough2map hough2map.launch bag:="path to bag file"
```
