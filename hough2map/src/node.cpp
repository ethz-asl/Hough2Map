#include "hough2map/detector.h"
#include <maplab-ros-common/gflags-interface.h>
#include <ros/ros.h>

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InstallFailureSignalHandler();
  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;

  ros::init(argc, argv, "hough2map");
  ros::NodeHandle nh, nh_private("~");

  // Reading RosParams and storing as gflags.
  ros_common::parseGflagsFromRosParams(argv[0], nh_private);

  // Create pole detector.
  ROS_INFO("Using Hough detector.");

  hough2map::Detector detector(nh, nh_private);

  return 0;
}
