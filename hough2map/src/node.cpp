#include <ros/ros.h>

#include "hough2map/detector.h"

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;

  ros::init(argc, argv, "hough2map");
  ros::NodeHandle nh, nh_private("~");

  image_transport::ImageTransport img_pipe(nh);

  // Create pole detector.
  ROS_INFO("Using Hough Detector");

  hough2map::Detector detector(nh, nh_private, img_pipe);

  ros::spin();

  return 0;
}
