#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <maplab-ros-common/gflags-interface.h>
#include <opencv2/core/persistence.hpp>
#include <ros/ros.h>
#include <ros/package.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <algorithm>
#include <fstream>
#include <numeric>
#include <vector>

DEFINE_string(rosbag_path, "", "Rosbag to process.");
DEFINE_string(event_topic, "/dvs/events", "Topic for event messages.");
DEFINE_string(
    dvs_calibration, "calibration.yaml", "Camera parameters for the DVS.");
DEFINE_double(
    trigger_threshold, 0.75, "Trigger threshold at which to declare it dead.");

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InstallFailureSignalHandler();
  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;

  ros::init(argc, argv, "calibrate");
  ros::NodeHandle nh_private("~");

  // Reading RosParams and storing as gflags.
  ros_common::parseGflagsFromRosParams(argv[0], nh_private);
  ROS_INFO("Calibrating dead pixels in DVS");

  // Read camera params to get resolution.
  std::string package_path = ros::package::getPath("hough2map");
  std::string calibration_file =
      package_path + "/share/" + FLAGS_dvs_calibration;

  cv::FileStorage fs(calibration_file, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    LOG(FATAL) << "Could not open calibration file:" << calibration_file
               << std::endl;
  }

  cv::FileNode cam = fs["cam0"];
  cv::FileNode resolution = cam["resolution"];
  CHECK_EQ(resolution.size(), 2)
      << ": Not enough calibration data regarding sensor size!";
  int32_t image_width_ = resolution[0];
  int32_t image_height_ = resolution[1];

  // Open rosbag with calibration data.
  rosbag::Bag bag;
  try {
    LOG(INFO) << "Opening bag: " << FLAGS_rosbag_path << ".";
    bag.open(FLAGS_rosbag_path, rosbag::bagmode::Read);
  } catch (const std::exception& ex) {  // NOLINT
    LOG(FATAL) << "Could not open the rosbag " << FLAGS_rosbag_path << ": "
               << ex.what();
  }

  // Count events that overtrigger.
  const size_t num_pixels = image_width_ * image_height_;
  std::vector<int64_t> event_counter(num_pixels, 0);

  // Iterate over bag taking only the dvs events topic.
  size_t num_messages = 0;
  std::vector<std::string> topics;
  topics.emplace_back(FLAGS_event_topic);

  rosbag::View bag_view(bag, rosbag::TopicQuery(topics));
  rosbag::View::iterator it_message = bag_view.begin();
  while (it_message != bag_view.end()) {
    dvs_msgs::EventArray::ConstPtr event_message =
        it_message->instantiate<dvs_msgs::EventArray>();
    CHECK(event_message);

    std::vector<bool> event_triggered(num_pixels, false);

    // Count how many times each event was triggered.
    for (const dvs_msgs::Event& e : event_message->events) {
      size_t index = e.x * image_height_ + e.y;
      if (!event_triggered[index]) {
        ++event_counter[index];
        event_triggered[index] = true;
      }
    }

    ++it_message;
    ++num_messages;
  }

  bag.close();

  // Sort pixels by events occurence.
  std::vector<size_t> indices(event_counter.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::stable_sort(indices.begin(), indices.end(),
    [&event_counter](size_t i1, size_t i2) {
      return event_counter[i1] > event_counter[i2];
  });

  // Check how many events got triggered with the lens cap on.
  const size_t message_threshold = FLAGS_trigger_threshold * num_messages;

  size_t num_bad = 0;
  while (num_bad < num_pixels) {
    if (event_counter[indices[num_bad]] < message_threshold) {
      break;
    }
    ++num_bad;
  }
  LOG(INFO) << "Found " << num_bad << " bad pixels.";

  // Write to calibration file.
  const char* pixels_file_path = "/tmp/pixels.txt";
  std::ofstream pixels_file(pixels_file_path);
  CHECK(pixels_file.is_open());

  pixels_file << num_bad << std::endl;
  for (size_t i = 0; i < num_bad; ++i) {
    size_t x = indices[i] / image_height_;
    size_t y = indices[i] % image_height_;
    pixels_file << x << "," << y << std::endl;
  }
  pixels_file.close();

  return 0;
}
