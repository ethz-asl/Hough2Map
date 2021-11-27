#ifndef HOUGH2MAP_DETECTOR_H_
#define HOUGH2MAP_DETECTOR_H_

#include <cmath>
#include <deque>
#include <fstream>
#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <omp.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <ros/ros.h>

#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/Quaternion.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <visualization_msgs/Marker.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/convert.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include "hough2map/structs.h"
#include "hough2map/tracker.h"
#include "hough2map/config.h"

namespace hough2map {
class Detector {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Detector(const ros::NodeHandle &nh, const ros::NodeHandle &nh_private,
           const image_transport::ImageTransport &img_pipe);
  virtual ~Detector();

 protected:
  std::string detector_name_ = "Hough";

 private:
  // Specifying number of threads.
  static const int kNumThreads = 4;

  // std::ofstream lines_file;
  std::ofstream map_file_;

  // Timing debugging.
  ProfilingInfo profiling_;

  // Configs
  Hough1Config hough1_config_;
  DetectorConfig detector_config_;
  CameraConfig cam_config_;
  OutputConfig output_config_;

  // Precomputing possible angles and their sin/cos values in order to vectorize
  // the HT.
  Eigen::VectorXf thetas_1_;
  Eigen::MatrixXf polar_param_mapping_1_;

  // Only updated maximas that must be cleared whenever read
  std::vector<HoughLine> maxima_updates_;

  // Viz Helpers

  image_transport::ImageTransport img_pipe_;
  visualization_msgs::Marker pole_marker_;
  visualization_msgs::Marker cam_marker_;
  int pole_count_;
  nav_msgs::Path pose_buffer_path_;
  std::deque<Tracker> viz_trackers_;

  // Tracker List
  TrackerManagerConfig tracker_mgr_config_;
  TrackerManager tracker_mgr_;

  std::deque<std::vector<int>> cluster_centroids_;

  // ROS interface.

  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  ros::Subscriber event_sub_;
  ros::Subscriber odom_pose_sub_;
  ros::Subscriber image_raw_sub_;

  ros::Publisher feature_pub_;
  ros::Publisher pole_viz_pub_;
  ros::Publisher cam_viz_pub_;
  ros::Publisher pose_buffer_pub_;

  image_transport::Publisher hough1_img_pub_;
  image_transport::Publisher xt_img_pub_;

  // Function definitions.

  // Callback functions for subscribers.
  void eventCallback(const dvs_msgs::EventArray::ConstPtr &msg);
  void poseCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr &msg);

  // Functions for Hough transform computation.
  hough2map::HoughLine addMaxima(int angle, int rad, double time);
  void addMaximaInRadius(int i, int radius, const Eigen::MatrixXi &total_hough_space,
                         int local_threshold, double timestamp,
                         std::vector<hough2map::HoughLine> *new_maxima,
                         std::vector<int> *new_maxima_value, bool skip_center = false);
  void applySuppressionRadius(const std::vector<hough2map::HoughLine> &new_maxima,
                              const std::vector<int> &new_maxima_value,
                              std::vector<hough2map::HoughLine> *current_maxima);
  template <typename DerivedVec, typename DerivedMat>
  void initializeSinCosMap(Eigen::EigenBase<DerivedVec> &angles,
                           Eigen::EigenBase<DerivedMat> &sin_cos_map, const int kMinAngle,
                           const int kMaxAngle, const int kNumSteps);
  bool isLocalMaxima(const Eigen::MatrixXi &hough_space, int i, int radius);
  void computeFullHoughTransform(const int time_step, const int nms_recompute_window,
                                 Eigen::MatrixXi &total_hough_space_neg,
                                 const Eigen::MatrixXi &radii);
  void updateHoughSpaceVotes(const bool increment, const int event_idx,
                             const Eigen::MatrixXi &radii, Eigen::MatrixXi &hough_space_neg);
  void computeFullNMS(const int time_step, const int nms_recompute_window,
                      const Eigen::MatrixXi &total_hough_space_neg,
                      std::vector<std::vector<hough2map::HoughLine>> &cur_maxima_list);
  void itterativeNMS(const int time_step, const int nms_recompute_window,
                     Eigen::MatrixXi &total_hough_space_neg,
                     std::vector<std::vector<hough2map::HoughLine>> &cur_maxima_list,
                     const Eigen::MatrixXi &radii);
  bool updateIncrementedNMS(const double kTimestamp, const int kAngle, const int kRadius,
                            const Eigen::MatrixXi &hough_space,
                            const std::vector<hough2map::HoughLine> &previous_maxima,
                            std::vector<int> &discard,
                            std::vector<hough2map::HoughLine> &new_maxima,
                            std::vector<int> &new_maxima_value);
  void updateDecrementedNMS(const double kTimestamp, const int kAngle, const int kRadius,
                            const Eigen::MatrixXi &hough_space,
                            const std::vector<hough2map::HoughLine> &previous_maxima,
                            std::vector<int> &discard,
                            std::vector<hough2map::HoughLine> &new_maxima,
                            std::vector<int> &new_maxima_value);
  void eventPreProcessing(const dvs_msgs::EventArray::ConstPtr &orig_msg, Eigen::MatrixXf &points);

  // Tracker
  void heuristicTrack(const std::vector<std::vector<hough2map::HoughLine>> &cur_maxima_list);
  void triangulateTracker(Tracker tracker);
  std::vector<int> getClusteringCentroids(Eigen::VectorXi detections);

  // Initialisation functions.
  void computeUndistortionMapping();

  void loadConfigFromParams();
  void loadCamConfigFromParams();

  geometry_msgs::PoseWithCovarianceStamped queryPoseAtTime(const double query_time);

  // Visualization functions.
  void drawPolarCorLine(cv::Mat &image_space, float rho, float theta, cv::Scalar color);
  void imageCallback(const sensor_msgs::Image::ConstPtr &msg);
  // void visualizeSecondHoughSpace(const std::vector<cv::Vec3f> &kDetectionsNeg);
  void visualizeCurrentLineDetections(
      const std::vector<std::vector<hough2map::HoughLine>> &cur_maxima_list);

  void visualizeTracker();

  std::deque<Eigen::MatrixXi> houghout_queue_;
  double houghout_queue_last_t;

  // Events from the previous dvs_msg need to be carried over to start of the
  // Hough computation of the next events. This basically ensures a continous
  // sliding window.
  dvs_msgs::EventArray feature_msg_;

  std::deque<geometry_msgs::PoseWithCovarianceStamped::ConstPtr> pose_buffer_;

  // Storing the current result of the non-max-suppression.
  cv::Mat cur_greyscale_img_;
};

}  // namespace hough2map

#endif  // HOUGH2MAP_DETECTOR_H_
