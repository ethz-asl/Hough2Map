#ifndef HOUGH2MAP_DETECTOR_H_
#define HOUGH2MAP_DETECTOR_H_

#include <ros/ros.h>

#include <custom_msgs/orientationEstimate.h>
#include <custom_msgs/positionEstimate.h>
#include <custom_msgs/velocityEstimate.h>
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>

#include "image_transport/image_transport.h"

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/convert.h>
#include <tf2/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_eigen/tf2_eigen.h>

#include <Eigen/Dense>
#include <boost/align/aligned_allocator.hpp>
#include <cmath>
#include <cv_bridge/cv_bridge.h>
#include <deque>
#include <fstream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include <omp.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

namespace hough2map {
class Detector {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Detector(const ros::NodeHandle &nh, const ros::NodeHandle &nh_private, const image_transport::ImageTransport &img_pipe);
  virtual ~Detector();

protected:
  std::string detector_name_ = "Hough";

private:
  // Struct for describing a detected line.
  struct line {
    int ID;
    int r;
    float theta;
    int theta_idx;
    double time;
    bool polarity;
  };

  // Struct for describing a tracked pole.
  struct pole {
    double rho;
    double theta;
    double t_enter;
    double t_leave;
    double t_center;
    double speed;
    int ID;
    bool polarity;
    double pos_x;
    double pos_y;
    double first_observed;
    float weight;
  };

  // Specifying number of threads.
  static const int kNumThreads = 4;

  // File Output.
  const bool file_output_parameter_logging = true;

  std::ofstream lines_file;
  std::ofstream map_file;

  // Timing debugging.
  double total_events_timing_us;
  double total_msgs_timing_ms;
  uint64_t total_events;
  uint64_t total_msgs;

  // General Parameters for 1st Hough Transform.
  static const int kHough1RadiusResolution = 660;
  static const int kHough1AngularResolution = 21;
  static const int kHough1MinAngle = -10;
  static const int kHough1MaxAngle = 10;

  // Precomputing possible angles and their sin/cos values in order to vectorize
  // the HT.
  Eigen::VectorXf thetas_1_;
  Eigen::MatrixXf polar_param_mapping_1_;

  // Hough transform objects.
  typedef Eigen::Matrix<int, kHough1RadiusResolution, kHough1AngularResolution>
      MatrixHough;

  // General parameters for 2nd Hough transform.
  static const int kHough2AngularResolution = 65;
  static const int kHough2MinAngle = 1;
  static const int kHough2MaxAngle = 65;
  Eigen::VectorXd thetas_2_;
  Eigen::MatrixXd polar_param_mapping_2_;

  static const int kHough2TimestepsPerMsg = 3;
  static const int kHough2MsgPerWindow = 100;

  int camera_resolution_width_;
  int camera_resolution_height_;

  const float kAcceptableDistortionRange = 40.0;
  float intrinsics_[4];
  float distortion_coeffs_[4];
  Eigen::Affine3d T_cam_to_body_;
  Eigen::MatrixXf undist_map_x_;
  Eigen::MatrixXf undist_map_y_;

  // Viz Helpers

  image_transport::ImageTransport img_pipe_;
  visualization_msgs::Marker pole_marker_;
  visualization_msgs::Marker cam_marker_;
  int pole_count_;
  nav_msgs::Path pose_buffer_path_;

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
  image_transport::Publisher hough2_img_pub_;

  // Function definitions.

  // Callback functions for subscribers.
  void eventCallback(const dvs_msgs::EventArray::ConstPtr &msg);  
  void poseCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr &msg);

  // Functions for Hough transform computation.
  hough2map::Detector::line addMaxima(int angle, int rad, double time,
                                      bool pol);
  void addMaximaInRadius(int i, int radius,
                         const MatrixHough &total_hough_space,
                         int local_threshold, bool polarity, double timestamp,
                         std::vector<hough2map::Detector::line> *new_maxima,
                         std::vector<int> *new_maxima_value,
                         bool skip_center = false);
  void applySuppressionRadius(
      const std::vector<hough2map::Detector::line> &new_maxima,
      const std::vector<int> &new_maxima_value,
      std::vector<hough2map::Detector::line> *current_maxima);
  template <typename DerivedVec, typename DerivedMat>
  void initializeSinCosMap(Eigen::EigenBase<DerivedVec> &angles,
                           Eigen::EigenBase<DerivedMat> &sin_cos_map,
                           const int kMinAngle, const int kMaxAngle,
                           const int kNumSteps);
  bool isLocalMaxima(const Eigen::MatrixXi &hough_space, int i, int radius);
  void newPoleDetection(double rho, double theta, double window_time, bool pol);
  void hough2nms(const int i, const int j, const Eigen::MatrixXi &hough_2_space,
                 std::vector<cv::Vec3f> &detections);
  void computeFullHoughTransform(const int time_step,
                                 const int nms_recompute_window,
                                 MatrixHough &total_hough_space_pos,
                                 MatrixHough &total_hough_space_neg,
                                 const Eigen::MatrixXi &radii);
  void updateHoughSpaceVotes(const bool increment, const int event_idx,
                             const bool pol, const Eigen::MatrixXi &radii,
                             MatrixHough &hough_space_pos,
                             MatrixHough &hough_space_neg);
  void computeFullNMS(
      const int time_step, const int nms_recompute_window,
      const MatrixHough &total_hough_space_pos,
      const MatrixHough &total_hough_space_neg,
      std::vector<std::vector<hough2map::Detector::line>> &cur_maxima_list);
  void itterativeNMS(
      const int time_step, const int nms_recompute_window,
      MatrixHough &total_hough_space_pos, MatrixHough &total_hough_space_neg,
      std::vector<std::vector<hough2map::Detector::line>> &cur_maxima_list,
      const Eigen::MatrixXi &radii);
  bool updateIncrementedNMS(
      const double kTimestamp, const bool polarity, const int kAngle,
      const int kRadius, const MatrixHough &hough_space,
      const std::vector<hough2map::Detector::line> &previous_maxima,
      std::vector<int> &discard,
      std::vector<hough2map::Detector::line> &new_maxima,
      std::vector<int> &new_maxima_value);
  void updateDecrementedNMS(
      const double kTimestamp, const bool polarity, const int kAngle,
      const int kRadius, const MatrixHough &hough_space,
      const std::vector<hough2map::Detector::line> &previous_maxima,
      std::vector<int> &discard,
      std::vector<hough2map::Detector::line> &new_maxima,
      std::vector<int> &new_maxima_value);
  void
  secondHoughTransform(const std::vector<std::vector<hough2map::Detector::line>>
                           &cur_maxima_list);
  void eventPreProcessing(const dvs_msgs::EventArray::ConstPtr &orig_msg,
                          Eigen::MatrixXf &points);

  // Initialisation functions.
  void computeUndistortionMapping();
  void loadCalibration();

  geometry_msgs::PoseWithCovarianceStamped queryPoseAtTime(const double query_time);

  // Visualization functions.
  void drawPolarCorLine(cv::Mat &image_space, float rho, float theta,
                        cv::Scalar color);
  void imageCallback(const sensor_msgs::Image::ConstPtr &msg);
  void visualizeSecondHoughSpace(const std::vector<cv::Vec3f> &kDetectionsPos,
                                 const std::vector<cv::Vec3f> &kDetectionsNeg);
  void visualizeCurrentLineDetections(
      const std::vector<std::vector<hough2map::Detector::line>>
          &cur_maxima_list);

  std::deque<
      Eigen::Matrix<int, kHough1RadiusResolution, kHough2TimestepsPerMsg>>
      hough2_queue_pos_;
  std::deque<
      Eigen::Matrix<int, kHough1RadiusResolution, kHough2TimestepsPerMsg>>
      hough2_queue_neg_;

  // Events from the previous dvs_msg need to be carried over to start of the
  // Hough computation of the next events. This basically ensures a continous
  // sliding window.
  dvs_msgs::EventArray feature_msg_;

  std::deque<geometry_msgs::PoseWithCovarianceStamped::ConstPtr> pose_buffer_;

  // Storing the current result of the non-max-suppression.
  cv::Mat cur_greyscale_img_;
};

} // namespace hough2map

#endif // HOUGH2MAP_DETECTOR_H_
