#ifndef HOUGH2MAP_DETECTOR_H_
#define HOUGH2MAP_DETECTOR_H_

#include <custom_msgs/orientationEstimate.h>
#include <custom_msgs/positionEstimate.h>
#include <custom_msgs/velocityEstimate.h>
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>
#include <geometry_msgs/PoseArray.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

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
  Detector(const ros::NodeHandle &nh, const ros::NodeHandle &nh_private);
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

  struct utm_coordinate {
    double x;
    double y;
    std::string zone;
  };

  // Specifying number of threads.
  static const int kNumThreads = 4;

  // File Output.
  const bool file_output_parameter_logging = true;

  const char *lines_file_path = "/tmp/lines.txt";
  std::ofstream lines_file;

  const char *map_file_path = "/tmp/map.txt";
  std::ofstream map_file;

  const char *calibration_file_name = "calibration.yaml";

  // Timing debugging.
  double total_events_timing_us;
  double total_msgs_timing_ms;
  uint64_t total_events;
  uint64_t total_msgs;

  // General Parameters for 1st Hough Transform.
  static const int kHough1RadiusResolution = 260;
  static const int kHough1AngularResolution = 21;
  static const int kHough1MinAngle = -10;
  static const int kHough1MaxAngle = 10;

  // Precomputing possible angles and their sin/cos values in order to vectorize
  // the HT.
  Eigen::VectorXf thetas_;
  Eigen::MatrixXf polar_param_mapping_;

  // Hough transform objects.
  typedef Eigen::Matrix<int, kHough1RadiusResolution, kHough1AngularResolution>
      MatrixHough;

  int camera_resolution_width_;
  int camera_resolution_height_;

  const float kAcceptableDistortionRange = 40.0;
  float intrinsics_[4];
  float distortion_coeffs_[4];
  Eigen::MatrixXf undist_map_x_;
  Eigen::MatrixXf undist_map_y_;

  // ROS interface.
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;
  ros::Publisher feature_pub_;
  ros::Subscriber event_sub_;
  ros::Subscriber image_raw_sub_;

  ros::Subscriber GPS_pos_;
  ros::Subscriber GPS_orient_;
  ros::Subscriber GPS_vel_;

  /* Function definitions. */

  // Callback functions for subscribers.
  void eventCallback(const dvs_msgs::EventArray::ConstPtr &msg);
  void positionCallback(const custom_msgs::positionEstimate msg);
  void velocityCallback(const custom_msgs::velocityEstimate msg);
  void orientationCallback(const custom_msgs::orientationEstimate msg);

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
  
  void eventPreProcessing(const dvs_msgs::EventArray::ConstPtr &orig_msg,
                          Eigen::MatrixXf &points);

  // Initialisation functions.
  void computeUndistortionMapping();
  void initializeTransformationMatrices();
  void loadCalibration();

  // Odometry processing functions.
  utm_coordinate deg2utm(double la, double lo);
  template <class S, int rows, int cols>
  Eigen::Matrix<S, rows, cols> queryOdometryBuffer(
      const double query_time,
      const std::deque<Eigen::Matrix<S, rows, cols>> &odometry_buffer);

  // Visualization functions.
  void drawPolarCorLine(cv::Mat &image_space, float rho, float theta,
                        cv::Scalar color);
  void imageCallback(const sensor_msgs::Image::ConstPtr &msg);
  void visualizeCurrentLineDetections(
      const std::vector<std::vector<hough2map::Detector::line>>
          &cur_maxima_list);

  // Events from the previous dvs_msg need to be carried over to start of the
  // Hough computation of the next events. This basically ensures a continous
  // sliding window.
  dvs_msgs::EventArray feature_msg_;

  // Odometry.
  std::deque<Eigen::Vector3d> raw_gps_buffer_;
  std::deque<Eigen::Vector3d> velocity_buffer_;
  std::deque<Eigen::Vector2d> orientation_buffer_;

  // Transformation matrix (in [m]) between train and sensors for triangulation.
  Eigen::Matrix3d C_camera_train_;
  Eigen::Matrix3d gps_offset_;
  Eigen::Matrix3d camera_train_offset_;

  // Storing the current result of the non-max-suppression.
  cv::Mat cur_greyscale_img_;
};
} // namespace hough2map

#endif // HOUGH2MAP_DETECTOR_H_
