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
#include <cmath>
#include <cv_bridge/cv_bridge.h>
#include <deque>
#include <queue>
#include <map>
#include <fstream>
#include <sstream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

namespace hough2map {
class Detector {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Detector(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private);
  virtual ~Detector();

 protected:
  std::string detector_name_ = "Hough";

 private:
  // Struct for describing a detected line.
  struct line {
    line(int _r, float _theta, int _theta_idx) :
        r(_r), theta(_theta), theta_idx(_theta_idx) {}

    bool operator==(const line& other) const {
      return (r == other.r) && (theta_idx == other.theta_idx);
    }

    bool operator!=(const line& other) const {
      return !operator==(other);
    }

    bool operator<(const line& other) const {
        return (r != other.r) ? r < other.r : theta_idx < other.theta_idx;
    }

    int r;
    float theta;
    int theta_idx;
  };

  // File Output.
  const char* map_file_path = "/tmp/map.txt";
  std::ofstream map_file;
  const char* debug_file_path = "/tmp/debug.txt";
  std::ofstream debug_file;

  // Timing debugging.
  double total_events_timing_us;
  double total_tracking_timing_ms;
  double total_msgs_timing_ms;
  uint64_t total_events;
  uint64_t total_msgs;

  // General Parameters for 1st Hough Transform.
  static const int kHoughRadiusResolution = 640;
  static const int kHoughAngularResolution = 21;
  static const int kHoughMinAngle = -5;
  static const int kHoughMaxAngle = 5;

  // Precomputing possible angles and their sin/cos values in order to vectorize
  // the HT. Also precomputing the squared suppression radius.
  Eigen::VectorXf thetas_;
  Eigen::MatrixXf polar_param_mapping_;
  int hough_nms_radius2_;

  // Hough transform objects.
  typedef Eigen::Matrix<int, kHoughRadiusResolution, kHoughAngularResolution>
      MatrixHough;

  int camera_resolution_width_;
  int camera_resolution_height_;

  float intrinsics_[4];
  float distortion_coeffs_[4];

  cv::Mat image_undist_map_x_;
  cv::Mat image_undist_map_y_;
  Eigen::MatrixXf event_undist_map_x_;
  Eigen::MatrixXf event_undist_map_y_;

  bool filter_dead_pixels_;
  std::vector<bool> is_dead_pixel_;

  // ROS interface.
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;
  ros::Subscriber event_sub_;
  ros::Subscriber image_raw_sub_;

  // Callback functions for subscribers.
  void eventCallback(const dvs_msgs::EventArray::ConstPtr& msg);
  void imageCallback(const sensor_msgs::Image::ConstPtr& msg);

  // Functions for Hough transform computation.
  void stepHoughTransform(
      const Eigen::MatrixXf& points, MatrixHough& hough_space,
      std::vector<Detector::line> *last_maxima, bool initialized,
      std::vector<std::vector<Detector::line>>* maxima_list,
      std::vector<size_t>* maxima_change);

  void addMaximaInRadius(
      int angle, int radius, const MatrixHough& hough_space,
      std::vector<Detector::line>* new_maxima,
      std::vector<int>* new_maxima_value, bool skip_center = false);
  void applySuppressionRadius(
      const std::vector<Detector::line>& candidate_maxima,
      const std::vector<int>& candidate_maxima_values,
      std::vector<Detector::line>* maxima);
  template <typename DerivedVec, typename DerivedMat>
  void initializeSinCosMap(
      Eigen::EigenBase<DerivedVec>& angles,
      Eigen::EigenBase<DerivedMat>& sin_cos_map, const int kMinAngle,
      const int kMaxAngle, const int kNumSteps);
  bool isLocalMaxima(const Eigen::MatrixXi& hough_space, int i, int radius);
  void newPoleDetection(double rho, double theta, double window_time, bool pol);

  void computeFullHoughSpace(
      size_t index, MatrixHough& hough_space, const Eigen::MatrixXi& radii);

  void computeFullNMS(
      const MatrixHough& hough_space, std::vector<Detector::line> *maxima);
  void iterativeNMS(
      const Eigen::MatrixXf& points, MatrixHough& hough_space, 
      const Eigen::MatrixXi& radii,
      std::vector<std::vector<Detector::line>>* maxima_list,
      std::vector<size_t>* maxima_change);

  void eventPreProcessing(
      const dvs_msgs::EventArray::ConstPtr& orig_msg,
      Eigen::MatrixXf* points, Eigen::VectorXd* times);

  // Initialisation functions.
  void computeUndistortionMapping();
  void loadCalibration();

  // Odometry processing functions.
  template <class S, int rows, int cols>
  Eigen::Matrix<S, rows, cols> queryOdometryBuffer(
      const double query_time,
      const std::deque<Eigen::Matrix<S, rows, cols>>& odometry_buffer);

  // Visualization functions.
  void drawPolarCorLine(
      cv::Mat& image_space, float rho, float  theta, cv::Scalar color) const;
  void visualizeCurrentLineDetections(
      const Eigen::MatrixXf& points,
      const std::vector<std::vector<Detector::line>>& maxima_list,
      const std::vector<size_t>& maxima_change) const;

  // For the very first message we need separate processing (e.g. a full HT).
  bool initialized;
  MatrixHough hough_space;
  std::vector<Detector::line> last_maxima;
  Eigen::MatrixXf last_points;
  Eigen::VectorXd last_times;

  // Image to display the events on for visualization purposes only.
  cv::Mat cur_greyscale_img_;

  // Tracking parameters
  int32_t tracking_angle_step;
  int32_t tracking_pixel_step;
  double tracking_max_jump;
  size_t tracking_mean_window;

  struct track_point {
    double pixel;
    double angle;
    double time;

    track_point() : pixel(-1.0), angle(-1.0), time(-1.0) {}

    track_point(double _pixel, double _angle, double _time) :
        pixel(_pixel), angle(_angle), time(_time) {}
  };

  size_t next_track_id;
  std::map<size_t, std::vector<track_point>> tracks;

  double start_time;
};
}  // namespace hough2map

#endif  // HOUGH2MAP_DETECTOR_H_
