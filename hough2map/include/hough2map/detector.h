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

  // File Output.
  const char* map_file_path = "/tmp/map.txt";
  std::ofstream map_file;
  const char* debug_file_path = "/tmp/debug.txt";
  std::ofstream debug_file;

  const char* calibration_file_name = "calibration.yaml";

  // Timing debugging.
  double total_events_timing_us;
  double total_msgs_timing_ms;
  uint64_t total_events;
  uint64_t total_msgs;

  // General Parameters for 1st Hough Transform.
  static const int kHoughRadiusResolution = 240;
  static const int kHoughAngularResolution = 21;
  static const int kHoughMinAngle = -10;
  static const int kHoughMaxAngle = 10;

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

  // ROS interface.
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;
  ros::Subscriber event_sub_;
  ros::Subscriber image_raw_sub_;

  ros::Subscriber GPS_pos_;
  ros::Subscriber GPS_orient_;
  ros::Subscriber GPS_vel_;

  /* Function definitions. */

  // Callback functions for subscribers.
  void eventCallback(const dvs_msgs::EventArray::ConstPtr& msg);
  void positionCallback(const custom_msgs::positionEstimate msg);
  void velocityCallback(const custom_msgs::velocityEstimate msg);
  void orientationCallback(const custom_msgs::orientationEstimate msg);

  // Functions for Hough transform computation.
  void stepHoughTransform(
      const Eigen::MatrixXf& points, MatrixHough& hough_space,
      std::vector<hough2map::Detector::line> *last_maxima, bool initialized,
      std::vector<std::vector<hough2map::Detector::line>> *maxima_list);

  void addMaximaInRadius(
      int angle, int radius, const MatrixHough& hough_space,
      std::vector<hough2map::Detector::line>* new_maxima,
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
  void itterativeNMS(
      const Eigen::MatrixXf& points, MatrixHough& hough_space, 
      const Eigen::MatrixXi& radii,
      std::vector<std::vector<Detector::line>>* maxima_list);

  void eventPreProcessing(
      const dvs_msgs::EventArray::ConstPtr& orig_msg,
      Eigen::MatrixXf& points_pos, Eigen::MatrixXf& points_neg);

  // Initialisation functions.
  void computeUndistortionMapping();
  void initializeTransformationMatrices();
  void loadCalibration();

  // Odometry processing functions.
  utm_coordinate deg2utm(double la, double lo);
  template <class S, int rows, int cols>
  Eigen::Matrix<S, rows, cols> queryOdometryBuffer(
      const double query_time,
      const std::deque<Eigen::Matrix<S, rows, cols>>& odometry_buffer);

  // Visualization functions.
  void drawPolarCorLine(
      cv::Mat& image_space, float rho, float theta, cv::Scalar color) const;
  void imageCallback(const sensor_msgs::Image::ConstPtr& msg);
  void visualizeCurrentLineDetections(
      bool polarity, const Eigen::MatrixXf& points, 
      const std::vector<std::vector<Detector::line>>& maxima_list,
      const MatrixHough& hough_space) const;

  // For the very first message we need separate processing (e.g. a full HT).
  bool initialized;
  MatrixHough hough_space_pos;
  MatrixHough hough_space_neg;
  std::vector<hough2map::Detector::line> last_maxima_pos;
  std::vector<hough2map::Detector::line> last_maxima_neg;
  Eigen::MatrixXf last_points_pos;
  Eigen::MatrixXf last_points_neg;
  
  // Odometry.
  std::deque<Eigen::Vector3d> raw_gps_buffer_;
  std::deque<Eigen::Vector3d> velocity_buffer_;
  std::deque<Eigen::Vector2d> orientation_buffer_;

  // Transformation matrix (in [m]) between train and sensors for triangulation.
  Eigen::Matrix3d C_camera_train_;
  Eigen::Matrix3d gps_offset_;
  Eigen::Matrix3d camera_train_offset_;

  // Image to display the events on for visualization purposes only.
  cv::Mat cur_greyscale_img_;
};
}  // namespace hough2map

#endif  // HOUGH2MAP_DETECTOR_H_
