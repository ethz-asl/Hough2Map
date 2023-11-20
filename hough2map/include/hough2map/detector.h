#ifndef HOUGH2MAP_DETECTOR_H_
#define HOUGH2MAP_DETECTOR_H_

#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <Eigen/Dense>
#include <cv_bridge/cv_bridge.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cmath>
#include <cstring>
#include <deque>
#include <iostream>
#include <fstream>
#include <sstream>
#include <queue>
#include <map>
#include <vector>

// General Parameters
constexpr int32_t kHoughSpaceHeight = 480 + 2;
constexpr int32_t kHoughSpaceWidth = 640 + 2;

constexpr int32_t kHoughMinRadius = 8;
constexpr int32_t kHoughMaxRadius = 14;
constexpr int32_t kHoughSpaceRadius = 
        (kHoughMaxRadius - kHoughMinRadius + 1) + 2;

constexpr size_t kMaxTimeToLive = 100;
constexpr size_t kMaxEventRate = 10 * kMaxTimeToLive;

namespace hough2map {
class Detector {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Detector(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private);
  virtual ~Detector();

 protected:
  std::string detector_name_ = "Hough";

 private:
  // Struct for a 2D point
  struct point {
    point() : x(0), y(0) {}
    point(int32_t _x, int32_t _y) : x(_x), y(_y) {}
    int32_t x, y;
  };

  // Struct for describing a detected circle.
  struct circle {
    circle(int32_t _x, int32_t _y, int32_t _r) :
        x(_x), y(_y), r(_r) {}

    bool operator==(const circle& other) const {
      return (x == other.x) && (y == other.y) && (r == other.r);
    }

    bool operator!=(const circle& other) const {
      return !operator==(other);
    }

    bool operator<(const circle& other) const {
        if (r != other.r) {
            return r < other.r;
        } else if (x != other.x) {
            return x < other.x;
        } else {
            return y < other.y;
        }
    }

    bool operator>(const circle& other) const {
        if (r != other.r) {
            return r > other.r;
        } else if (x != other.x) {
            return x > other.x;
        } else {
            return y > other.y;
        }
    }

    int32_t x, y, r;
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

  // Precomputing the squared suppression radius.
  int hough_nms_radius3_;

  // Precomputing the square pixel deviations.  
  std::vector<std::vector<point>> circle_xy;

  // Hough transform objects.
  typedef size_t HoughMatrix[
        kHoughSpaceRadius][kHoughSpaceHeight][kHoughSpaceWidth];
  typedef size_t (*HoughMatrixPtr)[kHoughSpaceHeight][kHoughSpaceWidth];

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
      const std::vector<point>& points,
      HoughMatrixPtr hough_space, std::vector<circle> *last_maxima,
      bool initialized, std::vector<std::vector<circle>>* maxima_list,
      std::vector<size_t>* maxima_change);

  void addMaximaInRadius(
      int32_t r, int32_t y, int32_t x, HoughMatrixPtr hough_space,
      std::vector<circle>* new_maxima, std::vector<int>* new_maxima_value,
      bool skip_center = false);
  void applySuppressionRadius(
      const std::vector<circle>& candidate_maxima,
      const std::vector<int>& candidate_maxima_values,
      std::vector<circle>* maxima);
  template <typename DerivedVec, typename DerivedMat>
  void initializeSinCosMap(
      Eigen::EigenBase<DerivedVec>& angles,
      Eigen::EigenBase<DerivedMat>& sin_cos_map, const int kMinAngle,
      const int kMaxAngle, const int kNumSteps);
  bool isLocalMaxima(HoughMatrixPtr hough_space, int32_t r, int32_t x, int32_t y);
  void newPoleDetection(double rho, double theta, double window_time, bool pol);

  void computeFullHoughSpace(
      size_t index, HoughMatrixPtr hough_space, const std::vector<point>& points);

  void computeFullNMS(
      const HoughMatrixPtr hough_space, std::vector<circle> *maxima);
  void iterativeNMS(
      const std::vector<point>& points, HoughMatrixPtr hough_space, 
      std::vector<std::vector<circle>>* maxima_list,
      std::vector<size_t>* maxima_change);

  void eventPreProcessing(
      const dvs_msgs::EventArray::ConstPtr& orig_msg,
      std::vector<point>* points, std::vector<double>* times);

  // Initialisation functions.
  void computeUndistortionMapping();
  void loadCalibration();

  // Visualization functions.
  void drawPolarCorLine(
      cv::Mat& image_space, float rho, float  theta, cv::Scalar color) const;
  void visualizeCurrentDetections(
      const std::vector<point>& points,
      const std::vector<std::vector<circle>>& maxima_list,
      const std::vector<size_t>& maxima_change) const;

  // For the very first message we need separate processing (e.g. a full HT).
  bool initialized;
  HoughMatrixPtr hough_space;

  std::vector<circle> last_maxima;
  std::vector<point> last_points;
  std::vector<double> last_times;

  size_t num_messages;
  std::queue<size_t> (*filter_grid_)[kHoughSpaceWidth];

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
