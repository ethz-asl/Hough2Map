#ifndef HOUGH2MAP_CONFIG_H_
#define HOUGH2MAP_CONFIG_H_

#include <Eigen/Dense>
#include <vector>

namespace hough2map {

// Config Struct with meaningful defaults
struct Hough1Config {
  int radial_resolution = 660;  // Radial resolution (in px) for Hough Tr 1
  int angular_resolution = 21;  // Angular resolution for Hough Tr 1
  int min_angle = -10;          // Min angle for Hough Tr 1
  int max_angle = 10;           // Max angle for Hough Tr 1
  int threshold = 15;           // Num votes threshold
  int window_size = 300;        // Window size for events
  int nms_radius = 10;          // NMS suppression radius (in px)
};

struct DetectorConfig {
  int evt_subsample_fac = 1;         // Subsampling factor for events
  double buffer_size_s = 30.0;       // Time in s for not discarding odom buffer
  int tsteps_per_msg = 3;            // Timesteps per msg
  int msg_per_window = 100;          // Messages per window
  int centroid_find_window = 7;      // Window (in px) to search centroid within along a maxima list
  int centroid_find_thresh = 5;      // Min num of maximas needed to find a centroid
  double triangln_sv_thresh = 0.05;  // Threshold of SV below which solution is accepted
};

struct CameraConfig {
  int evt_arr_frequency = 30;     // Frequency of event array message
  int cam_res_width = 240;        // Camera resolution width
  int cam_res_height = 180;       // Camera resolution height
  double acceptable_dist = 40.0;  // Acceptable distortion range
  double intrinsics[4];           // Intrinsics
  double dist_coeffs[4];          // Distortion Coeffs
  bool perform_undist = true;     // Perform undistortion Coeffs
  Eigen::Affine3d T_cam_to_body;  // Transformation from cam to body

  // Derived Quantities
  Eigen::MatrixXf undist_map_x;  // Dist mapping x
  Eigen::MatrixXf undist_map_y;  // Dist mapping y
};

struct TrackerConfig {
  int linearity_window = 7;  // Check linearity against n last points of tracker
  int linearity_tol_px = 3;  // Tolerance in px below which new point is added
};

struct TrackerManagerConfig {
  int tracker_spawn_threshold = 20;  // Min number of observations required to spawn tracker
  int centroid_buffer_size = 150;    // Size of centroids in the buffer
  int dx_cluster_tol = 3;            // Tolerance in px for passing
  double min_dx_dt = 50;             // Minimum dx/dt needed for fitting line
  double max_dx_dt = 5000;           // Maximum dx/dt needed for fitting line
  double max_dx_allowed = 100;       // Maximum dx allowed to be added into slope vec
  double maturity_age = 1.0;         // Time (in s) after which trackers are discarded
  TrackerConfig tracker_config;
};

struct OutputConfig {
  std::string map_file;  // File to store map points
  bool rviz;             // Flag to enable/disable rviz output
};

}  // namespace hough2map
#endif