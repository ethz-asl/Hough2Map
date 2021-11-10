#ifndef HOUGH2MAP_TRACKER_H_
#define HOUGH2MAP_TRACKER_H_

#include <stdint.h>
#include <Eigen/Dense>
#include <vector>
#include <deque>
#include <cmath>

#include "hough2map/config.h"

namespace hough2map {

struct PointTX {
  double t;
  int x;
};

// Tracker Class
class Tracker {
 public:
  // Constructor and Destructor
  Tracker(TrackerConfig config, std::vector<PointTX> init_pts);
  ~Tracker();

  // Util functions
  int length();
  double lastActiveTime();
  std::vector<PointTX> getPoints();

  // Essential functions
  bool checkAndAdd(PointTX p);

 private:
  TrackerConfig config_;                   // Tracker Config Options
  std::vector<PointTX> tracked_points_;    // Tracked Points
  std::pair<double, double> line_coeffs_;  // Line coefficients

  // Util functions
  void updateLastNLineCoeffs();
};

// Tracker Manager Class
class TrackerManager {
 public:
  // Constructor and Destructor
  TrackerManager();
  ~TrackerManager();

  // Util functions
  void init(TrackerManagerConfig config);

  // Essential functions
  void track(double t, std::vector<int> centroid_list);
  std::vector<Tracker> getFinishedTrackers(double t);

 private:
  TrackerManagerConfig config_;          // Tracker Config Options
  std::deque<PointTX> centroid_buffer_;  // Buffer to store centroids
  std::vector<Tracker> trackers_;
};

}  // namespace hough2map
#endif