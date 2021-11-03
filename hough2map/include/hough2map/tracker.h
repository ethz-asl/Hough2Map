#ifndef HOUGH2MAP_TRACKER_H_
#define HOUGH2MAP_TRACKER_H_

#include <stdint.h>
#include <vector>
#include <cmath>

namespace hough2map {

typedef std::pair<double, int> PairTX;

struct HeuristicTrackerConfig {
  double kP1ToleranceDpxDt = 120;  // Tolerance in abs(dpx/dt) above which second point is added
  double kPnToleranceDeltaDxDt = 0.05;  // Tolerance in % below which new point is added
  double kMaxAllowedDt = 1.0;           // Max dt (in s) allowed from last tracked point
};

class HeuristicTracker {
 public:
  HeuristicTracker(HeuristicTrackerConfig config);
  ~HeuristicTracker();

  int length();
  double lastActiveTime();
  std::vector<PairTX> getPoints();

  void terminate();
  bool checkAndAdd(double t, int x);

 private:
  HeuristicTrackerConfig config_;       // Tracker Config Options
  std::vector<PairTX> tracked_points_;  // Tracked Points
};

}  // namespace hough2map
#endif