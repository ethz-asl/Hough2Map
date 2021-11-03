#include "hough2map/tracker.h"

namespace hough2map {

HeuristicTracker::HeuristicTracker(HeuristicTrackerConfig config) : config_(config) {}

HeuristicTracker::~HeuristicTracker() {}

int HeuristicTracker::length() { return tracked_points_.size(); }

double HeuristicTracker::lastActiveTime() { return tracked_points_.back().first; }

std::vector<PairTX> HeuristicTracker::getPoints() { return tracked_points_; }

bool HeuristicTracker::checkAndAdd(double t, int x) {
  // Check if too much time has passed since last point
  // OR Check if it's not too close to last entry
  double dt;
  if (tracked_points_.size() > 0) {
    dt = std::abs(t - tracked_points_.back().first);
  } else {
    // Assign some feasible value to allow it to pass
    dt = config_.kMaxAllowedDt / 2;
  }
  if (dt > 1e-3 && dt < config_.kMaxAllowedDt) {
    if (tracked_points_.size() == 0) {
      // If it's the first point, just add
      PairTX first_point(t, x);
      tracked_points_.push_back(first_point);
      return true;
    } else if (tracked_points_.size() == 1) {
      // If it's the second point, check if it's within seed delta
      double dpx_dt = (x - tracked_points_.back().second) / (t - tracked_points_.back().first);
      if (std::abs(dpx_dt) >= config_.kP1ToleranceDpxDt) {
        PairTX second_point(t, x);
        tracked_points_.push_back(second_point);
        return true;
      }
    } else {
      // If it's the any other point, check if it's within D(dx/dt) tolerance
      const auto p_1 = tracked_points_.end()[-1];
      const auto p_2 = tracked_points_.end()[-2];
      const double m_1 = (p_2.second - p_1.second) / (p_2.first - p_1.first);
      const double m = (x - p_1.second) / (t - p_1.first);
      if (std::abs((m - m_1) / m_1) <= config_.kPnToleranceDeltaDxDt) {
        PairTX new_point(t, x);
        tracked_points_.push_back(new_point);
        return true;
      }
    }
  }
  return false;
}

}  // namespace hough2map