#include "hough2map/tracker.h"

// FIXME: Remove
#include <ros/ros.h>

namespace hough2map {

// ----------------------------------------------------------------------------
// Tracker Definitions
// ----------------------------------------------------------------------------

Tracker::Tracker(TrackerConfig config, std::vector<PointTX> init_pts)
    : config_(config), tracked_points_(init_pts) {
  // Compute c0 and c1 from from init_pts
  updateLastNLineCoeffs();
}

Tracker::~Tracker() {}

int Tracker::length() { return tracked_points_.size(); }

double Tracker::lastActiveTime() { return tracked_points_.back().t; }

std::vector<PointTX> Tracker::getPoints() { return tracked_points_; }

void Tracker::updateLastNLineCoeffs() {
  // Update linearity coeffs from given points

  int n_pts = config_.linearity_window;

  Eigen::MatrixXd A;
  A.resize(n_pts, 2);
  A.setOnes();
  Eigen::VectorXd b;
  b.resize(n_pts);

  // Set A and b
  int offset_idx = tracked_points_.size() - n_pts;
  double t0 = tracked_points_[offset_idx].t;
  for (int i = 0; i < n_pts; i++) {
    A(i, 1) = tracked_points_[offset_idx + i].t - t0;
    b(i) = tracked_points_[offset_idx + i].x;
  }

  // Compute c0 and c1 (LDLT: https://eigen.tuxfamily.org/dox/group__LeastSquares.html)
  // Eigen::Vector2d coeffs = (A.transpose() * A).ldlt().solve(A.transpose() * b);
  // Eigen::Vector2d coeffs = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
  Eigen::Vector2d coeffs = A.colPivHouseholderQr().solve(b);

  line_coeffs_.first = coeffs(0) - coeffs(1) * t0;
  line_coeffs_.second = coeffs(1);
}

bool Tracker::checkAndAdd(PointTX p) {
  // Check if new point p falls in the linearity of previous 7 points
  int x_hat = (int)(line_coeffs_.first + line_coeffs_.second * p.t);
  if (std::abs(p.x - x_hat) <= config_.linearity_tol_px) {
    tracked_points_.push_back(p);
    updateLastNLineCoeffs();
    return true;
  }
  return false;
}

// ----------------------------------------------------------------------------
// Tracker Manager Definitions
// ----------------------------------------------------------------------------

TrackerManager::TrackerManager() { config_ = TrackerManagerConfig(); }

TrackerManager::~TrackerManager() {}

void TrackerManager::init(TrackerManagerConfig config) { config_ = config; }

void TrackerManager::track(double t, std::vector<int> centroid_list) {
  // For each new centroid, check if a new tracker can be spawned
  for (auto &&max : centroid_list) {
    PointTX p = {t, max};

    // For each tracker, check if the current point can be added
    bool added = false;
    for (auto &&tracker : trackers_) {
      added = added || tracker.checkAndAdd(p);
    }

    if (added) continue;

    // Check if a line can be found from this current point
    std::vector<std::pair<double, std::vector<int>>> slope_vecs;

    // Populate Slope vectors
    for (int i = centroid_buffer_.size() - 1; i >= 0; i--) {
      double dt = p.t - centroid_buffer_[i].t;
      double dxdt = (p.x - centroid_buffer_[i].x) / dt;

      // Skip this point if slope is less
      if (std::abs(dxdt) < config_.min_dx_dt || std::abs(dxdt) > config_.max_dx_dt) continue;

      // Else try to see if dx/dt fits in any cluster
      bool added = false;
      for (auto &&sv : slope_vecs) {
        if ((int)(std::abs(dxdt - sv.first) * dt) > config_.dx_cluster_tol) continue;

        // Update avg theta and append to buffer index vector
        int n = sv.second.size();
        sv.first = (n * sv.first + dxdt) / (n + 1);
        sv.second.push_back(i);

        // Misc flags
        added = true;
        break;
      }

      if (added) continue;

      std::pair<double, std::vector<int>> new_vec;
      std::vector<int> p_list{i};
      new_vec.first = dxdt;
      new_vec.second = p_list;
      slope_vecs.push_back(new_vec);
    }

    // For each slope vecs, check if size > kTrackerSpawnThreshold
    bool spawned = false;
    for (auto &&sv : slope_vecs) {
      if (sv.second.size() >= config_.tracker_spawn_threshold) {
        // Create vector of points and remove from buffer
        std::vector<PointTX> pts;
        for (auto &&idx : sv.second) {
          pts.push_back(centroid_buffer_[idx]);
          centroid_buffer_.erase(std::next(centroid_buffer_.begin(), idx));
        }

        Tracker t(config_.tracker_config, pts);
        trackers_.push_back(t);
        // Flags
        spawned = true;
      }
    }

    // If yes, spawn a new tracker and remove these points from buffer
    if (spawned) continue;

    // If not, add these points to buffer and resize buffer
    centroid_buffer_.push_back(p);
    while (centroid_buffer_.size() > config_.centroid_buffer_size) {
      centroid_buffer_.pop_front();
    }
  }
}

std::vector<Tracker> TrackerManager::getFinishedTrackers(double t) {
  // See if any trackers are mature enough and haven't been updated
  std::vector<Tracker> return_trackers;
  for (size_t i = 0; i < trackers_.size(); i++) {
    if (t - trackers_[i].lastActiveTime() > config_.maturity_age) {
      return_trackers.push_back(trackers_[i]);
      trackers_.erase(std::next(trackers_.begin(), i));
    }
  }

  return return_trackers;
}

}  // namespace hough2map