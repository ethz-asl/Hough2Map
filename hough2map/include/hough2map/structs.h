#ifndef HOUGH2MAP_STRUCTS_H_
#define HOUGH2MAP_STRUCTS_H_

#include <stdint.h>

namespace hough2map {

struct ProfilingInfo {
  double total_events_timing_us;
  double total_msgs_timing_ms;
  uint64_t total_events;
  uint64_t total_msgs;
};

// Struct for describing a detected line.
struct HoughLine {
  int ID;
  int r;
  float theta;
  int theta_idx;
  double time;
};

// Struct for describing a tracked pole.
struct TrackerPole {
  int ID;
  double pos_x;
  double pos_y;
  double first_observed;
  float weight;
};

}  // namespace hough2map
#endif