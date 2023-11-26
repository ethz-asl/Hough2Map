#include "hough2map/detector.h"

#include <chrono>
//#include <omp.h>
#include <ros/package.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
//#include <thread>

DEFINE_string(rosbag_path, "", "Rosbag to process.");
DEFINE_string(event_topic, "/dvs/events", "Topic for event messages.");
DEFINE_string(image_topic, "", "Topic for image messages.");
DEFINE_string(
    dvs_calibration, "calibration.yaml", "Camera parameters for the DVS.");

DEFINE_int32(
    hough_threshold, 15, "Threshold for the first level Hough transform.");
DEFINE_int32(
    hough_window_size, 300, "Max queue length for the first Hough transform.");
DEFINE_int32(
    hough_nms_radius, 10,
    "Non-Maximum-Suppression suppression radius to enforce Maxima separation");

DEFINE_int32(
    event_subsample_factor, 1, "Subsample Events by a constant factor");
DEFINE_bool(
    show_lines_in_video, false, "Plot detected lines in the video stream");
DEFINE_int32(
    show_lines_every_nth, 10, "Event frequency at which to plot the lines.");
DEFINE_bool(map_output, false, "Export detected poles to file");

DEFINE_int32(tracking_angle_step, 3, "");
DEFINE_int32(tracking_pixel_step, 5, "");
DEFINE_double(tracking_max_jump, 0.4, "");
DEFINE_int32(tracking_mean_window, 20, "");
DEFINE_int32(tracking_min_length, 500, "");

namespace hough2map {

Detector::Detector(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private)
    : nh_(nh), nh_private_(nh_private) {
  // Checking that flags have reasonable values.
  CHECK_GE(FLAGS_hough_window_size, 1);
  CHECK_GT(FLAGS_event_subsample_factor, 0);
  CHECK_GT(FLAGS_hough_threshold, 0);
  CHECK(!FLAGS_rosbag_path.empty());

  // Debug printing
  LOG(INFO) << "Resolution: | height: " << kHoughSpaceHeight - 2
      << " | width: " << kHoughSpaceWidth - 2
      << " | radius: " << kHoughSpaceRadius - 2;
  LOG(INFO) << "Min radius: " << kHoughMinRadius
      << " | max radius: " << kHoughMaxRadius;

  // Initialize multi-threading
  //omp_set_num_threads(kNumThreads);

  // Allocate memory for the hough space
  hough_space = new HoughMatrix;
  gradient_window = new HoughImage;
  filter_grid_ = new std::queue<size_t>[kHoughSpaceHeight][kHoughSpaceWidth];

  // Keep track of number of dvs messages received for filtering
  num_messages = 0;

  // Tracking parameters
  next_track_id = 0;
  tracking_angle_step = FLAGS_tracking_angle_step;
  tracking_pixel_step = FLAGS_tracking_pixel_step;
  tracking_max_jump = FLAGS_tracking_max_jump;
  tracking_mean_window = static_cast<size_t>(FLAGS_tracking_mean_window);

  // Set initial status to false until first message is processed.
  initialized = false;

  // Output file for the map data.
  if (FLAGS_map_output) {
    map_file.open(map_file_path);
    CHECK(map_file.is_open());
  }

  debug_file.open(debug_file_path);
  CHECK(debug_file.is_open());

  // Timing statistics for performance evaluation.
  total_events_timing_us = 0.0;
  total_tracking_timing_ms = 0.0;
  total_msgs_timing_ms = 0.0;
  total_events = 0;
  total_msgs = 0;

  // Import calibration file.
  loadCalibration();

  // Compute undistortion for given camera parameters.
  computeUndistortionMapping();

  // Precompute lines for easier runtime performance
  lines_xy.resize(21 * 21);
  size_t line_count = 0;
  for (int32_t gx = -10; gx <= 10; ++gx) {
    for (int32_t gy = -10; gy <= 10; ++gy) {
      if (!gx) {
        for (int32_t r = kHoughMinRadius; r <= kHoughMaxRadius; ++r) {
          lines_xy[line_count].emplace_back(0, r);
          lines_xy[line_count].emplace_back(0, -r);
        }
      } else {
        double a = static_cast<double>(gy) / static_cast<double>(gx);
        for (int32_t r = kHoughMinRadius; r <= kHoughMaxRadius; ++r) {
          double aux = static_cast<double>(r) / std::sqrt(a * a + 1);
          int32_t x = std::round(aux);
          int32_t y = std::round(a * aux);

          // The +1's are a hacky way to speed up adding the padding
          // for the matrix so we have to do less edge checks.
          lines_xy[line_count].emplace_back(x + 1, y + 1);
          lines_xy[line_count].emplace_back(-x + 1, -y + 1);
        }
      }
      ++line_count;
    }
  }

  hough_nms_radius3_ = FLAGS_hough_nms_radius * 
      FLAGS_hough_nms_radius * FLAGS_hough_nms_radius;

  // Open the input bag.
  rosbag::Bag bag;
  try {
    LOG(INFO) << "Opening bag: " << FLAGS_rosbag_path << ".";
    bag.open(FLAGS_rosbag_path, rosbag::bagmode::Read);
  } catch (const std::exception& ex) {  // NOLINT
    LOG(FATAL) << "Could not open the rosbag " << FLAGS_rosbag_path << ": "
               << ex.what();
  }

  // Select relevant topics.
  std::vector<std::string> topics;
  topics.emplace_back(FLAGS_event_topic);

  if (FLAGS_show_lines_in_video) {
    LOG(WARNING)
        << "Visualization can be very slow, especially when using low values "
        << "for --show_lines_every_nth.";
    cv::namedWindow("Detected circles", cv::WINDOW_NORMAL);
    cv::namedWindow("Hough space", cv::WINDOW_NORMAL);
  }

  if (!FLAGS_image_topic.empty()) {
    cv::namedWindow("Camera image", cv::WINDOW_NORMAL);
    topics.emplace_back(FLAGS_image_topic);
  }

  // Iterate over the bag and invoke the appropriate callbacks.
  rosbag::View bag_view(bag, rosbag::TopicQuery(topics));
  rosbag::View::iterator it_message = bag_view.begin();
  while (it_message != bag_view.end()) {
    const rosbag::MessageInstance& message = *it_message;
    const std::string& topic = message.getTopic();
    CHECK(!topic.empty());

    if (topic == FLAGS_event_topic) {
      dvs_msgs::EventArray::ConstPtr event_message =
          message.instantiate<dvs_msgs::EventArray>();
      CHECK(event_message);
      eventCallback(event_message);
    } else if (topic == FLAGS_image_topic) {
      sensor_msgs::Image::ConstPtr image_message =
          message.instantiate<sensor_msgs::Image>();    
      CHECK(image_message);
      imageCallback(image_message);
    }

    ++it_message;
  }

  bag.close();
}

Detector::~Detector() {
  delete hough_space;
  delete gradient_window;
  delete filter_grid_;

  if (FLAGS_map_output) {
    map_file.close();
  }
}

void Detector::loadCalibration() {
  // File path to calibration file.
  std::string package_path = ros::package::getPath("hough2map");
  std::string calibration_file =
      package_path + "/share/" + FLAGS_dvs_calibration;

  cv::FileStorage fs(calibration_file, cv::FileStorage::READ);

  if (!fs.isOpened()) {
    LOG(FATAL) << "Could not open calibration file:" << calibration_file
               << std::endl;
  }

  // Import parameters from calibration file.
  cv::FileNode cam = fs["cam0"];

  // First, let's import the sensor resolution.
  cv::FileNode resolution = cam["resolution"];
  CHECK_EQ(resolution.size(), 2)
      << ": Not enough calibration data regarding sensor size!";

  // Importing sensor resolution.
  camera_resolution_width_ = resolution[0];
  camera_resolution_height_ = resolution[1];

  // Importing camera intrinsics. Expecting 4 values.
  CHECK_EQ(cam["intrinsics"].size(), 4)
      << ": Not enough calibration data regarding sensor intrinsics!";
  cv::FileNodeIterator it = cam["intrinsics"].begin(),
                       it_end = cam["intrinsics"].end();
  int i = 0;
  for (; it != it_end; ++it) {
    intrinsics_[i] = (*it).real();
    i++;
  }

  // Importing the distortion coefficients, again expecting 4 values.
  CHECK_EQ(cam["distortion_coeffs"].size(), 4)
      << ": Not enough calibration data regarding distortion coefficients!";
  it = cam["distortion_coeffs"].begin(),
  it_end = cam["distortion_coeffs"].end();
  i = 0;
  for (; it != it_end; ++it) {
    distortion_coeffs_[i] = (*it).real();
    i++;
  }
}

void Detector::computeUndistortionMapping() {
  // Setup camera intrinsics from calibration file.
  cv::Mat camera_matrix =
      (cv::Mat1d(3, 3) << intrinsics_[0], 0, intrinsics_[2], 0, intrinsics_[1],
       intrinsics_[3], 0, 0, 1);
  cv::Mat distortion_coeffs =
      (cv::Mat1d(1, 4) << distortion_coeffs_[0], distortion_coeffs_[1],
       distortion_coeffs_[2], distortion_coeffs_[3]);

  cv::Size image_size =
      cv::Size(camera_resolution_width_, camera_resolution_height_);
  cv::Mat optimal_matrix = cv::getOptimalNewCameraMatrix(
      camera_matrix, distortion_coeffs, image_size, 1.0);

  cv::initUndistortRectifyMap(
      camera_matrix, distortion_coeffs, cv::Mat::eye(3, 3, CV_32F),
      optimal_matrix, image_size, CV_32F, image_undist_map_x_, image_undist_map_y_);

  // Compute also the inverse mapping from distorted event points to the
  // undistorted image plane.
  std::vector<cv::Point2f> points;
  for (int i = 0; i < camera_resolution_width_; ++i) {
    for (int j = 0; j < camera_resolution_height_; ++j) {
      points.emplace_back(cv::Point2f(i, j));
    }
  }

  std::vector<cv::Point2f> points_undist;
  cv::undistortPoints(
        points, points_undist, camera_matrix, distortion_coeffs, cv::noArray(),
        optimal_matrix);

  event_undist_map_x_.resize(camera_resolution_height_, camera_resolution_width_);
  event_undist_map_y_.resize(camera_resolution_height_, camera_resolution_width_);

  size_t index = 0;
  for (int i = 0; i < camera_resolution_width_; ++i) {
    for (int j = 0; j < camera_resolution_height_; ++j) {
      event_undist_map_x_(j, i) = points_undist[index].x;
      event_undist_map_y_(j, i) = points_undist[index].y;
      ++index;
    }
  }
}

inline void Detector::calculateSobel(
    int32_t x, int32_t y, int32_t* magnitude, int32_t* direction) {
  int32_t G_x = 
    -2 * gradient_window[y][x - 1] - gradient_window[y - 1][x - 1]
    - gradient_window[y + 1][x - 1] + 2 * gradient_window[y][x + 1]
    + gradient_window[y - 1][x + 1] + gradient_window[y + 1][x + 1];

  int32_t G_y = 
    -2 * gradient_window[y - 1][x] - gradient_window[y - 1][x - 1]
    - gradient_window[y - 1][x + 1] + 2 * gradient_window[y + 1][x]
    + gradient_window[y + 1][x - 1] + gradient_window[y + 1][x + 1];

  if (std::abs(G_x) > 10 || std::abs(G_y) > 10) {
    LOG(INFO) << G_x << " " << G_y << " " << x << " " << y;
  }

  G_x = std::min(std::max(G_x, -10), 10);
  G_y = std::min(std::max(G_y, -10), 10);

  *magnitude = G_x * G_x + G_y * G_y;
  *direction = (G_y + 10) * 21 + (G_x + 10);
}

void Detector::stepHoughTransform(
      const std::vector<point>& points,
      std::vector<int32_t>* gradient_magnitudes,
      std::vector<int32_t>* gradient_directions,
      HoughMatrixPtr hough_space, std::vector<circle> *last_maxima,
      bool initialized, std::vector<std::vector<circle>> *maxima_list,
      std::vector<size_t> *maxima_change) {
  CHECK_NOTNULL(last_maxima);
  CHECK_NOTNULL(maxima_list);
  CHECK_NOTNULL(maxima_change);

  if (!initialized) {
    // Initialize hough space and gradient computation window to zero
    std::memset(hough_space, 0, sizeof(HoughMatrix));
    std::memset(gradient_window, 0, sizeof(HoughImage));

    // Cumulate gradient window
    for (size_t i = 0; i < FLAGS_hough_window_size; ++i) {
      ++gradient_window[points[i].y + 1][points[i].x + 1];
    }

    for (size_t i = 0; i < FLAGS_hough_window_size; ++i) {
      calculateSobel(
          points[i].x + 1, points[i].y + 1,
          &((*gradient_magnitudes)[i]),
          &((*gradient_directions)[i]));
    }

    computeFullHoughSpace(
        FLAGS_hough_window_size - 1, hough_space, points,
        *gradient_magnitudes, *gradient_directions);

    // Initialize first set of maxima.
    maxima_list->emplace_back();
    computeFullNMS(hough_space, maxima_list->data());
  } else {
    maxima_list->emplace_back(*last_maxima);
  }

  for (size_t i = FLAGS_hough_window_size; i < points.size(); ++i) {
    ++gradient_window[points[i].y + 1][points[i].x + 1];
    const size_t past_i = i - FLAGS_hough_window_size;
    --gradient_window[points[past_i].y + 1][points[past_i].x + 1];
    calculateSobel(
        points[i].x + 1, points[i].y + 1,
        &((*gradient_magnitudes)[i]),
        &((*gradient_directions)[i]));
  }

  /*for (size_t i = FLAGS_hough_window_size; i < points.size(); i += 100) {
    cv::Mat vis_frame(
        cv::Size(camera_resolution_width_, camera_resolution_height_), CV_8UC3,
            cv::Scalar(255, 255, 255));

    for (size_t j = i - FLAGS_hough_window_size + 1; j <= i; j++) {
      //if (gradient_magnitude[j] == 0) {
      //  continue;
      //}

      vis_frame.at<cv::Vec3b>(points[j].y, points[j].x) =
            cv::Vec3b(0, 0, 255);
    }

    for (size_t j = i - FLAGS_hough_window_size + 1; j <= i; j++) {
      if (gradient_magnitude[j] == 0) {
        continue;
      }

      cv::Mat vis_lines = vis_frame.clone();
      for (const point& p : lines_xy[gradient_direction[j]]) {
        vis_lines.at<cv::Vec3b>(points[j].y + p.y, points[j].x + p.x) =
            cv::Vec3b(0, 255, 0);
      }

      cv::imshow("Gradients", vis_lines);
      cv::waitKey(500);
    }
  }*/

  // Perform computations iteratively for the rest of the events.
  iterativeNMS(
      points, *gradient_magnitudes, *gradient_directions, hough_space,
      maxima_list, maxima_change);

  // Store last set of maxima to have a starting point for the next message.
  *last_maxima = maxima_list->back();
}

void Detector::eventCallback(const dvs_msgs::EventArray::ConstPtr& msg) {
  // Start timing.
  const auto kStartTime = std::chrono::high_resolution_clock::now();
  if (!initialized) {
    start_time = msg->header.stamp.toSec();
  }

  // Count number of messages for filtering purposes
  ++num_messages;

  // Reshaping the event array into an Eigen matrix.
  std::vector<point> points;
  std::vector<double> times;
  eventPreProcessing(msg, &points, &times);

  // The new points will now be the next old points.
  const int keep = std::min(
      static_cast<size_t>(FLAGS_hough_window_size), points.size());
  last_points.resize(keep);
  last_times.resize(keep);
  std::copy(points.end() - keep, points.end(), last_points.begin());
  std::copy(times.end() - keep, times.end(), last_times.begin());

  // Wait until we have enough points to initialize.
  if (points.size() < FLAGS_hough_window_size || num_messages < 100) {
    return;
  }

  // Each event is treated as a timestep. For each of these timesteps we keep
  // the active set of maxima in the Hough Space. These are basically the line
  // detections at each timestep. This whole storage is pre-initialized to
  // make it ready for parallelizing the whole process.
  std::vector<int32_t> gradient_magnitudes(points.size());
  std::vector<int32_t> gradient_directions(points.size());
  std::vector<std::vector<circle>> maxima_list;
  std::vector<size_t> maxima_change;

  // Copy over the gradient information here if we have any.
  if (initialized) {
    std::move(
        last_magnitudes.begin(), last_magnitudes.end(),
        gradient_magnitudes.begin());
    std::move(
        last_directions.begin(), last_directions.end(),
        gradient_directions.begin());
  }

  stepHoughTransform(
     points, &gradient_magnitudes, &gradient_directions, hough_space,
     &last_maxima, initialized, &maxima_list, &maxima_change);
  initialized = true;

  last_magnitudes.resize(keep);
  last_directions.resize(keep);
  std::copy(
      gradient_magnitudes.end() - keep, gradient_magnitudes.end(),
      last_magnitudes.begin());
  std::copy(
      gradient_directions.end() - keep, gradient_directions.end(),
      last_directions.begin());

  // Dump detected maxima to file
  /*if (FLAGS_map_output) {
    for (size_t i = 1; i < maxima_list.size(); ++i) {
      const size_t event_index = maxima_change[i - 1];
      map_file 
          << std::fixed << std::setprecision(12) << times(event_index)
          << "," << maxima_list[i].size() << std::endl;
      for (const circle& maxima : maxima_list[i]) {
        map_file << maxima.r << "," << maxima.theta_idx << std::endl;
      }
    }
  }*/

  const auto kHoughTime = std::chrono::high_resolution_clock::now();

  // If visualizations are turned on display them in the video stream.
  // NOTE: This slows everything down by a very large margin
  if (FLAGS_show_lines_in_video) {
    visualizeCurrentDetections(
        points, maxima_list, maxima_change);
  }

  // Greedy heuristic for line tracking
  /*for (size_t i = 1; i < maxima_list.size(); i++) {
    double time = times(maxima_change[i - 1]);

    // Get sliding mean of tracking head
    std::map<size_t, track_point> track_means;
    for (auto it = tracks.begin(); it != tracks.end(); it++) {
      const size_t track_len = it->second.size();
      const size_t back = std::min(tracking_mean_window, track_len);
      double mean_pixel = 0;
      double mean_angle = 0;
      for (size_t j = 0; j < back; ++j) {
        mean_pixel += it->second[track_len - 1 - j].pixel;
        mean_angle += it->second[track_len - 1 - j].angle;
      }
      mean_pixel /= back;
      mean_angle /= back;

      track_point mean_point(mean_pixel, mean_angle, 0.0);
      track_means.emplace(it->first, mean_point);
    }

    // Only update at the end to not mess up iteration and association.
    std::map<size_t, track_point> track_updates;
    std::vector<track_point> new_tracks;
    for (const circle& maxima : maxima_list[i]) {
      // Potential nearest neighbors.
      std::vector<size_t> potential_matches;
      for (auto it = track_means.begin(); it != track_means.end(); it++) {
        if ((std::abs(maxima.r - it->second.pixel) <= tracking_pixel_step) &&
            (std::abs(maxima.theta_idx - it->second.angle) <= tracking_angle_step)) {
          potential_matches.emplace_back(it->first);
        }
      }

      track_point new_point(maxima.r, maxima.theta_idx, time);
      if (potential_matches.size() == 0) {
        new_tracks.emplace_back(new_point);
      } else {
        // If multiple tracks might be associated pick the longest one.
        size_t max_len = 0;
        size_t max_match = 0;
        for (const size_t match : potential_matches) {
          if (tracks[match].size() > max_len) {
            max_len = tracks[match].size();
            max_match = match;
          }
        }
        track_updates.emplace(max_match, new_point);
      }
    }

    // Add new points to existing tracks if they matched.
    for (auto it = track_updates.begin(); it != track_updates.end(); it++) {
      tracks[it->first].emplace_back(it->second);
    }

    // Discard old tracks that we have no hope of continuing.
    for (auto it = tracks.begin(); it != tracks.end(); ) {
      if (it->second.back().time + tracking_max_jump < time) {
        if (it->second.size() > FLAGS_tracking_min_length) {
          if (FLAGS_map_output) {
            map_file << it->second.size() << std::endl;
            for (size_t j = 0; j < it->second.size(); ++j) {
              map_file << std::fixed << std::setprecision(12)
                << it->second[j].time <<  "," 
                << static_cast<size_t>(it->second[j].pixel) << ","
                << static_cast<size_t>(it->second[j].angle) << std::endl;
            }
          }
        }
        it = tracks.erase(it);
      } else {
        ++it;
      }
    }

    // Create new tracks from points that didn't match existing tracks.
    for (track_point& new_point : new_tracks) {
      tracks[next_track_id].emplace_back(new_point);
      ++next_track_id;
    }
  }*/

  // Calculate statistics
  const auto kTrackingTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> duration_hough_us =
      kHoughTime - kStartTime;
  std::chrono::duration<double, std::micro> duration_tracking_us =
      kTrackingTime - kHoughTime;

  total_events_timing_us += duration_hough_us.count();
  total_tracking_timing_ms += duration_tracking_us.count() / 1000.0;
  total_msgs_timing_ms += duration_hough_us.count() / 1000.0;

  total_events += points.size();
  total_msgs++;

  double elapsed_time = msg->header.stamp.toSec() - start_time;

  VLOG(1) << detector_name_ << std::fixed << std::setprecision(2)
          << std::setfill(' ') << " speed: " << std::setw(6)
          << total_events_timing_us / total_events << " us/event | "
          << std::setw(6) << total_msgs_timing_ms / total_msgs
          << " ms/msg | " << std::setw(6) << points.size() << " e/msg | "
          << "tracking: " << total_tracking_timing_ms / total_msgs
          << " ms/msg | " << "progress " << elapsed_time << " s.";
}

// Visualizing the current line detections and corresponding Hough space.
void Detector::visualizeCurrentDetections(
    const std::vector<point>& points, 
    const std::vector<std::vector<circle>>& maxima_list,
    const std::vector<size_t>& maxima_change) const {
  // Predefined colors and strings
  const cv::Scalar color_scalar = cv::Scalar(0, 0, 255);
  const cv::Vec3b color_vec3b = cv::Vec3b(0, 0, 255);

  // Getting the horizontal positions of all vertical line detections.
  size_t maxima_index = 0;
  const size_t step_size = FLAGS_show_lines_every_nth;
  for (size_t i = FLAGS_hough_window_size; i < points.size(); i += step_size) {
    cv::Mat vis_frame(
        cv::Size(camera_resolution_width_, camera_resolution_height_), CV_8UC3,
        cv::Scalar(255, 255, 255));

    for (size_t j = i - FLAGS_hough_window_size + 1; j <= i; j++) {
      vis_frame.at<cv::Vec3b>(points[j].y, points[j].x) = color_vec3b;
    }

    if ((maxima_index < maxima_change.size()) &&
        (i >= maxima_change[maxima_index])) {
      ++maxima_index;
    }

    for (auto& maxima : maxima_list[maxima_index]) {
      VLOG(10) << " Maxima: " << maxima.r - 1 + kHoughMinRadius << " at "
          << maxima.x - 1 << ", " << maxima.y - 1;
      cv::circle(
            vis_frame, cv::Point(maxima.x - 1, maxima.y - 1),
            maxima.r - 1 + kHoughMinRadius, cv::Scalar(255, 0, 0));
    }

    cv::imshow("Detected circles", vis_frame);
    cv::waitKey(1);
  }
}

// Performing itterative Non-Maximum suppression on the current batch of
// events based on a beginning Hough Space.
void Detector::iterativeNMS(
    const std::vector<point>& points,
    const std::vector<int32_t>& gradient_magnitudes,
    const std::vector<int32_t>& gradient_directions,
    HoughMatrixPtr hough_space,
    std::vector<std::vector<circle>>* maxima_list,
    std::vector<size_t> *maxima_change) {
  CHECK_NOTNULL(maxima_list);
  CHECK_NOTNULL(maxima_change)->clear();
  CHECK_GT(maxima_list->size(), 0u);

  std::vector<circle> new_maxima;
  std::vector<int> new_maxima_value;

  for (size_t event = FLAGS_hough_window_size; event < points.size(); event++) {
    // Take the maxima at the previous timestep.
    std::vector<circle> &previous_maxima =
        maxima_list->back();

    // Incrementing the accumulator cells for the current event.
    if (gradient_magnitudes[event]) {
      const int32_t line_idx = gradient_directions[event];
      for (size_t j = 0; j < lines_xy[line_idx].size(); ++j) {
        const int32_t r = (j >> 1) + 1;
        const int32_t x = points[event].x + lines_xy[line_idx][j].x;
        const int32_t y = points[event].y + lines_xy[line_idx][j].y;
        if ((x >= 1) && (x < kHoughSpaceWidth - 1) && 
            (y >= 1) && (y < kHoughSpaceHeight - 1)) {
          ++hough_space[r][y][x];
        }
      }
    }

    // Decrement the accumulator cells for the event to be removed.
    const size_t past_event = event - FLAGS_hough_window_size;
    if (gradient_magnitudes[past_event]) {
      const int32_t line_idx = gradient_directions[past_event];
      for (size_t j = 0; j < lines_xy[line_idx].size(); ++j) {
        const int32_t r = (j >> 1) + 1;
        const int32_t x = points[past_event].x + lines_xy[line_idx][j].x;
        const int32_t y = points[past_event].y + lines_xy[line_idx][j].y;
        if ((x >= 1) && (x < kHoughSpaceWidth - 1) && 
            (y >= 1) && (y < kHoughSpaceHeight - 1)) {
          --hough_space[r][y][x];
        }
      }
    }

    /*if (event % 100 == 0) {
      cv::Mat vis_frame(
          cv::Size(camera_resolution_width_, camera_resolution_height_),
          CV_8UC1);
      double max_hough = 0;
      for (int32_t m = 1; m < kHoughSpaceHeight - 1; ++m) {
        for (int32_t n = 1; n < kHoughSpaceWidth - 1; ++n) {
          if (hough_space[1][m][n] > max_hough) {
            max_hough = hough_space[1][m][n];
          }
        }
      }

      for (int32_t m = 1; m < kHoughSpaceHeight - 1; ++m) {
        for (int32_t n = 1; n < kHoughSpaceWidth - 1; ++n) {
          vis_frame.at<uint8_t>(m - 1, n - 1) = 
            static_cast<double>(hough_space[1][m][n]) / max_hough * 255.0;
        }
      }
      LOG(INFO) << " Max: " << max_hough;

      cv::imshow("Hough space", vis_frame);
      cv::waitKey(200);
    }*/

    // The Hough Spaces have been update. Now the iterative NMS has to run and
    // update the list of known maxima. First reset the temporary lists.
    new_maxima.clear();
    new_maxima_value.clear();

    // Remember which past maxima are no longer maxima so that
    // we don't need to check again at the end.
    bool changed = false;
    std::vector<bool> discard(previous_maxima.size(), false);

    // PHASE 1 - Obtain candidates for global maxima

    // For points that got incremented.
    if (gradient_magnitudes[event]) {
      const int32_t line_idx = gradient_directions[event];
      for (size_t j = 0; j < lines_xy[line_idx].size(); ++j) {
        const int32_t r = (j >> 1) + 1;
        const int32_t x = points[event].x + lines_xy[line_idx][j].x;
        const int32_t y = points[event].y + lines_xy[line_idx][j].y;
        if ((x <= 0) || (x >= kHoughSpaceWidth - 1) ||
            (y <= 0) || (y >= kHoughSpaceHeight - 1)) {
          continue;
        }

        // If any of the surrounding ones are equal the center
        // for sure it is not a local maximum.
        bool skip_center = false;

        // Iterate over neighbourhood to check if we might have
        // supressed a surrounding maximum by growing.
        for (int32_t m = r - 1; m <= r + 1; ++m) {
          for (int32_t n = y - 1; n <= y + 1; ++n) {
            for (int32_t p = x - 1; p <= x + 1; ++p) {
              // The center is a separate case.
              if ((m == r) && (n == y) && (p == x)) {
                continue;
              }

              // Compare point to its neighbors.
              if (hough_space[r][y][x] == hough_space[m][n][p]) {
                skip_center = true;
                // Compare to all known maxima from the previous timestep.
                for (size_t i = 0; i < previous_maxima.size(); ++i) {
                  if ((m == previous_maxima[i].r) &&
                      (n == previous_maxima[i].y) &&
                      (p == previous_maxima[i].x)) {
                    // We need to discard an old maximum.
                    changed = true;
                    discard[i] = true;

                    // And add a new one.
                    addMaximaInRadius(
                        m, n, p, hough_space, &new_maxima, &new_maxima_value);
                    break;
                  }
                }
              }
            }
          }
        }

        // The center and a neighbour have the same value so
        // no point in checking if it is a local maximum.
        if (skip_center) {
          continue;
        }

        // This is the case for the center point. First checking if it's currently
        // a maximum.
        if ((hough_space[r][y][x] > FLAGS_hough_threshold) &&
            isLocalMaxima(hough_space, r, y, x)) {
          bool add_maximum = true;
          // Check if it was a maximum previously.
          for (const auto& maximum : previous_maxima) {
            if ((r == maximum.r) && (y == maximum.y) && (x == maximum.x)) {
              add_maximum = false;
              break;
            }
          }

          // If required, add it to the list.
          if (add_maximum) {
            new_maxima.emplace_back(x, y, r);
            new_maxima_value.emplace_back(hough_space[r][y][x]);
          }
        }
      }
    }

    // For accumulator cells that got decremented.
    if (gradient_magnitudes[past_event]) {
      const int32_t line_idx = gradient_directions[past_event];
      for (size_t j = 0; j < lines_xy[line_idx].size(); ++j) {
        const int32_t r = (j >> 1) + 1;
        const int32_t x = points[past_event].x + lines_xy[line_idx][j].x;
        const int32_t y = points[past_event].y + lines_xy[line_idx][j].y;
        if ((x <= 0) || (x >= kHoughSpaceWidth - 1) ||
            (y <= 0) || (y >= kHoughSpaceHeight - 1)) {
          continue;
        }

        // If decremented accumulator cell was previously a maximum, remove it.
        // If it's still a maximum, we will deal with it later.
        bool skip_neighborhood = false;
        for (size_t k = 0; k < previous_maxima.size(); ++k) {
          if ((r == previous_maxima[k].r) &&
              (y == previous_maxima[k].y) &&
              (x == previous_maxima[k].x)) {
            // Mark as discarded since we will already have added it
            // in the next step if it still is above the threshold.
            changed = true;
            discard[k] = true;

            // Re-add to list of possible maxima for later pruning.
            addMaximaInRadius(
                r, y, x, hough_space, &new_maxima, &new_maxima_value);

            // The neighborhood of this accumulator cell has been checked as part of
            // addMaximaInRadius, so no need to do it again.
            skip_neighborhood = true;
            break;
          }
        }

        if (!skip_neighborhood) {
          // Iterate over neighbourhood to check if we might have
          // created a new local maxima by decreasing.
          for (int32_t m = r - 1; m <= r + 1; ++m) {
            for (int32_t n = y - 1; n <= y + 1; ++n) {
              for (int32_t p = x - 1; p <= x + 1; ++p) {
                // The center is a separate case.
                if ((m == r) && (n == y) && (p == x)) {
                  continue;
                }

                // Any neighbor points now larger and a maximum?
                if ((hough_space[r][y][x] + 1 == hough_space[m][n][p]) &&
                    (hough_space[m][n][p] > FLAGS_hough_threshold) &&
                    isLocalMaxima(hough_space, m, n, p)) {
                  // Add to temporary storage.
                  new_maxima.emplace_back(p, n, m);
                  new_maxima_value.emplace_back(hough_space[m][n][p]);
                }
              }
            }
          }
        }
      }
    }

    if (new_maxima.empty()) {
      // If no discards then nothing changed and we can skip this entirely.
      if (changed) {
        // No new maxima in the temporary storage, so we only get rid of the
        // expired ones and keep the rest as it was unchanged. This keeps the
        // sorting stable so no need to redo it.
        std::vector<circle> current_maxima;
        for (int i = 0; i < previous_maxima.size(); i++) {
          if (!discard[i]) {
            current_maxima.emplace_back(previous_maxima[i]);
          }
        }
        maxima_list->emplace_back(current_maxima);
        maxima_change->emplace_back(event);
      }
    } else {
      // We discard all expired old maxima and add the ones that are still
      // valid to the temporary list of new maxima.
      for (int i = 0; i < previous_maxima.size(); i++) {
        if (!discard[i]) {
          new_maxima.emplace_back(previous_maxima[i]);
          new_maxima_value.emplace_back(hough_space[
            previous_maxima[i].r][previous_maxima[i].y][previous_maxima[i].x]);
        }
      }

      // PHASE 2 - Detect and apply suppression
      std::vector<circle> current_maxima;

      // Apply the suppression radius.
      applySuppressionRadius(new_maxima, new_maxima_value, &current_maxima);

      // We need to check if any of the previous maxima that we
      // didn't touch have changed. Then we need to check for new
      // maxima in range and resuppress.
      while (true) {
        const int kNumMaximaCandidates = new_maxima.size();
        for (int i = 0; i < previous_maxima.size(); i++) {
          if (!discard[i]) {
            bool found = false;
            for (const circle& current_maximum : current_maxima) {
              if (current_maximum == previous_maxima[i]) {
                found = true;
                break;
              }
            }

            if (!found) {
              discard[i] = true;
              addMaximaInRadius(
                  previous_maxima[i].r, previous_maxima[i].y, previous_maxima[i].x,
                  hough_space, &new_maxima, &new_maxima_value, true);
            }
          }
        }

        if (kNumMaximaCandidates < new_maxima.size()) {
          applySuppressionRadius(new_maxima, new_maxima_value, &current_maxima);
        } else {
          break;
        }
      }

      // Sort and confirm if something changed.
      std::stable_sort(current_maxima.begin(), current_maxima.end());
      if (current_maxima != previous_maxima) {
        maxima_list->emplace_back(current_maxima);
        maxima_change->emplace_back(event);
      }
    }

    // DEBUG
    /*{
      HoughMatrixPtr hough_space_debug;
      hough_space_debug = new HoughMatrix;
      std::vector<circle> maxima_debug;

      std::memset(hough_space_debug, 0, sizeof(HoughMatrix));
      computeFullHoughSpace(event, hough_space_debug, points,
          gradient_magnitudes, gradient_directions);
      computeFullNMS(hough_space_debug, &maxima_debug);

      std::stable_sort(maxima_debug.begin(), maxima_debug.end());

      std::vector<circle> &current_maxima =
        maxima_list->back();

      bool different = false;
      if (maxima_debug.size() == current_maxima.size()) {
        for (size_t i = 0; i < maxima_debug.size(); ++i) {
          if (maxima_debug[i] != current_maxima[i]) {
            different = true;
            break;
          }
        }
      } else {
        different = true;
      }
      
      LOG(INFO) << "==================== "
          << event << " / " << points.size();
      for (size_t j = 0; j < maxima_debug.size(); ++j) {
        LOG(INFO) << "d: " << maxima_debug[j].r << " "
            << maxima_debug[j].x << " " << maxima_debug[j].y;
      }
      for (size_t j = 0; j < current_maxima.size(); ++j) {
        LOG(INFO) << "i: " << current_maxima[j].r << " "
            << current_maxima[j].x << " " << current_maxima[j].y;
      }

      if (different) {
        for (size_t j = 0; j < maxima_debug.size(); ++j) {
          LOG(INFO) << "houghspace: " << maxima_debug[j].r << " "
              << maxima_debug[j].x << " " << maxima_debug[j].y;
          for (size_t m = maxima_debug[j].r - 1; m <= maxima_debug[j].r + 1; ++m) {
            for (size_t n = maxima_debug[j].y - 1; n <= maxima_debug[j].y + 1; ++n) {
              std::string line = "";
              for (size_t p = maxima_debug[j].x - 1; p <= maxima_debug[j].x + 1; ++p) {
                line += std::to_string(hough_space[m][n][p]) + " "; 
              }
              LOG(INFO) << line;
            }
            LOG(INFO) << "-----------------";
          }
        }

        for (size_t j = 0; j < kHoughSpaceRadius; ++j) {
          for (size_t m = 0; m < kHoughSpaceHeight; ++m) {
            for (size_t n = 0; n < kHoughSpaceWidth; ++n) {
              if (hough_space_debug[j][m][n] != hough_space[j][m][n]) {
                LOG(INFO) << j << " " << m << " " << n << " "
                  << "dH: " << hough_space_debug[j][m][n]
                  << " iH " << hough_space[j][m][n];
              }
            }
          }
        }

        exit(0);
      }

      delete hough_space_debug;
    }*/
  }
}

// Computing a full Non-Maximum Suppression for a given current Hough Space.
void Detector::computeFullNMS(
    HoughMatrixPtr hough_space, std::vector<circle> *maxima) {
  CHECK_NOTNULL(maxima);

  // New detected maxima and their value.
  std::vector<circle> candidate_maxima;
  std::vector<int> candidate_maxima_value;

  // Checking every positions and radius hypothesis.
  for (int32_t r = 1; r < kHoughSpaceRadius - 1; ++r) {
    for (int32_t y = 1; y < kHoughSpaceHeight - 1; ++y) {
      for (int32_t x = 1; x < kHoughSpaceWidth - 1; ++x) {
        if (hough_space[r][y][x] > FLAGS_hough_threshold) {
          if (isLocalMaxima(hough_space, r, y, x)) {
            // Add as a possible maximum to the list.
            candidate_maxima.emplace_back(x, y, r);
            candidate_maxima_value.emplace_back(hough_space[r][y][x]);
          }
        }
      }    
    }
  }

  // This applies a suppression radius to the list of maxima hypothesis, to
  // only get the most prominent ones.
  applySuppressionRadius(candidate_maxima, candidate_maxima_value, maxima);
}

// Computing a full Hough Space based on a current set of events and their
// respective Hough parameters.
void Detector::computeFullHoughSpace(
    size_t index, HoughMatrixPtr hough_space, const std::vector<point>& points,
    const std::vector<int32_t>& gradient_magnitudes,
    const std::vector<int32_t>& gradient_directions) {
  CHECK_GE(index, FLAGS_hough_window_size - 1);

  // Looping over all events that have an influence on the current total
  // Hough space (i.e. fall in the detection window).
  for (size_t i = index - FLAGS_hough_window_size + 1; i <= index; ++i) {
    if (!gradient_magnitudes[i]) {
      continue;
    }

    const int32_t line_idx = gradient_directions[i];
    for (size_t j = 0; j < lines_xy[line_idx].size(); ++j) {
      const int32_t r = (j >> 1) + 1;
      const int32_t x = points[i].x + lines_xy[line_idx][j].x;
      const int32_t y = points[i].y + lines_xy[line_idx][j].y;
      if ((x >= 1) && (x < kHoughSpaceWidth - 1) && 
          (y >= 1) && (y < kHoughSpaceHeight - 1)) {
        ++hough_space[r][y][x];
      }
    }
  }
}

// Event preprocessing prior to first HT.
void Detector::eventPreProcessing(
    const dvs_msgs::EventArray::ConstPtr& orig_msg,
    std::vector<point>* points, std::vector<double>* times) {
  const int num_events = orig_msg->events.size();

  // How many points do we carry over from previous messages.
  const size_t num_last_points = last_points.size();

  // Preallocate more space than needed.
  points->reserve(num_events + num_last_points);
  times->reserve(num_events + num_last_points);
  points->resize(num_last_points);
  times->resize(num_last_points);

  // Concatenate previous points with the new ones.
  std::move(last_points.begin(), last_points.end(), points->begin());
  std::move(last_times.begin(), last_times.end(), times->begin());

  // Preprocess events and potentially subsample.
  size_t filtered = 0;
  for (int i = 0; i < num_events; i += FLAGS_event_subsample_factor) {
    const dvs_msgs::Event& e = orig_msg->events[i];
    double time = e.ts.toSec();

    // Undistort events based on camera calibration.
    //const size_t x = static_cast<size_t>(event_undist_map_x_(e.y, e.x));
    //const size_t y = static_cast<size_t>(event_undist_map_y_(e.y, e.x));
    const size_t x = static_cast<size_t>(e.x);
    const size_t y = static_cast<size_t>(e.y);

    // Empty pixel counter based on time to live
    while (!filter_grid_[y][x].empty() && 
        filter_grid_[y][x].front() + kMaxTimeToLive < num_messages) {
      filter_grid_[y][x].pop();
    }

    // Add current trigger
    filter_grid_[y][x].push(num_messages);

    // Check if we've exceeded the triggering threshold
    if (filter_grid_[y][x].size() > kMaxEventRate) {
      ++filtered;
      continue;
    }

    points->emplace_back(x, y);
    times->emplace_back(time);
  }

  VLOG(2) << "Filtered " << filtered << " pixels due to over-triggering.";
}

void Detector::addMaximaInRadius(
    int32_t r, int32_t y, int32_t x, HoughMatrixPtr hough_space,
    std::vector<circle>* new_maxima, std::vector<int>* new_maxima_value,
    bool skip_center) {
  int32_t m_l = std::max(r - FLAGS_hough_nms_radius, 1);
  int32_t m_r = std::min(r + FLAGS_hough_nms_radius + 1, kHoughSpaceRadius - 1);

  int32_t n_l = std::max(y - FLAGS_hough_nms_radius, 1);
  int32_t n_r = std::min(y + FLAGS_hough_nms_radius + 1, kHoughSpaceHeight - 1);

  int32_t p_l = std::max(x - FLAGS_hough_nms_radius, 1);
  int32_t p_r = std::min(x + FLAGS_hough_nms_radius + 1, kHoughSpaceWidth - 1);

  for (int32_t m = m_l; m <= m_r; ++m) {
    for (int32_t n = n_l; n <= n_r; ++n) {
      for (int32_t p = p_l; p <= p_r; ++p) {
        if (skip_center && (m == r) && (n == y) && (p == x)) {
          continue;
        }

        if ((hough_space[m][n][p] > FLAGS_hough_threshold) &&
            isLocalMaxima(hough_space, m, n, p)) {
          new_maxima->emplace_back(p, n, m);
          new_maxima_value->emplace_back(hough_space[m][n][p]);
        }
      }
    }
  }
}

void Detector::applySuppressionRadius(
    const std::vector<circle>& candidate_maxima,
    const std::vector<int>& candidate_maxima_value,
    std::vector<circle>* maxima) {
  // Create an index of all known maxima.
  std::vector<int> candidate_maxima_index(candidate_maxima_value.size());
  for (int i = 0; i < candidate_maxima_value.size(); i++) {
    candidate_maxima_index[i] = i;
  }

  // Sort the index of all currently known maxima. Sort them by:
  // 1) value 2) radius 3) x 4) y.
  std::stable_sort(
      candidate_maxima_index.begin(), candidate_maxima_index.end(),
      [&candidate_maxima_value, &candidate_maxima](const int i1, const int i2) {
        if (candidate_maxima_value[i1] != candidate_maxima_value[i2]) {
          return candidate_maxima_value[i1] > candidate_maxima_value[i2];
        } else {
          return candidate_maxima[i1] > candidate_maxima[i2];
        }
      });

  // Clear buffer of current maxima to re-add them later on.
  maxima->clear();

  // Loop over all maxima according to the sorted order.
  for (int i = 0; i < candidate_maxima.size(); i++) {
    const circle& candidate_maximum =
        candidate_maxima[candidate_maxima_index[i]];

    bool add_maximum = true;
    // Compare to all other maxima in the output buffer.
    for (int j = 0; j < maxima->size(); j++) {
      const circle& maximum = (*maxima)[j];

      // If no maximum in the output buffer is within the suppression radius,
      // the current maximum is kept and added to the output buffer.
      int distance =
          (maximum.r - candidate_maximum.r) * (maximum.r - candidate_maximum.r) +
          (maximum.y - candidate_maximum.y) * (maximum.y - candidate_maximum.y) +
          (maximum.x - candidate_maximum.x) * (maximum.x - candidate_maximum.x);

        if (distance < hough_nms_radius3_) {
          add_maximum = false;
          break;
        }
    }

    // Adding accepted maxima to the output buffer.
    if (add_maximum) {
      maxima->push_back(candidate_maximum);
    }
  }
}

// Check if the center value is a maxima.
inline bool Detector::isLocalMaxima(
    HoughMatrixPtr hough_space, int32_t r, int32_t y, int32_t x) {
  // Loop over the 8-connected neighborhood.
  for (int32_t m = r - 1; m <= r + 1; ++m) {
    for (int32_t n = y - 1; n <= y + 1; ++n) {
      for (int32_t p = x - 1; p <= x + 1; ++p) {
        if ((m != r) || (n != y) || (p != x)) {
          if (hough_space[m][n][p] >= hough_space[r][y][x]) {
            return false;
          }
        }
      }
    }
  }

  return true;
}

// Callback function for saving current greyscale image frame.
void Detector::imageCallback(const sensor_msgs::Image::ConstPtr& msg) {
  // Get greyscale image.
  cv::Mat cv_image_raw;
  cv_bridge::CvImagePtr cv_ptr;

  try {
    cv_ptr = cv_bridge::toCvCopy(msg);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  // Republish the camera image at the same rate for visualization.
  cv::imshow("Camera image", cv_ptr->image);
  cv::waitKey(1);
}

}  // namespace hough2map
