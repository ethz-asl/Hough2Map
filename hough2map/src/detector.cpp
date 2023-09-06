#include "hough2map/detector.h"

#include <chrono>
#include <ros/package.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

DEFINE_string(rosbag_path, "", "Rosbag to process.");
DEFINE_string(event_topic, "/dvs/events", "Topic for event messages.");
DEFINE_string(image_topic, "/dvs/image_raw", "Topic for image messages.");
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

DEFINE_bool(odometry_available, true, "A GPS Odometry is available");
DEFINE_double(
    odometry_event_alignment, 0,
    "Manual time synchronization to compensate misalignment between "
    "odometry and DVS timestamps");
DEFINE_double(camera_offset_x, 0, "Camera offset along the train axis");
DEFINE_double(
    camera_offset_y, 0,
    "Camera offset perpendicular to the left of the train axis");
DEFINE_double(buffer_size_s, 30, "Size of the odometry buffer in seconds");

namespace hough2map {

const int Detector::kHoughRadiusResolution;
const int Detector::kHoughAngularResolution;
const int Detector::kHoughMinAngle;
const int Detector::kHoughMaxAngle;

Detector::Detector(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private)
    : nh_(nh), nh_private_(nh_private) {
  // Checking that flags have reasonable values.
  CHECK_GE(FLAGS_hough_window_size, 1);
  CHECK_GT(FLAGS_event_subsample_factor, 0);
  CHECK_GE(FLAGS_buffer_size_s, 1);
  CHECK_GT(FLAGS_hough_threshold, 0);
  CHECK(!FLAGS_rosbag_path.empty());

  // Set initial status to false until first message is processed.
  initialized = false;
  last_points_pos.resize(2, 0);
  last_points_neg.resize(2, 0);
  last_times_pos.resize(0);
  last_times_neg.resize(0);

  // Output file for the map data.
  if (FLAGS_map_output) {
    map_file.open(map_file_path);

    if (map_file.is_open()) {
      map_file << "time,rho,theta" << std::endl;
    } else {
      LOG(FATAL) << "Could not open file:" << map_file_path << std::endl;
    }
  }

  debug_file.open(debug_file_path);
  CHECK(debug_file.is_open());

  // Timing statistics for performance evaluation.
  total_events_timing_us = 0.0;
  total_msgs_timing_ms = 0.0;
  total_events = 0;
  total_msgs = 0;

  // Import calibration file.
  loadCalibration();

  // Compute undistortion for given camera parameters.
  computeUndistortionMapping();

  // Precompute theta, sin and cos values.
  initializeSinCosMap(
      thetas_, polar_param_mapping_, kHoughMinAngle, kHoughMaxAngle,
      kHoughAngularResolution);
  hough_nms_radius2_ = FLAGS_hough_nms_radius * FLAGS_hough_nms_radius;

  // Initialize various transformation matrizes.
  initializeTransformationMatrices();

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

    cv::namedWindow("Detected lines (pos)", cv::WINDOW_NORMAL);
    cv::namedWindow("Detected lines (neg)", cv::WINDOW_NORMAL);
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
  if (FLAGS_map_output) {
    map_file.close();
  }
}

void Detector::initializeTransformationMatrices() {
  // Initialize transformation matrices.
  // Rotating camera relative to train.
  C_camera_train_ << 0, -1, 0, 1, 0, 0, 0, 0, 1;
  // GPS offset to center of train in meters.
  gps_offset_ << 1, 0, 0, 0, 1, -0.8, 0, 0, 1;
  // Camera offset to center of train in meters.
  camera_train_offset_ << 1, 0, -FLAGS_camera_offset_x, 0, 1,
      FLAGS_camera_offset_y, 0, 0, 1;
}

// Function to precompute angles, sin and cos values for a vectorized version
// of the HT. Templated to deal with float and double accuracy.
template <typename DerivedVec, typename DerivedMat>
void Detector::initializeSinCosMap(
    Eigen::EigenBase<DerivedVec>& angles,
    Eigen::EigenBase<DerivedMat>& sin_cos_map, const int kMinAngle,
    const int kMaxAngle, const int kNumSteps) {
  // Computing the angular step size.
  CHECK_GT(kNumSteps, 1);
  const double kDeltaTheta = (kMaxAngle - kMinAngle) / (kNumSteps - 1.0);
  // Resizing the respective output matrizes.
  angles.derived().resize(kNumSteps, 1);
  sin_cos_map.derived().resize(kNumSteps, 2);

  // Populating matrizes with angles and sin, cos values.
  for (int i = 0; i < kNumSteps; i++) {
    angles.derived()(i) = M_PI * (kMinAngle + i * kDeltaTheta) / 180.0;
    sin_cos_map.derived()(i, 0) = cos(angles.derived()(i));
    sin_cos_map.derived()(i, 1) = sin(angles.derived()(i));
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

// Processing incoming GPS position data.
void Detector::positionCallback(const custom_msgs::positionEstimate msg) {
  utm_coordinate current_position_utm = deg2utm(msg.latitude, msg.longitude);

  // Add raw_gps to current raw_gps buffer (with alighment for manual odometry
  // to DVS synchronization).
  const double kAlignedTimestamp =
      msg.header.stamp.toSec() + FLAGS_odometry_event_alignment;
  Eigen::Vector3d current_position_txy;
  current_position_txy[0] = kAlignedTimestamp;
  current_position_txy[1] = current_position_utm.x;
  current_position_txy[2] = current_position_utm.y;
  raw_gps_buffer_.push_back(current_position_txy);

  while (kAlignedTimestamp - raw_gps_buffer_.front()[0] > FLAGS_buffer_size_s) {
    raw_gps_buffer_.pop_front();
  }
}

// Processing velocity data and storing in buffer.
void Detector::velocityCallback(const custom_msgs::velocityEstimate msg) {
  // Manually synchronizing DVS to odometry timestamps.
  const double kAlignedTimestamp =
      msg.header.stamp.toSec() + FLAGS_odometry_event_alignment;
  // Add velocity to current velocity buffer.
  Eigen::Vector3d current_velocity_ten;
  current_velocity_ten[0] = kAlignedTimestamp;
  current_velocity_ten[1] = msg.velE;
  current_velocity_ten[2] = msg.velN;
  velocity_buffer_.push_back(current_velocity_ten);

  while (kAlignedTimestamp - velocity_buffer_.front()[0] >
         FLAGS_buffer_size_s) {
    velocity_buffer_.pop_front();
  }
}

void Detector::orientationCallback(const custom_msgs::orientationEstimate msg) {
  double yaw = msg.yaw * M_PI / 180.0;

  // Add orientation to current orientation buffer with manual time stamp
  // synchronization.
  const double kAlignedTimestamp =
      msg.header.stamp.toSec() + FLAGS_odometry_event_alignment;
  Eigen::Vector2d cur_orient;
  cur_orient[0] = kAlignedTimestamp;
  cur_orient[1] = yaw;
  orientation_buffer_.push_back(cur_orient);

  while (kAlignedTimestamp - orientation_buffer_.front()[0] >
         FLAGS_buffer_size_s) {
    orientation_buffer_.pop_front();
  }
}

void Detector::stepHoughTransform(
      const Eigen::MatrixXf& points, Detector::MatrixHough& hough_space,
      std::vector<hough2map::Detector::line> *last_maxima, bool initialized,
      std::vector<std::vector<hough2map::Detector::line>> *maxima_list,
      std::vector<size_t> *maxima_change) {
  CHECK_NOTNULL(last_maxima);
  CHECK_NOTNULL(maxima_list);
  CHECK_NOTNULL(maxima_change);

  // Pre-compute all the radii for each theta hypothesis. This parameter
  // represents one axis of the Hough space.
  Eigen::MatrixXi radii;
  radii.resize(kHoughAngularResolution, points.cols());
  radii = (polar_param_mapping_ * points).cast<int>();

  if (!initialized) {
    // Start from an empty hough space.
    hough_space.setZero();
    computeFullHoughSpace(FLAGS_hough_window_size - 1, hough_space, radii);

    // Initialize first set of maxima.
    maxima_list->emplace_back();
    computeFullNMS(hough_space, maxima_list->data());
  } else {
    maxima_list->emplace_back(*last_maxima);
  }

  // Perform computations iteratively for the rest of the events.
  iterativeNMS(points, hough_space, radii, maxima_list, maxima_change);

  // Store last set of maxima to have a starting point for the next message.
  *last_maxima = maxima_list->back();
}

void Detector::eventCallback(const dvs_msgs::EventArray::ConstPtr& msg) {
  CHECK_GE(msg->events.size(), 1);

  // Start timing.
  const auto kStartTime = std::chrono::high_resolution_clock::now();

  // Reshaping the event array into an Eigen matrix.
  Eigen::MatrixXf points_pos, points_neg;
  Eigen::VectorXd times_pos, times_neg;
  eventPreProcessing(msg, &points_pos, &points_neg, &times_pos, &times_neg);

  // The new points will now be the next old points.
  const int num_points_pos = points_pos.cols();
  const int num_points_neg = points_neg.cols();
  const int keep_pos = std::min(FLAGS_hough_window_size, num_points_pos);
  const int keep_neg = std::min(FLAGS_hough_window_size, num_points_neg);
  last_points_pos = points_pos.block(0, num_points_pos - keep_pos, 2, keep_pos);
  last_points_neg = points_neg.block(0, num_points_neg - keep_neg, 2, keep_neg);
  last_times_pos = times_pos.segment(num_points_pos - keep_pos, keep_pos);
  last_times_neg = times_neg.segment(num_points_neg - keep_neg, keep_neg);

  // Wait until we have enough points to initialize.
  if (num_points_pos < FLAGS_hough_window_size ||
      num_points_neg < FLAGS_hough_window_size) {
    return;
  }

  // Each event is treated as a timestep. For each of these timesteps we keep
  // the active set of maxima in the Hough Space. These are basically the line
  // detections at each timestep. This whole storage is pre-initialized to
  // make it ready for parallelizing the whole process.
  std::vector<std::vector<Detector::line>> maxima_list_pos;
  std::vector<std::vector<Detector::line>> maxima_list_neg;
  std::vector<size_t> maxima_change_pos;
  std::vector<size_t> maxima_change_neg;

  stepHoughTransform(
      points_pos, hough_space_pos, &last_maxima_pos, initialized,
      &maxima_list_pos, &maxima_change_pos);
  stepHoughTransform(
      points_neg, hough_space_neg, &last_maxima_neg, initialized,
      &maxima_list_neg, &maxima_change_neg);

  initialized = true;

  /*if (FLAGS_map_output) {
    for (const auto& maximas : maxima_list) {
      for (const Detector::line& maxima : maximas) {
        map_file << maxima.time << "," << maxima.r << "," << maxima.theta
                 << std::endl;
      }
    }
  }*/

  // Calculate statistics
  const size_t num_events = num_points_pos + num_points_neg;
  if (num_events > 0) {
    std::chrono::duration<double, std::micro> duration_us =
        std::chrono::high_resolution_clock::now() - kStartTime;

    total_events_timing_us += duration_us.count();
    total_msgs_timing_ms += duration_us.count() / 1000.0;

    total_events += num_events;
    total_msgs++;

    LOG(INFO) << detector_name_ << std::fixed << std::setprecision(2)
              << std::setfill(' ') << " speed: " << std::setw(6)
              << total_events_timing_us / total_events << " us/event | "
              << std::setw(6) << total_msgs_timing_ms / total_msgs
              << " ms/msg | " << std::setw(6) << num_events << " e/msg";
  }

  // If visualizations are turned on display them in the video stream.
  // NOTE: This slows everything down by a very large margin
  if (FLAGS_show_lines_in_video) {
    visualizeCurrentLineDetections(
        true, points_pos, maxima_list_pos, maxima_change_pos);
    visualizeCurrentLineDetections(
        false, points_neg, maxima_list_neg, maxima_change_neg);
  }
}

// Visualizing the current line detections and corresponding Hough space.
void Detector::visualizeCurrentLineDetections(
    bool polarity, const Eigen::MatrixXf& points, 
    const std::vector<std::vector<hough2map::Detector::line>>& maxima_list,
    const std::vector<size_t>& maxima_change) const {
  const int num_events = points.cols();

  // Predefined colors and strings
  const cv::Scalar color_scalar = 
      polarity ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
  const cv::Vec3b color_vec3b = 
      polarity ? cv::Vec3b(0, 0, 255) : cv::Vec3b(255, 0, 0);
  const std::string cv_window =
      polarity ? "Detected lines (pos)" : "Detected lines (neg)";

  // Getting the horizontal positions of all vertical line detections.
  size_t maxima_index = 0;
  const size_t step_size = FLAGS_show_lines_every_nth;
  for (size_t i = FLAGS_hough_window_size; i < num_events; i += step_size) {
    cv::Mat vis_frame(
        cv::Size(camera_resolution_width_, camera_resolution_height_), CV_8UC3,
        cv::Scalar(255, 255, 255));
    /*cv::remap(
        cur_greyscale_img_, vis_frame, image_undist_map_x_,
        image_undist_map_y_, cv::INTER_LINEAR);*/

    for (size_t j = i - FLAGS_hough_window_size + 1; j <= i; j++) {
      uint32_t x = static_cast<uint32_t>(points(0, j));
      uint32_t y = static_cast<uint32_t>(points(1, j));
      vis_frame.at<cv::Vec3b>(y, x) = color_vec3b;
    }

    if ((maxima_index < maxima_change.size()) &&
        (i >= maxima_change[maxima_index])) {
      ++maxima_index;
    }

    for (auto& maxima : maxima_list[maxima_index]) {
      drawPolarCorLine(vis_frame, maxima.r, maxima.theta, color_scalar);
    }

    cv::imshow(cv_window, vis_frame);
    cv::waitKey(1);
  }
}

// Performing itterative Non-Maximum suppression on the current batch of
// events based on a beginning Hough Space.
void Detector::iterativeNMS(
    const Eigen::MatrixXf& points, MatrixHough& hough_space, 
    const Eigen::MatrixXi& radii,
    std::vector<std::vector<Detector::line>>* maxima_list,
    std::vector<size_t> *maxima_change) {
  CHECK_NOTNULL(maxima_list);
  CHECK_NOTNULL(maxima_change)->clear();
  CHECK_GT(maxima_list->size(), 0u);

  std::vector<hough2map::Detector::line> new_maxima;
  std::vector<int> new_maxima_value;
  const int num_events = points.cols();

  for (int event = FLAGS_hough_window_size; event < num_events; event++) {
    // Take the maxima at the previous timestep.
    std::vector<hough2map::Detector::line> &previous_maxima =
        maxima_list->back();

    // Incrementing the accumulator cells for the current event.
    for (int angle = 0; angle < kHoughAngularResolution; angle++) {
      const int radius = radii(angle, event);
      if (radius >= 0 && radius < kHoughRadiusResolution) {
        hough_space(radius, angle)++;
      }
    }

    // Decrement the accumulator cells for the event to be removed.
    for (int angle = 0; angle < kHoughAngularResolution; angle++) {
      const int radius = radii(angle, event - FLAGS_hough_window_size);
      if (radius >= 0 && radius < kHoughRadiusResolution) {
        hough_space(radius, angle)--;
      }
    }

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
    for (int angle = 0; angle < kHoughAngularResolution; ++angle) {
      const int radius = radii(angle, event);
      if ((radius < 0) || (radius >= kHoughRadiusResolution)) {
        continue;
      }

      // If any of the surrounding ones are equal the center
      // for sure it is not a local maximum.
      bool skip_center = false;

      // Iterate over neighbourhood to check if we might have
      // supressed a surrounding maximum by growing.
      const int m_l = std::max(angle - 1, 0);
      const int m_r = std::min(angle + 1, kHoughAngularResolution - 1);
      const int n_l = std::max(radius - 1, 0);
      const int n_r = std::min(radius + 1, kHoughRadiusResolution - 1);
      for (int m = m_l; m <= m_r; ++m) {
        for (int n = n_l; n <= n_r; ++n) {
          // The center is a separate case.
          if ((m == angle) && (n == radius)) {
            continue;
          }

          // Compare point to its neighbors.
          if (hough_space(radius, angle) == hough_space(n, m)) {
            skip_center = true;
            // Compare to all known maxima from the previous timestep.
            for (size_t i = 0; i < previous_maxima.size(); ++i) {
              if ((n == previous_maxima[i].r) &&
                  (m == previous_maxima[i].theta_idx)) {
                // We need to discard an old maximum.
                changed = true;
                discard[i] = true;

                // And add a new one.
                addMaximaInRadius(
                    m, n, hough_space, &new_maxima, &new_maxima_value);
                break;
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
      if ((hough_space(radius, angle) > FLAGS_hough_threshold) &&
          isLocalMaxima(hough_space, angle, radius)) {
        bool add_maximum = true;
        // Check if it was a maximum previously.
        for (const auto& maximum : previous_maxima) {
          if ((radius == maximum.r) && (angle == maximum.theta_idx)) {
            add_maximum = false;
            break;
          }
        }

        // If required, add it to the list.
        if (add_maximum) {
          new_maxima.emplace_back(radius, thetas_(angle), angle);
          new_maxima_value.emplace_back(hough_space(radius, angle));
        }
      }
    }

    // For accumulator cells that got decremented.
    for (int angle = 0; angle < kHoughAngularResolution; ++angle) {
      const int radius = radii(angle, event - FLAGS_hough_window_size);
      if ((radius < 0) || (radius >= kHoughRadiusResolution)) {
        continue;
      }

      // If decremented accumulator cell was previously a maximum, remove it.
      // If it's still a maximum, we will deal with it later.
      bool skip_neighborhood = false;
      for (size_t k = 0; k < previous_maxima.size(); ++k) {
        if ((radius == previous_maxima[k].r) &&
            (angle == previous_maxima[k].theta_idx)) {
          // Mark as discarded since we will already have added it
          // in the next step if it still is above the threshold.
          changed = true;
          discard[k] = true;

          // Re-add to list of possible maxima for later pruning.
          addMaximaInRadius(
              angle, radius, hough_space, &new_maxima, &new_maxima_value);

          // The neighborhood of this accumulator cell has been checked as part of
          // addMaximaInRadius, so no need to do it again.
          skip_neighborhood = true;
          break;
        }
      }

      if (!skip_neighborhood) {
        // Iterate over neighbourhood to check if we might have
        // created a new local maxima by decreasing.
        const int m_l = std::max(angle - 1, 0);
        const int m_r = std::min(angle + 1, kHoughAngularResolution - 1);
        const int n_l = std::max(radius - 1, 0);
        const int n_r = std::min(radius + 1, kHoughRadiusResolution - 1);
        for (int m = m_l; m <= m_r; m++) {
          for (int n = n_l; n <= n_r; n++) {
            // The center is a separate case.
            if ((m == angle) && (n == radius)) {
              continue;
            }

            // Any neighbor points now larger and a maximum?
            if ((hough_space(radius, angle) + 1 == hough_space(n, m)) &&
                (hough_space(n, m) > FLAGS_hough_threshold) &&
                isLocalMaxima(hough_space, m, n)) {
              // Add to temporary storage.
              new_maxima.emplace_back(n, thetas_(m), m);
              new_maxima_value.emplace_back(hough_space(n, m));
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
        std::vector<Detector::line> current_maxima;
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
          new_maxima_value.emplace_back(hough_space(
              previous_maxima[i].r, previous_maxima[i].theta_idx));
        }
      }

      // PHASE 2 - Detect and apply suppression
      std::vector<Detector::line> current_maxima;

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
            for (const Detector::line& current_maximum : current_maxima) {
              if ((current_maximum.r == previous_maxima[i].r) &&
                  (current_maximum.theta_idx == previous_maxima[i].theta_idx)) {
                found = true;
                break;
              }
            }

            if (!found) {
              discard[i] = true;
              addMaximaInRadius(
                  previous_maxima[i].theta_idx, previous_maxima[i].r,
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
      MatrixHough hough_space_debug;
      std::vector<Detector::line> maxima_debug;

      hough_space_debug.setZero();
      computeFullHoughSpace(event, hough_space_debug, radii);
      computeFullNMS(hough_space_debug, &maxima_debug);

      std::stable_sort(maxima_debug.begin(), maxima_debug.end());

      std::vector<hough2map::Detector::line> &current_maxima =
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

      if (different) {
        LOG(INFO) << "==================== " << event;
        for (size_t j = 0; j < maxima_debug.size(); ++j) {
          LOG(INFO) << "d: " << maxima_debug[j].r << " "
              << maxima_debug[j].theta_idx;
        }
        for (size_t j = 0; j < current_maxima.size(); ++j) {
          LOG(INFO) << "i: " << current_maxima[j].r << " "
              << current_maxima[j].theta_idx;
        }

        for (int i = 0; i < hough_space.rows(); ++i) {
          std::stringstream row;
          row << std::setw(3) << i << ": ";
          for (int j = 0; j < hough_space.cols(); ++j) {
            row << std::setw(2) << hough_space(i, j) << ", ";
          }
          LOG(INFO) << row.str();
        }

        exit(0);
      }
    }*/
  }
}

// Computing a full Non-Maximum Suppression for a given current Hough Space.
void Detector::computeFullNMS(
    const MatrixHough& hough_space, std::vector<Detector::line> *maxima) {
  CHECK_NOTNULL(maxima);

  // New detected maxima and their value.
  std::vector<hough2map::Detector::line> candidate_maxima;
  std::vector<int> candidate_maxima_value;

  // Checking every angle and radius hypothesis.
  for (int i = 0; i < kHoughAngularResolution; i++) {
    for (int j = 0; j < kHoughRadiusResolution; j++) {
      // Check in the Hough space, whether this hypothesis is larger than the
      // threshold and larger than its neighbors.
      if (hough_space(j, i) > FLAGS_hough_threshold) {
        if (isLocalMaxima(hough_space, i, j)) {
          // Add as a possible maximum to the list.
          candidate_maxima.emplace_back(j, thetas_(i), i);
          candidate_maxima_value.emplace_back(hough_space(j, i));
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
    size_t index, MatrixHough& hough_space, const Eigen::MatrixXi& radii) {
  CHECK_GE(index, FLAGS_hough_window_size - 1);

  // Looping over all events that have an influence on the current total
  // Hough space (i.e. fall in the detection window).
  for (int i = index - FLAGS_hough_window_size + 1; i <= index; i++) {
    for (int j = 0; j < kHoughAngularResolution; j++) {
      const int radius = radii(j, i);
      if (radius >= 0 && radius < kHoughRadiusResolution) {
        hough_space(radius, j)++;
      }
    }
  }
}

// Event preprocessing prior to first HT.
void Detector::eventPreProcessing(
    const dvs_msgs::EventArray::ConstPtr& orig_msg,
    Eigen::MatrixXf* points_pos, Eigen::MatrixXf* points_neg,
    Eigen::VectorXd* times_pos, Eigen::VectorXd* times_neg) {
  const int num_events = orig_msg->events.size();

  // How many points do we carry over from previous messages.
  const int num_last_points_pos = last_points_pos.cols();
  const int num_last_points_neg = last_points_neg.cols();

  // Preallocate more space than needed.
  points_pos->resize(2, num_events + num_last_points_pos);
  points_neg->resize(2, num_events + num_last_points_neg);
  times_pos->resize(num_events + num_last_points_pos);
  times_neg->resize(num_events + num_last_points_neg);

  // Concatenate previous points with the new ones.
  points_pos->block(0, 0, 2, num_last_points_pos) = last_points_pos;
  points_neg->block(0, 0, 2, num_last_points_neg) = last_points_neg;
  times_pos->segment(0, num_last_points_pos) = last_times_pos;
  times_neg->segment(0, num_last_points_neg) = last_times_neg;

  // Filter grid and initialization
  const size_t grid_size = 8;
  const size_t max_events = 8;
  const double event_ttl = 0.005;
  std::queue<double> filter_grid[480 / grid_size][640 / grid_size];
  for (int i = 0; i < num_last_points_pos; ++i) {
    size_t grid_x = last_points_pos(0, i) / grid_size;
    size_t grid_y = last_points_pos(1, i) / grid_size;
    filter_grid[grid_y][grid_x].push(last_times_pos(i));
  }
  for (int i = 0; i < num_last_points_neg; ++i) {
    size_t grid_x = last_points_neg(0, i) / grid_size;
    size_t grid_y = last_points_neg(1, i) / grid_size;
    filter_grid[grid_y][grid_x].push(last_times_neg(i));
  }

  // Preprocess events and potentially subsample.
  int count_pos = num_last_points_pos;
  int count_neg = num_last_points_neg;
  for (int i = 0; i < num_events; i += FLAGS_event_subsample_factor) {
    const dvs_msgs::Event& e = orig_msg->events[i];

    // Seemingly broken pixels in the DVS (millions of exactly equal events at
    // once at random). This needs to be adapted if you use another device.
    /*if (((e.x == 19) && (e.y == 18)) || ((e.x == 43) && (e.y == 72)) ||
        ((e.x == 89) && (e.y == 52)) || ((e.x == 25) && (e.y == 42)) ||
        ((e.x == 61) && (e.y == 71)) || ((e.x == 37) && (e.y == 112))) {
      continue;
    }*/

    double time = e.ts.toSec();

    // Undistort events based on camera calibration.
    float x = event_undist_map_x_(e.y, e.x);
    float y = event_undist_map_y_(e.y, e.x);

    size_t grid_x = x / grid_size;
    size_t grid_y = y / grid_size;
    while (!filter_grid[grid_y][grid_x].empty() && 
        filter_grid[grid_y][grid_x].front() + event_ttl < time) {
      filter_grid[grid_y][grid_x].pop();
    }

    if (filter_grid[grid_y][grid_x].size() < max_events) {
      filter_grid[grid_y][grid_x].push(time);
    } else {
      continue;
    }
    
    // Sort by polarity as we calculate two separate HTs, since lines are
    // usually one homogeneous transition in intensity across.
    if (e.polarity) {
      (*points_pos)(0, count_pos) = x;
      (*points_pos)(1, count_pos) = y;
      (*times_pos)(count_pos) = time;
      count_pos++;
    } else {
      (*points_neg)(0, count_neg) = x;
      (*points_neg)(1, count_neg) = y;
      (*times_neg)(count_neg) = time;
      count_neg++;
    }
  }

  points_pos->conservativeResize(2, count_pos);
  points_neg->conservativeResize(2, count_neg);
  times_pos->conservativeResize(count_pos);
  times_neg->conservativeResize(count_neg);
}

void Detector::drawPolarCorLine(
    cv::Mat& image_space, float rho, float theta, cv::Scalar color) const {
  // Function to draw a line based on polar coordinates
  cv::Point pt1, pt2;
  const double a = cos(theta), b = sin(theta);
  const double x0 = a * rho, y0 = b * rho;
  pt1.x = cvRound(x0 + 1000 * (-b));
  pt1.y = cvRound(y0 + 1000 * (a));
  pt2.x = cvRound(x0 - 1000 * (-b));
  pt2.y = cvRound(y0 - 1000 * (a));
  cv::line(image_space, pt1, pt2, color, 3, cv::LINE_AA);
}

void Detector::addMaximaInRadius(
    int angle, int radius, const MatrixHough& hough_space,
    std::vector<hough2map::Detector::line>* new_maxima,
    std::vector<int>* new_maxima_value, bool skip_center) {
  int m_l = std::max(angle - FLAGS_hough_nms_radius, 0);
  int m_r = std::min(angle + FLAGS_hough_nms_radius + 1, kHoughAngularResolution);
  int n_l = std::max(radius - FLAGS_hough_nms_radius, 0);
  int n_r = std::min(radius + FLAGS_hough_nms_radius + 1, kHoughRadiusResolution);

  for (int m = m_l; m < m_r; m++) {
    for (int n = n_l; n < n_r; n++) {
      if (skip_center && (n == radius) && (m == angle)) {
        continue;
      }

      if ((hough_space(n, m) > FLAGS_hough_threshold) &&
          isLocalMaxima(hough_space, m, n)) {
        new_maxima->emplace_back(n, thetas_(m), m);
        new_maxima_value->emplace_back(hough_space(n, m));
      }
    }
  }
}

void Detector::applySuppressionRadius(
    const std::vector<hough2map::Detector::line>& candidate_maxima,
    const std::vector<int>& candidate_maxima_value,
    std::vector<hough2map::Detector::line>* maxima) {
  // Create an index of all known maxima.
  std::vector<int> candidate_maxima_index(candidate_maxima_value.size());
  for (int i = 0; i < candidate_maxima_value.size(); i++) {
    candidate_maxima_index[i] = i;
  }

  // Sort the index of all currently known maxima. Sort them by: 1. value; 2.
  // rho value; 3. theta value.
  std::stable_sort(
      candidate_maxima_index.begin(), candidate_maxima_index.end(),
      [&candidate_maxima_value, &candidate_maxima](const int i1, const int i2) {
        if (candidate_maxima_value[i1] != candidate_maxima_value[i2]) {
          return candidate_maxima_value[i1] > candidate_maxima_value[i2];
        } else {
          if (candidate_maxima[i1].r != candidate_maxima[i2].r) {
            return candidate_maxima[i1].r > candidate_maxima[i2].r;
          } else {
            return candidate_maxima[i1].theta_idx > candidate_maxima[i2].theta_idx;
          }
        }
      });

  // Clear buffer of current maxima to re-add them later on.
  maxima->clear();

  // Loop over all maxima according to the sorted order.
  for (int i = 0; i < candidate_maxima.size(); i++) {
    const Detector::line& candidate_maximum =
        candidate_maxima[candidate_maxima_index[i]];

    bool add_maximum = true;
    // Compare to all other maxima in the output buffer.
    for (int j = 0; j < maxima->size(); j++) {
      const Detector::line& maximum = (*maxima)[j];

      // If no maximum in the output buffer is within the suppression radius,
      // the current maximum is kept and added to the output buffer.
      int distance =
          (maximum.r - candidate_maximum.r) * (maximum.r - candidate_maximum.r) +
          (maximum.theta_idx - candidate_maximum.theta_idx) *
              (maximum.theta_idx - candidate_maximum.theta_idx);

        if (distance < hough_nms_radius2_) {
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
bool Detector::isLocalMaxima(
    const Eigen::MatrixXi& hough_space, int i, int radius) {
  // Define the 8-connected neighborhood.
  const int m_l = std::max(i - 1, 0);
  const int m_r = std::min(i + 1, (int)hough_space.cols() - 1);

  const int n_l = std::max(radius - 1, 0);
  const int n_r = std::min(radius + 1, (int)hough_space.rows() - 1);

  // Loop over all neighborhood points
  for (int m = m_l; m <= m_r; m++) {
    for (int n = n_l; n <= n_r; n++) {
      if ((m != i) || (n != radius)) {
        if (hough_space(n, m) >= hough_space(radius, i)) {
          // If anyone was larger or equal, it's not a maximum.
          return false;
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

// Generalized buffer query function for all odometry buffers.
template <class S, int rows, int cols>
Eigen::Matrix<S, rows, cols> Detector::queryOdometryBuffer(
    const double query_time,
    const std::deque<Eigen::Matrix<S, rows, cols>>& odometry_buffer) {
  // Getting data for current time from respective odometry buffer.

  // Check that required timestamp is within buffer.
  CHECK_GT(query_time, odometry_buffer[0][0])
      << ": Query time out of range of buffer!";

  if (query_time >= odometry_buffer[odometry_buffer.size() - 1][0]) {
    return odometry_buffer[odometry_buffer.size() - 1];
  }

  // Finding the the upper closest and lower closest data points for
  // interpolation.
  auto lower_it =
      std::upper_bound(
          odometry_buffer.begin(), odometry_buffer.end(), query_time,
          [](double lhs, Eigen::Matrix<S, rows, cols> rhs) -> bool {
            return lhs < rhs[0];
          }) -
      1;

  auto upper_it = std::lower_bound(
      odometry_buffer.begin(), odometry_buffer.end(), query_time,
      [](Eigen::Matrix<S, rows, cols> lhs, double rhs) -> bool {
        return lhs[0] < rhs;
      });

  // Interpolate datapoint.
  double delta =
      (query_time - (*lower_it)[0]) / ((*upper_it)[0] - (*lower_it)[0]);

  Eigen::Matrix<S, rows, cols> interpolation_result =
      (*lower_it) + delta * ((*upper_it) - (*lower_it));

  return interpolation_result;
}

// Function for converting lat/lon to UTM messages.
Detector::utm_coordinate Detector::deg2utm(double la, double lo) {
  utm_coordinate result;

  constexpr double sa = 6378137.000000;
  constexpr double sb = 6356752.314245;

  constexpr double e2 = std::sqrt((sa * sa) - (sb * sb)) / sb;

  constexpr double e2squared = e2 * e2;

  constexpr double c = (sa * sa) / sb;

  const double lat = la * M_PI / 180;
  const double lon = lo * M_PI / 180;

  const double H_d = (lo / 6) + 31;
  int H;

  if (H_d > 0) {
    H = std::floor(H_d);
  } else {
    H = std::ceil(H_d);
  }

  const int S = (H * 6) - 183;
  const double deltaS = lon - (S * (M_PI / 180));

  char letter;

  if (la < -72)
    letter = 'C';
  else if (la < -64)
    letter = 'D';
  else if (la < -56)
    letter = 'E';
  else if (la < -48)
    letter = 'F';
  else if (la < -40)
    letter = 'G';
  else if (la < -32)
    letter = 'H';
  else if (la < -24)
    letter = 'J';
  else if (la < -16)
    letter = 'K';
  else if (la < -8)
    letter = 'L';
  else if (la < -0)
    letter = 'M';
  else if (la < 8)
    letter = 'N';
  else if (la < 16)
    letter = 'P';
  else if (la < 24)
    letter = 'Q';
  else if (la < 32)
    letter = 'R';
  else if (la < 40)
    letter = 'S';
  else if (la < 48)
    letter = 'T';
  else if (la < 56)
    letter = 'U';
  else if (la < 64)
    letter = 'V';
  else if (la < 72)
    letter = 'W';
  else
    letter = 'X';

  const double a = std::cos(lat) * std::sin(deltaS);
  const double epsilon = 0.5 * std::log((1 + a) / (1 - a));
  const double nu = std::atan(std::tan(lat) / std::cos(deltaS)) - lat;
  const double v =
      (c / std::sqrt((1 + (e2squared * (std::cos(lat) * std::cos(lat)))))) *
      0.9996;
  const double ta =
      (e2squared / 2) * epsilon * epsilon * (std::cos(lat) * std::cos(lat));
  const double a1 = std::sin(2 * lat);
  const double a2 = a1 * (std::cos(lat) * std::cos(lat));
  const double j2 = lat + (a1 / 2);
  const double j4 = ((3 * j2) + a2) / 4;
  const double j6 = ((5 * j4) + (a2 * (std::cos(lat) * std::cos(lat)))) / 3;

  const double alfa = (3.0 / 4) * e2squared;
  const double beta = (5.0 / 3) * alfa * alfa;
  const double gama = (35.0 / 27) * alfa * alfa * alfa;

  const double Bm = 0.9996 * c * (lat - alfa * j2 + beta * j4 - gama * j6);

  const double xx = epsilon * v * (1 + (ta / 3)) + 500000;
  double yy = nu * v * (1 + ta) + Bm;

  if (yy < 0) {
    yy = 9999999 + yy;
  }

  result.x = xx;
  result.y = yy;
  result.zone = std::to_string(H) + letter;

  return result;
}

}  // namespace hough2map
