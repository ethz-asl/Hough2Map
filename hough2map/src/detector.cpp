#include "hough2map/detector.h"

#include <chrono>
#include <ros/package.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

DEFINE_string(rosbag_path, "", "Rosbag to process.");
DEFINE_string(event_topic, "/dvs/events", "Topic for event messages.");
DEFINE_string(image_topic, "/dvs/image_raw", "Topic for image messages.");

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
DEFINE_bool(
    perform_camera_undistortion, true,
    "Undistort event data according to camera calibration");
DEFINE_double(buffer_size_s, 30, "Size of the odometry buffer in seconds");

namespace hough2map {
Detector::Detector(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private)
    : nh_(nh), nh_private_(nh_private) {
  // Checking that flags have reasonable values.
  CHECK_GE(FLAGS_hough_window_size, 1);
  CHECK_GT(FLAGS_event_subsample_factor, 0);
  CHECK_GE(FLAGS_buffer_size_s, 1);
  CHECK_GT(FLAGS_hough_threshold, 0);
  CHECK(!FLAGS_rosbag_path.empty());

  // Output file for the map data.
  if (FLAGS_map_output) {
    map_file.open(map_file_path);

    if (map_file.is_open()) {
      map_file << "id,type,time,x,y,orientation,velocity,weight\n";

    } else {
      LOG(FATAL) << "Could not open file:" << map_file_path << std::endl;
    }
  }

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

    cv::namedWindow("Detected poles", cv::WINDOW_NORMAL);
    cv::namedWindow("Hough space (pos)", cv::WINDOW_NORMAL);

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
  CHECK_GT(kNumSteps - 1, 0);
  const double kDeltaTheta = (kMaxAngle - kMinAngle) / (kNumSteps - 1);
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
      package_path + "/share/" + calibration_file_name;

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

void Detector::eventCallback(const dvs_msgs::EventArray::ConstPtr& msg) {
  const auto kStartTime = std::chrono::high_resolution_clock::now();

  feature_msg_.header = msg->header;
  feature_msg_.width = msg->width;
  feature_msg_.height = msg->height;

  int num_events = msg->events.size();
  CHECK_GE(num_events, 1);

  // If initialized then make sure the last FLAGS_hough_window_size events
  // are prepended to the current list of events and remove older events.
  if (feature_msg_.events.size() > FLAGS_hough_window_size) {
    std::copy(
        feature_msg_.events.end() - FLAGS_hough_window_size,
        feature_msg_.events.end(), feature_msg_.events.begin());
    feature_msg_.events.resize(FLAGS_hough_window_size);
  }

  // Reshaping the event array into an Eigen matrix.
  Eigen::MatrixXf points;
  eventPreProcessing(msg, points);

  // Number of events after filtering and subsampling.
  num_events = feature_msg_.events.size();
  CHECK_GE(num_events, 1);

  // Check there are enough events for our window size. This is only relevant
  // during initialization.
  if (num_events <= FLAGS_hough_window_size) {
    return;
  }

  // Computing all the radii for each theta hypothesis. This parameter pair
  // forms the Hough space. This is done all at once for each event.
  Eigen::MatrixXi radii;
  radii.resize(kHoughAngularResolution, num_events);
  radii = (polar_param_mapping_ * points).cast<int>();

  // Initializing total Hough spaces. Total means the Hough Space for a full
  // current window, rather than the Hough Space of an individual event. It is
  // therefore the sum of all the Hough Spaces of the events in the current
  // window.
  MatrixHough total_hough_spaces_pos;
  MatrixHough total_hough_spaces_neg;

  // At this point we are starting the parallelisation scheme of this
  // pipeline. As events have to be processed sequentially, the sequence is
  // split into parts. The beginning of each part depends on the end of the
  // previous one. This beginning state is computed using a full HT and full
  // NMS.

  // Resetting the accumulator cells of all Hough spaces
  total_hough_spaces_pos.setZero();
  total_hough_spaces_neg.setZero();

  // Computing total Hough space every at the beginning. This depends on the
  // last FLAGS_hough_window_size of the previous batch.
  computeFullHoughTransform(
      total_hough_spaces_pos, total_hough_spaces_neg, radii);

  // Each event is treated as a timestep. For each of these timesteps we keep
  // the active set of maxima in the Hough Space. These are basically the line
  // detections at each timestep. This whole storage is pre-initialized to
  // make it ready for parallelizing the whole process.
  std::vector<std::vector<hough2map::Detector::line>> maxima_list(num_events);

  // As we compute a full HT at the beginning of each batch, we also need a
  // full NMS. This is computed here.
  computeFullNMS(total_hough_spaces_pos, total_hough_spaces_neg, maxima_list);

  // Within each parallelised NMS batch, we can now perform the rest of the
  // computations iterativels, processing the events in their correct
  // sequence. This is done in parallel for all batches.
  itterativeNMS(
      total_hough_spaces_pos, total_hough_spaces_neg, maxima_list, radii);

  // If visualizations are turned on display them in the video stream.
  if (FLAGS_show_lines_in_video) {
    visualizeCurrentLineDetections(
        points, maxima_list, total_hough_spaces_pos, total_hough_spaces_neg);
  }

  // Publish events that were part of the Hough transform (because they were
  // not filtered out).
  feature_pub_.publish(feature_msg_);

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
}

// Visualizing the current line detections of the first Hough Transform.
// This function only visualizes vertical lines, for other orientations it needs
// to be adjusted.
void Detector::visualizeCurrentLineDetections(
    const Eigen::MatrixXf& points, 
    const std::vector<std::vector<hough2map::Detector::line>>& cur_maxima_list,
    const MatrixHough& hough_pos, const MatrixHough& hough_neg) const {
  int num_events = feature_msg_.events.size();

  // Getting the horizontal positions of all vertical line detections.
  size_t step_size = FLAGS_show_lines_every_nth;
  for (size_t i = FLAGS_hough_window_size - 1; i < num_events; i += step_size) {
    cv::Mat vis_frame;
    cv::remap(
        cur_greyscale_img_, vis_frame, image_undist_map_x_, image_undist_map_y_,
        cv::INTER_LINEAR);

    for (size_t j = i - FLAGS_hough_window_size + 1; j <= i; j++) {
      const dvs_msgs::Event& event = feature_msg_.events[j];
      uint32_t x = static_cast<uint32_t>(points(0, j));
      uint32_t y = static_cast<uint32_t>(points(1, j));
      vis_frame.at<cv::Vec3b>(y, x) =
          event.polarity ? cv::Vec3b(0, 0, 255) : cv::Vec3b(255, 0, 0);
    }

    for (auto& maxima : cur_maxima_list[i]) {
      if (maxima.polarity) {
        drawPolarCorLine(
          vis_frame, maxima.r, maxima.theta, cv::Scalar(0, 0, 255)); 
      } else {
        drawPolarCorLine(
          vis_frame, maxima.r, maxima.theta, cv::Scalar(255, 0, 0)); 
      }
    }

    cv::imshow("Detected poles", vis_frame);
    cv::waitKey(1);
  }
}

// Performing itterative Non-Maximum suppression on the current batch of
// events based on a beginning Hough Space.
void Detector::itterativeNMS(
    MatrixHough& total_hough_space_pos, MatrixHough& total_hough_space_neg,
    std::vector<std::vector<hough2map::Detector::line>>& cur_maxima_list,
    const Eigen::MatrixXi& radii) {
  std::vector<hough2map::Detector::line> new_maxima;
  std::vector<int> new_maxima_value;
  int num_events = feature_msg_.events.size();

  /* Iterative Hough transform */

  // Itterating over all events which are part of this current batch. These
  // will be added and removed through the iteration process.
  const int left = FLAGS_hough_window_size;
  const int right = num_events;
  CHECK_GE(right, left);

  for (int l = left; l < right; l++) {
    // Getting the event that gets added right now.
    const dvs_msgs::Event& event = feature_msg_.events[l];
    const double kTimestamp = event.ts.toSec();
    CHECK_GE(l - 1, 0);

    // Establishing the lists of maxima for this timestep and the ones of the
    // previous timestep
    std::vector<hough2map::Detector::line>& current_maxima = cur_maxima_list[l];
    std::vector<hough2map::Detector::line>& previous_maxima =
        cur_maxima_list[l - 1];

    // Incrementing the accumulator cells for the current event.
    updateHoughSpaceVotes(
        true, l, event.polarity, radii, total_hough_space_pos,
        total_hough_space_neg);

    // Find the oldest event in the current window and get ready to remove it.
    const int kLRemove = l - FLAGS_hough_window_size;
    CHECK_GE(l - FLAGS_hough_window_size, 0);
    const dvs_msgs::Event& event_remove = feature_msg_.events[kLRemove];

    // Decrement the accumulator cells for the event to be removed.
    updateHoughSpaceVotes(
        false, kLRemove, event_remove.polarity, radii, total_hough_space_pos,
        total_hough_space_neg);

    /* Iterative non-maxima suppression */

    // The Hough Spaces have been update. Now the iterative NMS has to run and
    // update the list of known maxima. First reset the temporary lists.
    new_maxima.clear();
    new_maxima_value.clear();

    // Remember which past maxima are no longer maxima so that
    // we don't need to check again at the end.
    std::vector<int> discard(previous_maxima.size(), 0);

    /* Phase 1 - Obtain candidates for global maxima */

    // For points that got incremented.
    for (int i = 0; i < kHoughAngularResolution; ++i) {
      const int kRadius = radii(i, l);

      if ((kRadius >= 0) && (kRadius < kHoughRadiusResolution)) {
        if (event.polarity) {
          bool skip_center = false;
          updateIncrementedNMS(
              kTimestamp, event.polarity, i, kRadius, total_hough_space_pos,
              previous_maxima, discard, new_maxima, new_maxima_value);
          // The center and a neighbour have the same value so
          // no point in checking if it is a local maximum.
          if (skip_center) {
            continue;
          }

        } else {
          bool skip_center = false;
          updateIncrementedNMS(
              kTimestamp, event.polarity, i, kRadius, total_hough_space_neg,
              previous_maxima, discard, new_maxima, new_maxima_value);
          // The center and a neighbour have the same value so
          // no point in checking if it is a local maximum.
          if (skip_center) {
            continue;
          }
        }
      }
    }

    // For accumulator cells that got decremented.
    for (int i = 0; i < kHoughAngularResolution; i++) {
      const int kRadius = radii(i, kLRemove);

      if ((kRadius >= 0) && (kRadius < kHoughRadiusResolution)) {
        if (event_remove.polarity) {
          updateDecrementedNMS(
              kTimestamp, event_remove.polarity, i, kRadius,
              total_hough_space_pos, previous_maxima, discard, new_maxima,
              new_maxima_value);

        } else {
          updateDecrementedNMS(
              kTimestamp, event_remove.polarity, i, kRadius,
              total_hough_space_neg, previous_maxima, discard, new_maxima,
              new_maxima_value);
        }
      }
    }

    if (new_maxima.empty()) {
      // No new maxima in the temporary storage, so we only get rid of the
      // expired ones and keep the rest as it was unchanged.
      for (int i = 0; i < previous_maxima.size(); i++) {
        if (!discard[i]) {
          current_maxima.push_back(previous_maxima[i]);
        }
      }
    } else {
      // We discard all expired old maxima and add the ones that are still
      // valid to the temporary list of new maxima.
      for (int i = 0; i < previous_maxima.size(); i++) {
        if (!discard[i]) {
          const hough2map::Detector::line& kMaximum = previous_maxima[i];

          new_maxima.push_back(kMaximum);

          if (kMaximum.polarity) {
            new_maxima_value.push_back(
                total_hough_space_pos(kMaximum.r, kMaximum.theta_idx));
          } else {
            new_maxima_value.push_back(
                total_hough_space_neg(kMaximum.r, kMaximum.theta_idx));
          }
        }
      }

      /* Phase 2 - Detect and apply suppression */

      // Apply the suppression radius.
      applySuppressionRadius(new_maxima, new_maxima_value, &current_maxima);

      // We need to check if any of the previous maxima that we
      // didn't touch have changed. Then we need to check for new
      // maxima in range and resuppress.
      while (true) {
        const int kNumMaximaCandidates = new_maxima.size();
        for (int i = 0; i < previous_maxima.size(); i++) {
          if (!discard[i]) {
            const hough2map::Detector::line& kPreviousMaximum =
                previous_maxima[i];

            bool found = false;
            for (const hough2map::Detector::line& current_maximum :
                 current_maxima) {
              if ((current_maximum.polarity == kPreviousMaximum.polarity) &&
                  (current_maximum.r == kPreviousMaximum.r) &&
                  (current_maximum.theta_idx == kPreviousMaximum.theta_idx)) {
                found = true;
                break;
              }
            }

            if (!found) {
              discard[i] = true;

              if (kPreviousMaximum.polarity) {
                addMaximaInRadius(
                    kPreviousMaximum.theta_idx, kPreviousMaximum.r,
                    total_hough_space_pos, FLAGS_hough_threshold, true,
                    kTimestamp, &new_maxima, &new_maxima_value, true);
              } else {
                addMaximaInRadius(
                    kPreviousMaximum.theta_idx, kPreviousMaximum.r,
                    total_hough_space_neg, FLAGS_hough_threshold, false,
                    kTimestamp, &new_maxima, &new_maxima_value, true);
              }
            }
          }
        }

        if (kNumMaximaCandidates < new_maxima.size()) {
          applySuppressionRadius(new_maxima, new_maxima_value, &current_maxima);
        } else {
          break;
        }
      }
    }
  }
}

// Updating the iterative Non-Maximum suppression for decremented events.
void Detector::updateDecrementedNMS(
    const double kTimestamp, const bool polarity, const int kAngle,
    const int kRadius, const MatrixHough& hough_space,
    const std::vector<hough2map::Detector::line>& previous_maxima,
    std::vector<int>& discard,
    std::vector<hough2map::Detector::line>& new_maxima,
    std::vector<int>& new_maxima_value) {
  // If decremented accumulator cell was previously a maximum, remove it. If
  // it's still a maximum, we will deal with it later.
  int k = 0;
  bool skip_neighborhood = false;
  for (const hough2map::Detector::line& maximum : previous_maxima) {
    if ((maximum.polarity == polarity) && (kRadius == maximum.r) &&
        (kAngle == maximum.theta_idx)) {
      // Mark as discarded since we will added already
      // in the next step if it still is above the threshold.
      discard[k] = true;

      // Re-add to list of possible maxima for later pruning.
      addMaximaInRadius(
          kAngle, kRadius, hough_space, FLAGS_hough_threshold, polarity,
          kTimestamp, &new_maxima, &new_maxima_value);

      // The neighborhood of this accumulator cell has been checked as part of
      // addMaximaInRadius, so no need to do it again.
      skip_neighborhood = true;
      break;
    }

    k++;
  }

  if (!skip_neighborhood) {
    // Iterate over neighbourhood to check if we might have
    // created a new local maxima by decreasing.
    const int m_l = std::max(kAngle - 1, 0);
    const int m_r = std::min(kAngle + 1, kHoughAngularResolution - 1);
    const int n_l = std::max(kRadius - 1, 0);
    const int n_r = std::min(kRadius + 1, kHoughRadiusResolution - 1);
    for (int m = m_l; m <= m_r; m++) {
      for (int n = n_l; n <= n_r; n++) {
        // The center is a separate case.
        if ((m == kAngle) && (n == kRadius)) {
          continue;
        }

        // Any neighbor points now larger and a maximum?
        if ((hough_space(kRadius, kAngle) + 1 == hough_space(n, m)) &&
            (hough_space(n, m) > FLAGS_hough_threshold) &&
            isLocalMaxima(hough_space, m, n)) {
          // Add to temporary storage.
          new_maxima.push_back(addMaxima(m, n, kTimestamp, polarity));
          new_maxima_value.push_back(hough_space(n, m));
        }
      }
    }
  }
}

// Updating the iterative Non-Maximum suppression for incrementing events.
bool Detector::updateIncrementedNMS(
    const double kTimestamp, const bool polarity, const int kAngle,
    const int kRadius, const MatrixHough& hough_space,
    const std::vector<hough2map::Detector::line>& previous_maxima,
    std::vector<int>& discard,
    std::vector<hough2map::Detector::line>& new_maxima,
    std::vector<int>& new_maxima_value) {
  // If any of the surrounding ones are equal the center
  // for sure is not a local maximum.
  bool skip_center = false;

  // Iterate over neighbourhood to check if we might have
  // supressed a surrounding maximum by growing.
  const int m_l = std::max(kAngle - 1, 0);
  const int m_r = std::min(kAngle + 1, kHoughAngularResolution - 1);
  const int n_l = std::max(kRadius - 1, 0);
  const int n_r = std::min(kRadius + 1, kHoughRadiusResolution - 1);
  for (int m = m_l; m <= m_r; m++) {
    for (int n = n_l; n <= n_r; n++) {
      // The center is a separate case.
      if ((m == kAngle) && (n == kRadius)) {
        continue;
      }

      // Compare point to its neighbors.
      if (hough_space(kRadius, kAngle) == hough_space(n, m)) {
        skip_center = true;
        // Compare to all known maxima from the previous timestep.
        int k = 0;
        for (const hough2map::Detector::line& maximum : previous_maxima) {
          if ((maximum.polarity == polarity) && (n == maximum.r) &&
              (m == maximum.theta_idx)) {
            // We need to discard an old maximum.
            discard[k] = true;

            // And add a new one.
            addMaximaInRadius(
                m, n, hough_space, FLAGS_hough_threshold, polarity, kTimestamp,
                &new_maxima, &new_maxima_value);
            break;
          }

          k++;
        }
      }
    }
  }

  // The center and a neighbour have the same value so
  // no point in checking if it is a local maximum.
  if (skip_center) {
    return true;
  }

  // This is the case for the center point. First checking if it's currently a
  // maximum.
  if ((hough_space(kRadius, kAngle) > FLAGS_hough_threshold) &&
      isLocalMaxima(hough_space, kAngle, kRadius)) {
    bool add_maximum = true;
    // Check if it was a maximum previously.
    for (const auto& maximum : previous_maxima) {
      if ((maximum.polarity == polarity) && (kRadius == maximum.r) &&
          (kAngle == maximum.theta_idx)) {
        add_maximum = false;
        break;
      }
    }

    // If required, add it to the list.
    if (add_maximum) {
      new_maxima.push_back(addMaxima(kAngle, kRadius, kTimestamp, polarity));
      new_maxima_value.push_back(hough_space(kRadius, kAngle));
    }
  }
  return false;
}

// Computing a full Non-Maximum Suppression for a given current Hough Space.
void Detector::computeFullNMS(
    const MatrixHough& total_hough_space_pos,
    const MatrixHough& total_hough_space_neg,
    std::vector<std::vector<hough2map::Detector::line>>& cur_maxima_list) {
  // Index of the current event in the frame of all events of the current
  // message with carry-over from previous message.
  const int kNmsIndex = FLAGS_hough_window_size - 1;
  std::vector<hough2map::Detector::line>& current_maxima =
      cur_maxima_list[kNmsIndex];

  // New detected maxima and their value.
  std::vector<hough2map::Detector::line> new_maxima;
  std::vector<int> new_maxima_value;

  // Checking every angle and radius hypothesis.
  for (int i = 0; i < kHoughAngularResolution; i++) {
    for (int j = 0; j < kHoughRadiusResolution; j++) {
      // Get the current events for a current time stamp.
      const dvs_msgs::Event& event = feature_msg_.events[kNmsIndex];

      // Checking positive Hough space, whether it is larger than threshold
      // and larger than neighbors.
      if (total_hough_space_pos(j, i) > FLAGS_hough_threshold) {
        if (isLocalMaxima(total_hough_space_pos, i, j)) {
          // Add as a possible maximum to the list.
          new_maxima.push_back(addMaxima(i, j, event.ts.toSec(), true));
          new_maxima_value.push_back(total_hough_space_pos(j, i));
        }
      }

      // Checking positive Hough space, whether it is larger than threshold
      // and larger than neighbors.
      if (total_hough_space_neg(j, i) > FLAGS_hough_threshold) {
        if (isLocalMaxima(total_hough_space_neg, i, j)) {
          // Add as a possible maximum to the list.
          new_maxima.push_back(addMaxima(i, j, event.ts.toSec(), false));
          new_maxima_value.push_back(total_hough_space_neg(j, i));
        }
      }
    }
  }
  // This applies a suppression radius to the list of maxima hypothesis, to
  // only get the most prominent ones.
  applySuppressionRadius(new_maxima, new_maxima_value, &current_maxima);
}

// Computing a full Hough Space based on a current set of events and their
// respective Hough parameters.
void Detector::computeFullHoughTransform(
    MatrixHough& total_hough_space_pos, MatrixHough& total_hough_space_neg,
    const Eigen::MatrixXi& radii) {
  // Looping over all events that have an influence on the current total
  // Hough space (i.e. fall in the detection window).
  for (int j = 0; j < FLAGS_hough_window_size; j++) {
    updateHoughSpaceVotes(
        true, j, feature_msg_.events[j].polarity, radii, total_hough_space_pos,
        total_hough_space_neg);
  }
}

// Incrementing a HoughSpace for a certain event.
void Detector::updateHoughSpaceVotes(
    const bool increment, const int event_idx, const bool pol,
    const Eigen::MatrixXi& radii, MatrixHough& hough_space_pos,
    MatrixHough& hough_space_neg) {
  // Looping over all confirmed hypothesis and adding or removing them from the
  // Hough space.
  for (int k = 0; k < kHoughAngularResolution; k++) {
    const int kRadius = radii(k, event_idx);
    // making sure the parameter set is within the domain of the HS.
    if ((kRadius >= 0) && (kRadius < kHoughRadiusResolution)) {
      // Incrementing or decrement the respective accumulator cells.
      if (pol) {
        if (increment) {
          hough_space_pos(kRadius, k)++;
        } else {
          hough_space_pos(kRadius, k)--;
        }
      } else {
        if (increment) {
          hough_space_neg(kRadius, k)++;
        } else {
          hough_space_neg(kRadius, k)--;
        }
      }
    }
  }
}

// Event preprocessing prior to first HT.
void Detector::eventPreProcessing(
    const dvs_msgs::EventArray::ConstPtr& orig_msg, Eigen::MatrixXf& points) {
  int num_events = orig_msg->events.size();
  // Filtering for dead pixels and subsampling the leftover events.
  //
  // TODO: Could be parallelized by splitting into two steps, one that counts
  // second one that does the actual shuffle in memory, if done correctly
  // with some caching overhead would be very small.
  //
  // Also would be faster to just preallocate the eigen matrix to max
  // size and write directly into it and afterwards resize.
  for (int i = 0; i < num_events; i += FLAGS_event_subsample_factor) {
    const dvs_msgs::Event& e = orig_msg->events[i];

    // Seemingly broken pixels in the DVS (millions of exactly equal events at
    // once at random). This needs to be adapted if you use another device.
    if (((e.x != 19) || (e.y != 18)) && ((e.x != 43) || (e.y != 72)) &&
        ((e.x != 89) || (e.y != 52)) && ((e.x != 25) || (e.y != 42)) &&
        ((e.x != 61) || (e.y != 71)) && ((e.x != 37) || (e.y != 112))) {
      feature_msg_.events.push_back(e);
    }
  }

  // Number of events after filtering and subsampling.
  num_events = feature_msg_.events.size();

  // Check there are enough events for our window size. This is only relevant
  // during initialization.
  if (num_events <= FLAGS_hough_window_size) {
    return;
  }

  // Reshaping the event array into an Eigen matrix.
  points.resize(2, num_events);
  points.setZero();

  // Add points from the actual message.
  const auto ptr = points.data();
  CHECK_NOTNULL(ptr);

  if (FLAGS_perform_camera_undistortion) {
    for (int i = 0; i < num_events; i++) {
      const dvs_msgs::Event& event = feature_msg_.events[i];

      *(ptr + 2 * i) = event_undist_map_x_(event.y, event.x);
      *(ptr + 2 * i + 1) = event_undist_map_y_(event.y, event.x);
    }
  } else {
    for (int i = 0; i < num_events; i++) {
      const dvs_msgs::Event& event = feature_msg_.events[i];

      *(ptr + 2 * i) = event.x;
      *(ptr + 2 * i + 1) = event.y;
    }
  }
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
    int i, int radius, const MatrixHough& total_hough_space,
    int local_threshold, bool polarity, double timestamp,
    std::vector<hough2map::Detector::line>* new_maxima,
    std::vector<int>* new_maxima_value, bool skip_center) {
  int m_l = std::max(i - FLAGS_hough_nms_radius, 0);
  int m_r = std::min(i + FLAGS_hough_nms_radius + 1, kHoughAngularResolution);
  int n_l = std::max(radius - FLAGS_hough_nms_radius, 0);
  int n_r =
      std::min(radius + FLAGS_hough_nms_radius + 1, kHoughRadiusResolution);

  for (int m = m_l; m < m_r; m++) {
    for (int n = n_l; n < n_r; n++) {
      if (skip_center && (n == radius) && (m == i)) {
        continue;
      }

      if ((total_hough_space(n, m) > local_threshold) &&
          isLocalMaxima(total_hough_space, m, n)) {
        new_maxima->push_back(addMaxima(m, n, timestamp, polarity));
        new_maxima_value->push_back(total_hough_space(n, m));
      }
    }
  }
}

void Detector::applySuppressionRadius(
    const std::vector<hough2map::Detector::line>& new_maxima,
    const std::vector<int>& new_maxima_value,
    std::vector<hough2map::Detector::line>* current_maxima) {
  // Create an index of all known maxima.
  std::vector<int> new_maxima_index(new_maxima_value.size());
  for (int i = 0; i < new_maxima_value.size(); i++) {
    new_maxima_index[i] = i;
  }

  // Sort the index of all currently known maxima. Sort them by: 1. value; 2.
  // rho value; 3. theta value.
  std::stable_sort(
      new_maxima_index.begin(), new_maxima_index.end(),
      [&new_maxima_value, &new_maxima](const int i1, const int i2) {
        if (new_maxima_value[i1] != new_maxima_value[i2]) {
          return new_maxima_value[i1] > new_maxima_value[i2];
        } else {
          if (new_maxima[i1].r != new_maxima[i2].r) {
            return new_maxima[i1].r > new_maxima[i2].r;
          } else {
            return new_maxima[i1].theta_idx > new_maxima[i2].theta_idx;
          }
        }
      });

  // Clear buffer of current maxima to re-add them later on.
  current_maxima->clear();

  // Loop over all maxima according to the sorted order.
  for (int i = 0; i < new_maxima.size(); i++) {
    const hough2map::Detector::line& new_maximum =
        new_maxima[new_maxima_index[i]];

    bool add_maximum = true;
    // Compare to all other maxima in the output buffer.
    for (int j = 0; j < current_maxima->size(); j++) {
      const hough2map::Detector::line& cur_maximum = (*current_maxima)[j];

      // If no maximum in the output buffer is of the same polarity and within
      // the radius, the current maximum is kept and added to the output
      // buffer.
      if (cur_maximum.polarity == new_maximum.polarity) {
        // Suppression radius.
        float distance =
            (cur_maximum.r - new_maximum.r) * (cur_maximum.r - new_maximum.r) +
            (cur_maximum.theta_idx - new_maximum.theta_idx) *
                (cur_maximum.theta_idx - new_maximum.theta_idx);

        if (distance < FLAGS_hough_nms_radius * FLAGS_hough_nms_radius) {
          add_maximum = false;
          break;
        }
      }
    }

    // Adding accepted maxima to the output buffer.
    if (add_maximum) {
      current_maxima->push_back(new_maximum);
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

  cur_greyscale_img_ = cv_ptr->image;
  cv::cvtColor(cur_greyscale_img_, cur_greyscale_img_, cv::COLOR_GRAY2BGR);
}

// Just a funciton for creating new line structs.
Detector::line Detector::addMaxima(
    int angle, int rad, double cur_time, bool pol) {
  hough2map::Detector::line new_line;

  new_line.ID = 0;
  new_line.r = rad;
  new_line.theta = thetas_(angle);
  new_line.theta_idx = angle;
  new_line.time = cur_time;
  new_line.polarity = pol;

  return new_line;
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
