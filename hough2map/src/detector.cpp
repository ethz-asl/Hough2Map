#include <chrono>
#include <omp.h>
#include <ros/package.h>
#include <thread>

#include "hough2map/detector.h"

DEFINE_int32(hough_1_threshold, 15,
             "Threshold for the first level Hough transform.");
DEFINE_int32(hough_1_window_size, 300,
             "Max queue length for the first Hough transform.");
DEFINE_int32(
    hough_space_NMS_suppression_radius, 10,
    "Non-Maximum-Suppression suppression radius to enforce Maxima separation");
DEFINE_int32(hough_2_min_threshold, 7000,
             "Minimal Threshold for the second Hough transform");
DEFINE_int32(event_subsample_factor, 1,
             "Subsample Events by a constant factor");
DEFINE_bool(show_lines_in_video, false,
            "Plot detected lines in the video stream");
DEFINE_bool(lines_output, false, "Output detected lines to a file");
DEFINE_bool(map_output, false, "Export detected poles to file");
DEFINE_bool(display_2nd_hough_space, false,
            "Display the current 2nd degree hough Space");
DEFINE_bool(odometry_available, true, "A GPS Odometry is available");
DEFINE_double(odometry_event_alignment, 0,
              "Manual time synchronization to compensate misalignment between "
              "odometry and DVS timestamps");
DEFINE_double(camera_offset_x, 0, "Camera offset along the train axis");
DEFINE_double(camera_offset_y, 0,
              "Camera offset perpendicular to the left of the train axis");
DEFINE_bool(perform_camera_undistortion, true,
            "Undistort event data according to camera calibration");
DEFINE_double(buffer_size_s, 30, "Size of the odometry buffer in seconds");
DEFINE_double(event_array_frequency, 30.0,
              "Expected frequency of event arrays.");
DEFINE_double(hough_2_nms_min_angle_separation, 20.0,
              "The suppression radius in the Non-maximum-suppression for the "
              "second Hough Transform for angles..");
DEFINE_double(hough_2_nms_min_rho_separation, 50.0,
              "The suppression radius in the Non-maximum-suppression for the "
              "second Hough Transform for pixel separation.");
DEFINE_double(
    hough_2_nms_neg_pos_angular_matching, 0.05,
    "Angular separation between a positive and a negative pole detection for "
    "them to be confirmed as a pole detection. Value is in radians sqared.");
DEFINE_double(
    hough_2_nms_neg_pos_radial_matching, 2500,
    "Radial separation between a positive and a negative pole detection for "
    "them to be confirmed as a pole detection. Value is in pixels sqared.");

namespace hough2map {
Detector::Detector(const ros::NodeHandle &nh, const ros::NodeHandle &nh_private)
    : nh_(nh), nh_private_(nh_private) {
  // Checking that flags have reasonable values.
  CHECK_GT(FLAGS_event_array_frequency, 0);
  CHECK_GE(FLAGS_hough_1_window_size, 1);
  CHECK_GT(FLAGS_event_subsample_factor, 0);
  CHECK_GE(FLAGS_buffer_size_s, 1);
  CHECK_GT(FLAGS_hough_1_threshold, 0);
  CHECK_GT(FLAGS_hough_2_min_threshold, 0);

  // File output for the line parameters of the lines in the event stream.
  if (FLAGS_lines_output) {
    lines_file.open(lines_file_path);

    if (lines_file.is_open()) {
      lines_file << "time,param,pol\n";
    } else {
      LOG(FATAL) << "Could not open file:" << lines_file_path << std::endl;
    }
  }

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

  omp_set_num_threads(kNumThreads);

  // Various subscribers and publishers for event and odometry data.
  feature_pub_ = nh_.advertise<dvs_msgs::EventArray>("/feature_events", 1);
  event_sub_ = nh_.subscribe("/dvs/events", 0, &Detector::eventCallback, this);
  GPS_pos_ =
      nh_.subscribe("/oxts/position", 0, &Detector::positionCallback, this);
  GPS_vel_ =
      nh_.subscribe("/oxts/velocity", 0, &Detector::velocityCallback, this);
  GPS_orient_ = nh_.subscribe("/oxts/orientation", 0,
                              &Detector::orientationCallback, this);

  // Plot current hough detections in the video.
  if (FLAGS_show_lines_in_video) {
    cv::namedWindow("Detected poles", CV_WINDOW_NORMAL);
    image_raw_sub_ =
        nh_.subscribe("/dvs/image_raw", 0, &Detector::imageCallback, this);
  }

  // Initializig theta, sin and cos values for first and second Hough
  // transform.
  initializeSinCosMap(thetas_1_, polar_param_mapping_1_, kHough1MinAngle,
                      kHough1MaxAngle, kHough1AngularResolution);
  initializeSinCosMap(thetas_2_, polar_param_mapping_2_, kHough2MinAngle,
                      kHough2MaxAngle, kHough2AngularResolution);

  // Initializing various transformation matrizes.
  initializeTransformationMatrices();

  if (FLAGS_display_2nd_hough_space) {
    cv::namedWindow("Hough Transform #2", CV_WINDOW_NORMAL);
  }
}

Detector::~Detector() {
  // Close all open files.
  if (FLAGS_lines_output) {
    lines_file.close();
  }

  if (FLAGS_map_output) {
    map_file.close();
  }
}

const int Detector::kHough1AngularResolution;
const int Detector::kHough1RadiusResolution;

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
void Detector::initializeSinCosMap(Eigen::EigenBase<DerivedVec> &angles,
                                   Eigen::EigenBase<DerivedMat> &sin_cos_map,
                                   const int kMinAngle, const int kMaxAngle,
                                   const int kNumSteps) {

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
  cv::Mat camera_matrix = (cv::Mat1d(3, 3) << intrinsics_[0], 0, intrinsics_[2],
                           0, intrinsics_[1], intrinsics_[3], 0, 0, 1);
  cv::Mat distortionCoefficients =
      (cv::Mat1d(1, 4) << distortion_coeffs_[0], distortion_coeffs_[1],
       distortion_coeffs_[2], distortion_coeffs_[3]);

  undist_map_x_.resize(camera_resolution_height_, camera_resolution_width_);
  undist_map_y_.resize(camera_resolution_height_, camera_resolution_width_);

  // Compute undistortion mapping.
  for (int i = 0; i < camera_resolution_width_; i++) {
    for (int j = 0; j < camera_resolution_height_; j++) {
      cv::Mat_<cv::Point2f> points(1, 1);
      points(0) = cv::Point2f(i, j);
      cv::Mat dst;

      cv::undistortPoints(points, dst, camera_matrix, distortionCoefficients);
      const float u = intrinsics_[0] * dst.at<float>(0, 0) + intrinsics_[2];
      const float v = intrinsics_[1] * dst.at<float>(0, 1) + intrinsics_[3];

      CHECK_GT(u, 0.0 - kAcceptableDistortionRange)
          << "Horizontal undistortion is larger than expected";
      CHECK_LT(u, camera_resolution_width_ + kAcceptableDistortionRange)
          << "Horizontal undistortion is larger than expected";
      CHECK_GT(v, 0.0 - kAcceptableDistortionRange)
          << "Vertical undistortion is larger than expected";
      CHECK_LT(v, camera_resolution_height_ + kAcceptableDistortionRange)
          << "Vertical undistortion is larger than expected";

      undist_map_x_(j, i) = u;
      undist_map_y_(j, i) = v;
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

void Detector::eventCallback(const dvs_msgs::EventArray::ConstPtr &msg) {
  const auto kStartTime = std::chrono::high_resolution_clock::now();

  feature_msg_.header = msg->header;
  feature_msg_.width = msg->width;
  feature_msg_.height = msg->height;

  int num_events = msg->events.size();
  CHECK_GE(num_events, 1);

  // If initialized then make sure the last FLAGS_hough_1_window_size events
  // are prepended to the current list of events and remove older events.
  if (feature_msg_.events.size() > FLAGS_hough_1_window_size) {
    std::copy(feature_msg_.events.end() - FLAGS_hough_1_window_size,
              feature_msg_.events.end(), feature_msg_.events.begin());
    feature_msg_.events.resize(FLAGS_hough_1_window_size);
  }

  // Reshaping the event array into an Eigen matrix.
  Eigen::MatrixXf points;
  eventPreProcessing(msg, points);

  // Number of events after filtering and subsampling.
  num_events = feature_msg_.events.size();
  CHECK_GE(num_events, 1);

  // Check there are enough events for our window size. This is only relevant
  // during initialization.
  if (num_events <= FLAGS_hough_1_window_size) {
    return;
  }

  // Computing all the radii for each theta hypothesis. This parameter pair
  // forms the Hough space. This is done all at once for each event.
  Eigen::MatrixXi radii;
  radii.resize(kHough1AngularResolution, num_events);
  radii = (polar_param_mapping_1_ * points).cast<int>();

  // Total Hough Space at NMS Intervals
  // kNmsBatchCount is the reduced number of iterations. This is basically
  // the number of sub-batches that will be processed in parallel
  CHECK_GE(kNumThreads, 1);
  int nms_recompute_window =
      std::ceil(float(num_events - FLAGS_hough_1_window_size) / kNumThreads);
  nms_recompute_window =
      std::max(nms_recompute_window, FLAGS_hough_1_window_size);
  CHECK_GT(nms_recompute_window, 0);
  const int kNmsBatchCount = std::ceil(
      float(num_events - FLAGS_hough_1_window_size) / nms_recompute_window);

  // Initializing total Hough spaces. Total means the Hough Space for a full
  // current window, rather than the Hough Space of an individual event. It is
  // therefore the sum of all the Hough Spaces of the events in the current
  // window.
  std::vector<MatrixHough> total_hough_spaces_pos(kNmsBatchCount);
  std::vector<MatrixHough> total_hough_spaces_neg(kNmsBatchCount);

  // At this point we are starting the parallelisation scheme of this
  // pipeline. As events have to be processed sequentially, the sequence is
  // split into parts. The beginning of each part depends on the end of the
  // previous one. This beginning state is computed using a full HT and full
  // NMS.

  // Resetting the accumulator cells of all Hough spaces for the beginning of
  // all batches.
#pragma omp parallel for
  for (int i = 0; i < kNmsBatchCount; i++) {
    total_hough_spaces_pos[i].setZero();
    total_hough_spaces_neg[i].setZero();
  }

  // Computing total Hough space every N steps, so for the beginning of each
  // parallelisation batch. This depends on the last FLAGS_hough_1_window_size
  // of the previous batch.
#pragma omp parallel for
  for (int i = 0; i < kNmsBatchCount; i++) {
    computeFullHoughTransform(i, nms_recompute_window,
                              total_hough_spaces_pos[i],
                              total_hough_spaces_neg[i], radii);
  }

  // Each event is treated as a timestep. For each of these timesteps we keep
  // the active set of maxima in the Hough Space. These are basically the line
  // detections at each timestep. This whole storage is pre-initialized to
  // make it ready for parallelizing the whole process.
  std::vector<std::vector<hough2map::Detector::line>> maxima_list(num_events);

  // As we compute a full HT at the beginning of each batch, we also need a
  // full NMS. This is computed here.
#pragma omp parallel for
  for (int k = 0; k < kNmsBatchCount; k++) {
    computeFullNMS(k, nms_recompute_window, total_hough_spaces_pos[k],
                   total_hough_spaces_neg[k], maxima_list);
  }

  // Within each parallelised NMS batch, we can now perform the rest of the
  // computations iterativels, processing the events in their correct
  // sequence. This is done in parallel for all batches.
#pragma omp parallel for
  for (int k = 0; k < kNmsBatchCount; k++) {
    itterativeNMS(k, nms_recompute_window, total_hough_spaces_pos[k],
                  total_hough_spaces_neg[k], maxima_list, radii);
  }
  // If visualizations are turned on display them in the video stream.
  if (FLAGS_show_lines_in_video) {
    visualizeCurrentLineDetections(maxima_list);
  }

  // Run the second Hough Transform for spatio-temporal tracking.
  secondHoughTransform(maxima_list);

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
    const std::vector<std::vector<hough2map::Detector::line>>
        &cur_maxima_list) {
  int num_events = feature_msg_.events.size();

  int positive_detections[camera_resolution_width_] = {0};
  int negative_detections[camera_resolution_width_] = {0};

  // Getting the horizontal positions of all vertical line detections.
  for (int i = 0; i < num_events; i++) {
    const dvs_msgs::Event &e = feature_msg_.events[i];
    for (auto &maxima : cur_maxima_list[i]) {
      if (maxima.polarity) {
        positive_detections[maxima.r] = 1;
      } else {
        negative_detections[maxima.r] = 1;
      }
    }
  }

  cv::Mat cur_frame = cur_greyscale_img_;

  // Plottin current line detections.
  for (int i = 0; i < camera_resolution_width_; i++) {
    if (positive_detections[i] == 1) {
      cv::line(cur_frame, cv::Point(i, 0),
               cv::Point(i, camera_resolution_height_), cv::Scalar(255, 0, 0),
               2, 8);
    }
    if (negative_detections[i] == 1) {
      cv::line(cur_frame, cv::Point(i, 0),
               cv::Point(i, camera_resolution_height_), cv::Scalar(0, 0, 255),
               2, 8);
    }
  }

  cv::imshow("Detected poles", cur_frame);
  cv::waitKey(1);
}

// Performing itterative Non-Maximum suppression on the current batch of
// events based on a beginning Hough Space.
void Detector::itterativeNMS(
    const int time_step, const int nms_recompute_window,
    MatrixHough &total_hough_space_pos, MatrixHough &total_hough_space_neg,
    std::vector<std::vector<hough2map::Detector::line>> &cur_maxima_list,
    const Eigen::MatrixXi &radii) {

  std::vector<hough2map::Detector::line> new_maxima;
  std::vector<int> new_maxima_value;
  int num_events = feature_msg_.events.size();

  /* Iterative Hough transform */

  // Itterating over all events which are part of this current batch. These
  // will be added and removed through the iteration process.
  const int left =
      FLAGS_hough_1_window_size + time_step * nms_recompute_window + 1;
  const int right = std::min(left + nms_recompute_window - 1, num_events);
  CHECK_GE(right, left);

  for (int l = left; l < right; l++) {
    // Getting the event that gets added right now.
    const dvs_msgs::Event &event = feature_msg_.events[l];
    const double kTimestamp = event.ts.toSec();
    CHECK_GE(l - 1, 0);

    // Establishing the lists of maxima for this timestep and the ones of the
    // previous timestep
    std::vector<hough2map::Detector::line> &current_maxima = cur_maxima_list[l];
    std::vector<hough2map::Detector::line> &previous_maxima =
        cur_maxima_list[l - 1];

    // Incrementing the accumulator cells for the current event.
    updateHoughSpaceVotes(true, l, event.polarity, radii, total_hough_space_pos,
                          total_hough_space_neg);

    // Find the oldest event in the current window and get ready to remove it.
    const int kLRemove = l - FLAGS_hough_1_window_size;
    CHECK_GE(l - FLAGS_hough_1_window_size, 0);
    const dvs_msgs::Event &event_remove = feature_msg_.events[kLRemove];

    // Decrement the accumulator cells for the event to be removed.
    updateHoughSpaceVotes(false, kLRemove, event_remove.polarity, radii,
                          total_hough_space_pos, total_hough_space_neg);

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
    for (int i = 0; i < kHough1AngularResolution; ++i) {
      const int kRadius = radii(i, l);

      if ((kRadius >= 0) && (kRadius < kHough1RadiusResolution)) {
        if (event.polarity) {
          bool skip_center = false;
          updateIncrementedNMS(kTimestamp, event.polarity, i, kRadius,
                               total_hough_space_pos, previous_maxima, discard,
                               new_maxima, new_maxima_value);
          // The center and a neighbour have the same value so
          // no point in checking if it is a local maximum.
          if (skip_center) {
            continue;
          }

        } else {
          bool skip_center = false;
          updateIncrementedNMS(kTimestamp, event.polarity, i, kRadius,
                               total_hough_space_neg, previous_maxima, discard,
                               new_maxima, new_maxima_value);
          // The center and a neighbour have the same value so
          // no point in checking if it is a local maximum.
          if (skip_center) {
            continue;
          }
        }
      }
    }

    // For accumulator cells that got decremented.
    for (int i = 0; i < kHough1AngularResolution; i++) {
      const int kRadius = radii(i, kLRemove);

      if ((kRadius >= 0) && (kRadius < kHough1RadiusResolution)) {
        if (event_remove.polarity) {
          updateDecrementedNMS(kTimestamp, event_remove.polarity, i, kRadius,
                               total_hough_space_pos, previous_maxima, discard,
                               new_maxima, new_maxima_value);

        } else {
          updateDecrementedNMS(kTimestamp, event_remove.polarity, i, kRadius,
                               total_hough_space_neg, previous_maxima, discard,
                               new_maxima, new_maxima_value);
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
          const hough2map::Detector::line &kMaximum = previous_maxima[i];

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
            const hough2map::Detector::line &kPreviousMaximum =
                previous_maxima[i];

            bool found = false;
            for (const hough2map::Detector::line &current_maximum :
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
                addMaximaInRadius(kPreviousMaximum.theta_idx,
                                  kPreviousMaximum.r, total_hough_space_pos,
                                  FLAGS_hough_1_threshold, true, kTimestamp,
                                  &new_maxima, &new_maxima_value, true);
              } else {
                addMaximaInRadius(kPreviousMaximum.theta_idx,
                                  kPreviousMaximum.r, total_hough_space_neg,
                                  FLAGS_hough_1_threshold, false, kTimestamp,
                                  &new_maxima, &new_maxima_value, true);
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
    const int kRadius, const MatrixHough &hough_space,
    const std::vector<hough2map::Detector::line> &previous_maxima,
    std::vector<int> &discard,
    std::vector<hough2map::Detector::line> &new_maxima,
    std::vector<int> &new_maxima_value) {

  // If decremented accumulator cell was previously a maximum, remove it. If
  // it's still a maximum, we will deal with it later.
  int k = 0;
  bool skip_neighborhood = false;
  for (const hough2map::Detector::line &maximum : previous_maxima) {
    if ((maximum.polarity == polarity) && (kRadius == maximum.r) &&
        (kAngle == maximum.theta_idx)) {
      // Mark as discarded since we will added already
      // in the next step if it still is above the threshold.
      discard[k] = true;

      // Re-add to list of possible maxima for later pruning.
      addMaximaInRadius(kAngle, kRadius, hough_space, FLAGS_hough_1_threshold,
                        polarity, kTimestamp, &new_maxima, &new_maxima_value);

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
    const int m_r = std::min(kAngle + 1, kHough1AngularResolution - 1);
    const int n_l = std::max(kRadius - 1, 0);
    const int n_r = std::min(kRadius + 1, kHough1RadiusResolution - 1);
    for (int m = m_l; m <= m_r; m++) {
      for (int n = n_l; n <= n_r; n++) {
        // The center is a separate case.
        if ((m == kAngle) && (n == kRadius)) {
          continue;
        }

        // Any neighbor points now larger and a maximum?
        if ((hough_space(kRadius, kAngle) + 1 == hough_space(n, m)) &&
            (hough_space(n, m) > FLAGS_hough_1_threshold) &&
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
    const int kRadius, const MatrixHough &hough_space,
    const std::vector<hough2map::Detector::line> &previous_maxima,
    std::vector<int> &discard,
    std::vector<hough2map::Detector::line> &new_maxima,
    std::vector<int> &new_maxima_value) {

  // If any of the surrounding ones are equal the center
  // for sure is not a local maximum.
  bool skip_center = false;

  // Iterate over neighbourhood to check if we might have
  // supressed a surrounding maximum by growing.
  const int m_l = std::max(kAngle - 1, 0);
  const int m_r = std::min(kAngle + 1, kHough1AngularResolution - 1);
  const int n_l = std::max(kRadius - 1, 0);
  const int n_r = std::min(kRadius + 1, kHough1RadiusResolution - 1);
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
        for (const hough2map::Detector::line &maximum : previous_maxima) {
          if ((maximum.polarity == polarity) && (n == maximum.r) &&
              (m == maximum.theta_idx)) {
            // We need to discard an old maximum.
            discard[k] = true;

            // And add a new one.
            addMaximaInRadius(m, n, hough_space, FLAGS_hough_1_threshold,
                              polarity, kTimestamp, &new_maxima,
                              &new_maxima_value);
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
  if ((hough_space(kRadius, kAngle) > FLAGS_hough_1_threshold) &&
      isLocalMaxima(hough_space, kAngle, kRadius)) {
    bool add_maximum = true;
    // Check if it was a maximum previously.
    for (const auto &maximum : previous_maxima) {
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
    const int time_step, const int nms_recompute_window,
    const MatrixHough &total_hough_space_pos,
    const MatrixHough &total_hough_space_neg,
    std::vector<std::vector<hough2map::Detector::line>> &cur_maxima_list) {

  // Index of the current event in the frame of all events of the current
  // message with carry-over from previous message.
  const int kNmsIndex =
      FLAGS_hough_1_window_size + time_step * nms_recompute_window;
  std::vector<hough2map::Detector::line> &current_maxima =
      cur_maxima_list[kNmsIndex];

  // New detected maxima and their value.
  std::vector<hough2map::Detector::line> new_maxima;
  std::vector<int> new_maxima_value;

  // Checking every angle and radius hypothesis.
  for (int i = 0; i < kHough1AngularResolution; i++) {
    for (int j = 0; j < kHough1RadiusResolution; j++) {
      // Get the current events for a current time stamp.
      const dvs_msgs::Event &event = feature_msg_.events[kNmsIndex];

      // Checking positive Hough space, whether it is larger than threshold
      // and larger than neighbors.
      if (total_hough_space_pos(j, i) > FLAGS_hough_1_threshold) {
        if (isLocalMaxima(total_hough_space_pos, i, j)) {
          // Add as a possible maximum to the list.
          new_maxima.push_back(addMaxima(i, j, event.ts.toSec(), true));
          new_maxima_value.push_back(total_hough_space_pos(j, i));
        }
      }

      // Checking positive Hough space, whether it is larger than threshold
      // and larger than neighbors.
      if (total_hough_space_neg(j, i) > FLAGS_hough_1_threshold) {
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
void Detector::computeFullHoughTransform(const int time_step,
                                         const int nms_recompute_window,
                                         MatrixHough &total_hough_space_pos,
                                         MatrixHough &total_hough_space_neg,
                                         const Eigen::MatrixXi &radii) {
  // Looping over all events that have an influence on the current total
  // Hough space, so the past 300.
  const int kRight =
      FLAGS_hough_1_window_size + time_step * nms_recompute_window;
  const int kLeft = kRight - FLAGS_hough_1_window_size;
  CHECK_GT(kRight, kLeft);

  for (int j = kRight; j > kLeft; j--) {
    // Looping over all confirmed hypothesis and adding them to the Hough
    // space.
    updateHoughSpaceVotes(true, j, feature_msg_.events[j].polarity, radii,
                          total_hough_space_pos, total_hough_space_neg);
  }
}

// Incrementing a HoughSpace for a certain event.
void Detector::updateHoughSpaceVotes(const bool increment, const int event_idx,
                                     const bool pol,
                                     const Eigen::MatrixXi &radii,
                                     MatrixHough &hough_space_pos,
                                     MatrixHough &hough_space_neg) {
  // Looping over all confirmed hypothesis and adding or removing them from the
  // Hough space.
  for (int k = 0; k < kHough1AngularResolution; k++) {
    const int kRadius = radii(k, event_idx);
    // making sure the parameter set is within the domain of the HS.
    if ((kRadius >= 0) && (kRadius < kHough1RadiusResolution)) {
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

// Apply the scond Hough tranform.
void Detector::secondHoughTransform(
    const std::vector<std::vector<hough2map::Detector::line>>
        &cur_maxima_list) {

  int num_events = feature_msg_.events.size();

  // Second Hough tranform. x-t-space
  // Track the obtained maxima over time.

  // Step 1: Reshape maxima list into x-t data.
  // Step 2: Discretizing maxima into timesteps.

  Eigen::Matrix<int, kHough1RadiusResolution, kHough2TimestepsPerMsg>
      tracked_maxima_pos;
  Eigen::Matrix<int, kHough1RadiusResolution, kHough2TimestepsPerMsg>
      tracked_maxima_neg;

  tracked_maxima_pos.setZero();
  tracked_maxima_neg.setZero();

  const double kTimestampMsgBegin = feature_msg_.events[0].ts.toSec();

  CHECK_GT(kHough2TimestepsPerMsg, 0);
  const double kColumnLengthInSec =
      (1.0 / FLAGS_event_array_frequency) / (kHough2TimestepsPerMsg);

  for (int i = 0; i < num_events; i++) {
    const dvs_msgs::Event &e = feature_msg_.events[i];
    for (auto &maxima : cur_maxima_list[i]) {
      // Not every message is exactly 33ms long. To deal with this, we currently
      // squeeze longer messages down to be 33ms long.

      const int time_idx =
          std::min((int)std::floor((e.ts.toSec() - kTimestampMsgBegin) /
                                   kColumnLengthInSec),
                   kHough2TimestepsPerMsg - 1);

      CHECK_LT(time_idx, kHough2TimestepsPerMsg)
          << ": Something wrong with the time index!" + time_idx;

      // This discretizes the maxima. Many will be same or super close, so we
      // accumulate them.
      if (maxima.polarity) {
        tracked_maxima_pos(maxima.r, time_idx)++;
      } else {
        tracked_maxima_neg(maxima.r, time_idx)++;
      }
    }
  }

  // Ok, all the maxima are nicely arranged. Now I can do another Hough
  // transform with them,but first I want to look at them I need to establish
  // a window of points.

  // All maxima are now arranged in the current time window and discretized to
  // reduce the required computation time. We can now apply the next Hough
  // Transform.

  // Size of window depending on number of messages in window, assuming 30
  // messages per second.

  const double kWindowSizeInSec =
      kHough2MsgPerWindow / FLAGS_event_array_frequency;

  const double kWindowEndTime = feature_msg_.events[num_events - 1].ts.toSec();

  // Window for data storage.
  hough2_queue_pos_.push_back(tracked_maxima_pos);
  hough2_queue_neg_.push_back(tracked_maxima_neg);
  CHECK_EQ(hough2_queue_pos_.size(), hough2_queue_neg_.size())
      << ": Something wrong with window size of the 2nd hough transform!";

  // Removing old stuff when the window is full.
  if (hough2_queue_pos_.size() > kHough2MsgPerWindow) {
    hough2_queue_pos_.pop_front();
    hough2_queue_neg_.pop_front();
  }

  // Initializing Hough spaces for the 2nd HT.
  const int test = kHough2AngularResolution;
  Eigen::MatrixXi hough_2_space_pos(
      kHough2MsgPerWindow * kHough2TimestepsPerMsg, test);
  Eigen::MatrixXi hough_2_space_neg(
      kHough2MsgPerWindow * kHough2TimestepsPerMsg, test);

  hough_2_space_pos.setZero();
  hough_2_space_neg.setZero();

  // For every message in the window.

#pragma omp parallel for
  for (int i = 0; i < hough2_queue_pos_.size(); i++) {
    for (int j = 0; j < kHough1RadiusResolution; j++) {
      for (int k = 0; k < kHough2TimestepsPerMsg; k++) {
        Eigen::Matrix<double, 2, 1> point;
        Eigen::Matrix<int, kHough2AngularResolution, 1> rho;

        point(0, 0) = i * kHough2TimestepsPerMsg + k;
        point(1, 0) = j;

        // Accumulated maxima from the discretization step now function as
        // weights for the HT.
        const int kHough2WeightsPos = hough2_queue_pos_[i](j, k);
        const int kHough2WeightsNeg = hough2_queue_neg_[i](j, k);

        rho = (polar_param_mapping_2_ * point).cast<int>();

        // For every angle.
        for (int l = 0; l < kHough2AngularResolution; l++) {
          // Do Hough transform.
          if ((rho(l, 0) > 0) &&
              (rho(l, 0) < kHough2MsgPerWindow * kHough2TimestepsPerMsg)) {
            hough_2_space_pos(rho(l, 0), l) += kHough2WeightsPos;
            hough_2_space_neg(rho(l, 0), l) += kHough2WeightsNeg;
          }
        }
      }
    }
  }

  std::vector<cv::Vec3f> detected_lines_pos;
  std::vector<cv::Vec3f> detected_lines_neg;

  // Tuning parameter for enforcing separation of maxima.
  const float kMinAngleSep = FLAGS_hough_2_nms_min_angle_separation *
                             FLAGS_hough_2_nms_min_angle_separation;
  const float kMinRadiusSep = FLAGS_hough_2_nms_min_rho_separation *
                              FLAGS_hough_2_nms_min_rho_separation;

  for (int i = 0; i < kHough2AngularResolution; i++) {
    for (int j = 0; j < kHough2MsgPerWindow * kHough2TimestepsPerMsg; j++) {
      // Checking positive Hough space.
      hough2nms(i, j, hough_2_space_pos, detected_lines_pos);

      // Checking negative Hough space.
      hough2nms(i, j, hough_2_space_neg, detected_lines_neg);
    }
  }

  // At this point we check whether positive and negative line detections line
  // up. If there is a pole infront of the camera, there will be a positive and
  // a negative detection in parallel an close proximity from the two
  // edges of the pole. If it is another object, such as a building or
  // bridge, these two edges will be separated much further.
  for (size_t i = 0; i < detected_lines_pos.size(); i++) {
    // Parameters of current positive line.
    const float kRhoPos = detected_lines_pos[i][0];
    const float kThetaPos = detected_lines_pos[i][1];
    // Compare against all current negative lines (typically there are only a
    // hand full of lines simultaneously, so not so expensive).
    for (size_t j = 0; j < detected_lines_neg.size(); j++) {
      const float kRhoNeg = detected_lines_neg[j][0];
      const float kThetaNeg = detected_lines_neg[j][1];

      const float kPosNegRadiusSeparation =
          (kRhoPos - kRhoNeg) * (kRhoPos - kRhoNeg);
      const float kPosNegAngularSeparation =
          (kThetaPos - kThetaNeg) * (kThetaPos - kThetaNeg);

      // If the two lines are nearly parallel and relativity close to each other
      // pixel wise, they are approved to be a pole detection.
      if ((kPosNegAngularSeparation <
           FLAGS_hough_2_nms_neg_pos_angular_matching) &&
          (kPosNegRadiusSeparation <
           FLAGS_hough_2_nms_neg_pos_radial_matching)) {
        // Compute pole timestamps. I need the pole speed in px/s and a
        // timestamp for finding the according train speed.

        double timestamp_enter;
        double timestamp_leave;
        const double kWindowTimestampBeginning =
            kWindowEndTime - kWindowSizeInSec;

        timestamp_leave = (1 / cos(kThetaPos)) * (kRhoPos);
        timestamp_enter = (1 / cos(kThetaPos)) *
                          (kRhoPos - kHough2AngularResolution * sin(kThetaPos));

        timestamp_enter =
            (timestamp_enter / (kHough2MsgPerWindow * kHough2TimestepsPerMsg)) *
                kWindowSizeInSec +
            kWindowTimestampBeginning;
        timestamp_leave =
            (timestamp_leave / (kHough2MsgPerWindow * kHough2TimestepsPerMsg)) *
                kWindowSizeInSec +
            kWindowTimestampBeginning;

        CHECK_GT(timestamp_leave, timestamp_enter)
            << ":Timestamps seem to be wrong, the pole leaves before it "
               "enters?!";

        // If an odometry is available, we can triangulate the new pole.
        if (FLAGS_odometry_available) {
          newPoleDetection(kRhoPos, kThetaPos, kWindowTimestampBeginning, true);
        }
      }
    }
  }

  // Plot the lines.
  if (FLAGS_display_2nd_hough_space) {
    visualizeSecondHoughSpace(detected_lines_pos, detected_lines_neg);
  }
}

// Second Hough space non maximum suppression.
void Detector::hough2nms(const int i, const int j,
                         const Eigen::MatrixXi &hough_2_space,
                         std::vector<cv::Vec3f> &detections) {
  // Tuning parameter for enforcing separation of maxima.
  const float kMinAngleSep = FLAGS_hough_2_nms_min_angle_separation *
                             FLAGS_hough_2_nms_min_angle_separation;
  const float kMinRadiusSep = FLAGS_hough_2_nms_min_rho_separation *
                              FLAGS_hough_2_nms_min_rho_separation;

  if (hough_2_space(j, i) > FLAGS_hough_2_min_threshold) {
    if (isLocalMaxima(hough_2_space, i, j)) {
      cv::Vec3f new_vector;
      new_vector[0] = j;
      new_vector[1] = thetas_2_(i);
      new_vector[2] = hough_2_space(j, i);

      // Check if lines already exist.
      if (detections.size() > 0) {

        bool add_to_list = true;
        for (size_t k = 0; k < detections.size(); k++) {
          const float kRho = detections[k][0];
          const float kTheta = detections[k][1];
          const float kVal = detections[k][2];

          const float kCurAngleSeparation =
              (kTheta - new_vector[1]) * (kTheta - new_vector[1]);
          const float kCurRadiusSeparation =
              (kRho - new_vector[0]) * (kRho - new_vector[0]);

          // If line is close, check which is larger.
          if ((kCurAngleSeparation < kMinAngleSep) &&
              (kCurRadiusSeparation < kMinRadiusSep)) {
            add_to_list = false;

            // overwrite or discard
            if (new_vector[2] > kVal) {
              detections[k] = new_vector;
            }
          }
        }

        // If none was bigger, add to the end.
        if (add_to_list) {
          detections.push_back(new_vector);
        }
      } else {
        // First line, so add it.
        detections.push_back(new_vector);
      }
    }
  }
}

// Event preprocessing prior to first HT.
void Detector::eventPreProcessing(
    const dvs_msgs::EventArray::ConstPtr &orig_msg, Eigen::MatrixXf &points) {

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
    const dvs_msgs::Event &e = orig_msg->events[i];

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
  if (num_events <= FLAGS_hough_1_window_size) {
    return;
  }

  // Reshaping the event array into an Eigen matrix.
  points.resize(2, num_events);
  points.setZero();

  // Add points from the actual message.
  const auto ptr = points.data();
  CHECK_NOTNULL(ptr);

  if (FLAGS_perform_camera_undistortion) {
#pragma omp parallel for
    for (int i = 0; i < num_events; i++) {
      const dvs_msgs::Event &event = feature_msg_.events[i];

      *(ptr + 2 * i) = undist_map_x_(event.y, event.x);
      *(ptr + 2 * i + 1) = undist_map_y_(event.y, event.x);
    }
  } else {
#pragma omp parallel for
    for (int i = 0; i < num_events; i++) {
      const dvs_msgs::Event &event = feature_msg_.events[i];

      *(ptr + 2 * i) = event.x;
      *(ptr + 2 * i + 1) = event.y;
    }
  }
}

// Function for visualizing the current second Hough Space.
void Detector::visualizeSecondHoughSpace(
    const std::vector<cv::Vec3f> &kDetectionsPos,
    const std::vector<cv::Vec3f> &kDetectionsNeg) {

  // Window for visualization.
  cv::Mat line_space_pos(kHough1RadiusResolution,
                         kHough2MsgPerWindow * kHough2TimestepsPerMsg, CV_8UC1,
                         1);
  cv::Mat line_space_neg(kHough1RadiusResolution,
                         kHough2MsgPerWindow * kHough2TimestepsPerMsg, CV_8UC1,
                         1);

#pragma omp parallel for
  for (int i = 0; i < hough2_queue_pos_.size(); i++) {
    for (int j = 0; j < hough2_queue_pos_[i].rows(); j++) {
      for (int k = 0; k < kHough2TimestepsPerMsg; k++) {
        line_space_pos.at<uchar>(j, i * kHough2TimestepsPerMsg + k, 0) =
            hough2_queue_pos_[i](j, k);
      }
    }
    for (int j = 0; j < hough2_queue_neg_[i].rows(); j++) {
      for (int k = 0; k < kHough2TimestepsPerMsg; k++) {
        line_space_neg.at<uchar>(j, i * kHough2TimestepsPerMsg + k, 0) =
            hough2_queue_neg_[i](j, k);
      }
    }
  }

  cv::cvtColor(line_space_pos, line_space_pos, cv::COLOR_GRAY2BGR);
  cv::cvtColor(line_space_neg, line_space_neg, cv::COLOR_GRAY2BGR);

  for (size_t i = 0; i < kDetectionsPos.size(); i++) {
    float rho = kDetectionsPos[i][0], theta = kDetectionsPos[i][1];
    drawPolarCorLine(line_space_pos, rho, theta, cv::Scalar(255, 0, 0));
  }

  for (size_t i = 0; i < kDetectionsNeg.size(); i++) {
    float rho = kDetectionsNeg[i][0], theta = kDetectionsNeg[i][1];
    drawPolarCorLine(line_space_neg, rho, theta, cv::Scalar(0, 0, 255));
  }

  // Flip image for nicer viewing.
  cv::Mat out;

  cv::flip(line_space_pos, line_space_pos, 0);
  cv::flip(line_space_neg, line_space_neg, 0);

  cv::vconcat(line_space_pos, line_space_neg, out);

  cv::imshow("Hough Transform #2", out);
  cv::waitKey(1);
}

void Detector::drawPolarCorLine(cv::Mat &image_space, float rho, float theta,
                                cv::Scalar color) {
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
    int i, int radius, const MatrixHough &total_hough_space,
    int local_threshold, bool polarity, double timestamp,
    std::vector<hough2map::Detector::line> *new_maxima,
    std::vector<int> *new_maxima_value, bool skip_center) {
  int m_l = std::max(i - FLAGS_hough_space_NMS_suppression_radius, 0);
  int m_r = std::min(i + FLAGS_hough_space_NMS_suppression_radius + 1,
                     kHough1AngularResolution);
  int n_l = std::max(radius - FLAGS_hough_space_NMS_suppression_radius, 0);
  int n_r = std::min(radius + FLAGS_hough_space_NMS_suppression_radius + 1,
                     kHough1RadiusResolution);

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
    const std::vector<hough2map::Detector::line> &new_maxima,
    const std::vector<int> &new_maxima_value,
    std::vector<hough2map::Detector::line> *current_maxima) {

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
    const hough2map::Detector::line &new_maximum =
        new_maxima[new_maxima_index[i]];

    bool add_maximum = true;
    // Compare to all other maxima in the output buffer.
    for (int j = 0; j < current_maxima->size(); j++) {
      const hough2map::Detector::line &cur_maximum = (*current_maxima)[j];

      // If no maximum in the output buffer is of the same polarity and within
      // the radius, the current maximum is kept and added to the output
      // buffer.
      if (cur_maximum.polarity == new_maximum.polarity) {
        // Suppression radius.
        float distance =
            (cur_maximum.r - new_maximum.r) * (cur_maximum.r - new_maximum.r) +
            (cur_maximum.theta_idx - new_maximum.theta_idx) *
                (cur_maximum.theta_idx - new_maximum.theta_idx);

        if (distance < FLAGS_hough_space_NMS_suppression_radius *
                           FLAGS_hough_space_NMS_suppression_radius) {
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
bool Detector::isLocalMaxima(const Eigen::MatrixXi &hough_space, int i,
                             int radius) {
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
void Detector::imageCallback(const sensor_msgs::Image::ConstPtr &msg) {
  // Get greyscale image.
  cv::Mat cv_image_raw;
  cv_bridge::CvImagePtr cv_ptr;

  try {
    cv_ptr = cv_bridge::toCvCopy(msg);
  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  cur_greyscale_img_ = cv_ptr->image;
  cv::cvtColor(cur_greyscale_img_, cur_greyscale_img_, cv::COLOR_GRAY2BGR);
}

void Detector::newPoleDetection(double rho, double theta, double window_time,
                                bool pol) {
  // Creating new pole object. This is on the one hand legacy code, on the
  // other hand ready for future on the go map storage, or something like
  // that.
  pole new_pole;

  new_pole.rho = rho;
  new_pole.theta = theta;
  new_pole.polarity = pol;

  // Find the point in time of the first observation.
  double y = camera_resolution_width_;
  double first_observation = (1 / cos(theta)) * (rho - y * sin(theta));
  const double kWindowSizeInSec =
      kHough2MsgPerWindow / FLAGS_event_array_frequency;

  // Convert window time-steps to continous time.
  first_observation *=
      kWindowSizeInSec / (kHough2MsgPerWindow * kHough2TimestepsPerMsg);
  first_observation += window_time;

  new_pole.first_observed = first_observation;

  // The observation timestamp of the pole is still within the timespan
  // covered by the odometry buffer.
  if (first_observation > raw_gps_buffer_.front()[0]) {

    // Query raw gps position at observation beginning time.
    Eigen::Vector3d last_position =
        queryOdometryBuffer(first_observation, raw_gps_buffer_);
    Eigen::Vector3d last_velocity =
        queryOdometryBuffer(first_observation, velocity_buffer_);

    // In the next step we want to inspect the pole at each observation
    // timepoint, so each time it moved from one pixel to the next. This means
    // that we turn the continous spatio-temporal line of the pole observation
    // back into individual x-t points, by sampling the line at every possible x
    // (every horizontal pixel).

    std::vector<Eigen::Vector2d> pixel_pos;
    std::vector<Eigen::Matrix<double, 2, 3>> projection_mats;
    std::vector<Eigen::Matrix3d> transformation_mats;

    // Observation at each horizontal pixel position.
    for (int i = camera_resolution_width_; i > 0; i--) {
      // Timestamp of this observation.
      double observation_timestamp = (1 / cos(theta)) * (rho - i * sin(theta));
      observation_timestamp *=
          kWindowSizeInSec / (kHough2MsgPerWindow * kHough2TimestepsPerMsg);
      observation_timestamp += window_time;

      CHECK_GE(observation_timestamp, first_observation)
          << ":Something is wrong with observation odometry integration!";

      // Integrate odometry to get respective train transformations.
      if (observation_timestamp <= raw_gps_buffer_.back()[0]) {
        // Get the latest odometry.
        Eigen::Vector3d cur_velocity =
            queryOdometryBuffer(observation_timestamp, velocity_buffer_);
        Eigen::Vector2d cur_orientation =
            queryOdometryBuffer(observation_timestamp, orientation_buffer_);

        Eigen::Vector3d new_position = last_position;

        // Compute deltaT.
        const double diff = observation_timestamp - last_position[0];

        // Integrating odom.
        new_position[0] = observation_timestamp;
        new_position[1] += 0.5 * (cur_velocity[1] + last_velocity[1]) * diff;
        new_position[2] += 0.5 * (cur_velocity[2] + last_velocity[2]) * diff;

        last_velocity = cur_velocity;
        last_position = new_position;

        // Assemble train transformation matrix.
        Eigen::Matrix3d train_to_world_transformation;

        train_to_world_transformation << std::cos(cur_orientation[1]),
            std::sin(cur_orientation[1]), new_position[1],
            -std::sin(cur_orientation[1]), std::cos(cur_orientation[1]),
            new_position[2], 0, 0, 1;

        // Compute pole position in global frame.
        Eigen::Matrix3d T_camera_to_world;
        T_camera_to_world = (train_to_world_transformation * C_camera_train_ *
                             camera_train_offset_ * gps_offset_);

        Eigen::Matrix3d T_gps_to_world;
        T_gps_to_world =
            (train_to_world_transformation * C_camera_train_ * gps_offset_);

        // Invert train transformation matrix to get world to camera
        // transformaiton.
        Eigen::Matrix2d rot_part = T_camera_to_world.block<2, 2>(0, 0);
        Eigen::Vector2d trans_part = T_camera_to_world.block<2, 1>(0, 2);

        Eigen::Matrix3d world_to_camera_transformation;
        world_to_camera_transformation = Eigen::Matrix3d::Identity();
        world_to_camera_transformation.block<2, 2>(0, 0) = rot_part.transpose();
        world_to_camera_transformation.block<2, 1>(0, 2) =
            -rot_part.transpose() * trans_part;

        // Formulate projection matrix.
        Eigen::Matrix<double, 2, 3> cam_matrix;
        cam_matrix << intrinsics_[0], 0, intrinsics_[2], 0, 0, 1;

        Eigen::Matrix<double, 2, 3> projection_matrix;
        projection_matrix = cam_matrix * world_to_camera_transformation;

        // Everything I need for a DLT trianguaiton.
        Eigen::Vector2d pixel_position(i, 1);
        pixel_pos.push_back(pixel_position);
        projection_mats.push_back(projection_matrix);
        transformation_mats.push_back(world_to_camera_transformation);
      }
    }

    // Use a Singular Value Decomposition (SVD) to perform the triangulation.
    // This is also known as a Direct Linear Transform (DLT).
    int num_rows = projection_mats.size();

    // At least two observations are required for a triangulation.
    if (num_rows > 2) {
      // Assemble matrix A for DLT.
      Eigen::MatrixXd A;
      A.resize(num_rows, 3);
      for (int i = 0; i < projection_mats.size(); i++) {
        // Convert pixel frame to cam frame.
        double position = ((pixel_pos[i][0] - intrinsics_[3]) / intrinsics_[0]);

        A.row(i) = position * transformation_mats[i].row(1) -
                   transformation_mats[i].row(0);
      }
      // Singular Value decomposition.
      Eigen::BDCSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU |
                                                Eigen::ComputeThinV);

      // Get the last column.
      Eigen::Vector3d x = svd.matrixV().col(svd.matrixV().cols() - 1);

      // Normalize homogenous coordinates.
      new_pole.pos_x = x[0] / x[2];
      new_pole.pos_y = x[1] / x[2];

      // Store new map point in file.
      if (FLAGS_map_output) {

        map_file << std::fixed << "0"
                 << ","
                 << "pole"
                 << "," << new_pole.first_observed << "," << new_pole.pos_x
                 << "," << new_pole.pos_y << ","
                 << "0"
                 << ","
                 << "0" << std::endl;
      }
    }
  }
}

// Just a funciton for creating new line structs.
Detector::line Detector::addMaxima(int angle, int rad, double cur_time,
                                   bool pol) {
  hough2map::Detector::line new_line;

  new_line.ID = 0;
  new_line.r = rad;
  new_line.theta = thetas_1_(angle);
  new_line.theta_idx = angle;
  new_line.time = cur_time;
  new_line.polarity = pol;

  return new_line;
}

// Generalized buffer query function for all odometry buffers.
template <class S, int rows, int cols>
Eigen::Matrix<S, rows, cols> Detector::queryOdometryBuffer(
    const double query_time,
    const std::deque<Eigen::Matrix<S, rows, cols>> &odometry_buffer) {
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

} // namespace hough2map
