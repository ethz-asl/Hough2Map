#include "hough2map/detector.h"

#include <omp.h>
#include <ros/package.h>

#include <chrono>
#include <thread>

namespace hough2map {
Detector::Detector(const ros::NodeHandle &nh, const ros::NodeHandle &nh_private,
                   const image_transport::ImageTransport &img_pipe)
    : nh_(nh), nh_private_(nh_private), img_pipe_(img_pipe) {
  // Load Calibration
  loadCamConfigFromParams();

  // Load Config
  loadConfigFromParams();

  // Configure Tracker Manager
  tracker_mgr_.init(tracker_mgr_config_);

  // Output file for the map data.
  if (!output_config_.map_file.empty()) {
    map_file_.open(output_config_.map_file);

    if (map_file_.is_open()) {
      map_file_ << "id,type,time,x,y,orientation,velocity,weight\n";
    } else {
      LOG(FATAL) << "Could not open file:" << output_config_.map_file << std::endl;
    }
  }

  // Timing statistics for performance evaluation.
  profiling_.total_events_timing_us = 0.0;
  profiling_.total_msgs_timing_ms = 0.0;
  profiling_.total_events = 0;
  profiling_.total_msgs = 0;

  // TrackerPole counter
  pole_count_ = 1;

  // Update Hough1 resolutions
  hough1_config_.radial_resolution = (int)(cam_config_.cam_res_width * 1.1);

  // Compute undistortion for given camera parameters.
  computeUndistortionMapping();

  omp_set_num_threads(kNumThreads);

  // Various subscribers and publishers for event and odometry data.
  event_sub_ = nh_.subscribe("/dvs/events", 0, &Detector::eventCallback, this);
  odom_pose_sub_ = nh_.subscribe("/odometry", 0, &Detector::poseCallback, this);

  feature_pub_ = nh_.advertise<dvs_msgs::EventArray>("/feature_events", 1);

  // Plot current hough detections in the video.
  if (output_config_.rviz) {
    // Setup subscribers and publishers
    image_raw_sub_ = nh_.subscribe("/dvs/image_raw", 0, &Detector::imageCallback, this);
    hough1_img_pub_ = img_pipe_.advertise("/hough1/image", 10);
    xt_img_pub_ = img_pipe_.advertise("/xt_space/image", 10);

    pole_viz_pub_ = nh_.advertise<visualization_msgs::Marker>("/poles", 10);
    cam_viz_pub_ = nh_.advertise<visualization_msgs::Marker>("/cams", 10);
    pose_buffer_pub_ = nh_.advertise<nav_msgs::Path>("/pose_buffer", 10);

    pole_marker_.header.frame_id = "map";
    pole_marker_.header.stamp = ros::Time();
    pole_marker_.ns = "";
    pole_marker_.id = 0;
    pole_marker_.type = visualization_msgs::Marker::CYLINDER;
    pole_marker_.action = visualization_msgs::Marker::ADD;
    pole_marker_.pose.position.x = 0;
    pole_marker_.pose.position.y = 0;
    pole_marker_.pose.position.z = 5;
    pole_marker_.pose.orientation.w = 1.0;
    pole_marker_.pose.orientation.x = 0.0;
    pole_marker_.pose.orientation.y = 0.0;
    pole_marker_.pose.orientation.z = 0.0;
    pole_marker_.scale.x = 0.15;
    pole_marker_.scale.y = 0.15;
    pole_marker_.scale.z = 10;
    pole_marker_.color.r = 1.0;
    pole_marker_.color.g = 1.0;
    pole_marker_.color.b = 0.0;
    pole_marker_.color.a = 0.6;

    cam_marker_.header.frame_id = "map";
    cam_marker_.header.stamp = ros::Time();
    cam_marker_.ns = "";
    cam_marker_.id = 0;
    cam_marker_.type = visualization_msgs::Marker::SPHERE;
    cam_marker_.action = visualization_msgs::Marker::ADD;
    cam_marker_.pose.position.x = 0;
    cam_marker_.pose.position.y = 0;
    cam_marker_.pose.position.z = 0;
    cam_marker_.pose.orientation.x = 0.0;
    cam_marker_.pose.orientation.y = 0.0;
    cam_marker_.pose.orientation.z = 0.0;
    cam_marker_.pose.orientation.w = 1.0;
    cam_marker_.scale.x = 0.2;
    cam_marker_.scale.y = 0.2;
    cam_marker_.scale.z = 0.2;
    cam_marker_.color.r = 0.0;
    cam_marker_.color.g = 0.0;
    cam_marker_.color.b = 1.0;
    cam_marker_.color.a = 0.4;

    pose_buffer_path_.header.frame_id = "map";
  }

  // Initializig theta, sin and cos values for first and second Hough
  // transform.
  initializeSinCosMap(thetas_1_, polar_param_mapping_1_, hough1_config_.angle_min,
                      hough1_config_.angle_max, hough1_config_.angular_resolution);
  // initializeSinCosMap(thetas_2_, polar_param_mapping_2_, kHough2MinAngle, kHough2MaxAngle,
  //                     kHough2AngularResolution);
}

Detector::~Detector() {
  if (!output_config_.map_file.empty() && map_file_.is_open()) {
    map_file_.close();
  }
}

// Function to precompute angles, sin and cos values for a vectorized version
// of the HT. Templated to deal with float and double accuracy.
template <typename DerivedVec, typename DerivedMat>
void Detector::initializeSinCosMap(Eigen::EigenBase<DerivedVec> &angles,
                                   Eigen::EigenBase<DerivedMat> &sin_cos_map, const int kMinAngle,
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

void Detector::loadCamConfigFromParams() {
  bool ld = true;

  ros::NodeHandle *nh = &nh_private_;

  // Camera Resolution Info
  std::vector<int> res;
  ld = ld && nh->getParam("cam/resolution", res);
  CHECK_EQ(res.size(), 2);
  cam_config_.cam_res_width = res[0];
  cam_config_.cam_res_height = res[1];

  // Misc Info
  ld = ld && nh->getParam("cam/events_frequency", cam_config_.evt_arr_frequency);
  ld = ld && nh->getParam("cam/distortion_acceptable", cam_config_.acceptable_dist);
  ld = ld && nh->getParam("cam/distorted", cam_config_.perform_undist);

  // Intrinsics
  std::vector<double> intrinsics;
  ld = ld && nh->getParam("cam/intrinsics", intrinsics);
  CHECK_EQ(intrinsics.size(), 4) << ": Not enough intrinsics!";
  for (size_t i = 0; i < intrinsics.size(); i++) {
    cam_config_.intrinsics[i] = intrinsics[i];
  }

  // Distortion Coeffs
  std::vector<double> dist_coeffs;
  ld = ld && nh->getParam("cam/distortion_coeffs", dist_coeffs);
  CHECK_EQ(dist_coeffs.size(), 4) << ": Not enough dist coeffs!";
  for (size_t i = 0; i < dist_coeffs.size(); i++) {
    cam_config_.dist_coeffs[i] = dist_coeffs[i];
  }

  // Extrinsics
  std::vector<double> flat_extrinsics;
  Eigen::Matrix<double, 4, 4> flat_extrinsics_mat;
  ld = ld && nh->getParam("cam/cam_to_body", flat_extrinsics);
  CHECK_EQ(flat_extrinsics.size(), 16) << ": Not enough extrinsics!";
  for (size_t i = 0; i < flat_extrinsics.size(); i++) {
    flat_extrinsics_mat(i) = flat_extrinsics[i];
  }
  cam_config_.T_cam_to_body.matrix() = flat_extrinsics_mat.transpose();

  // Debug message
  if (!ld) {
    ROS_WARN_STREAM("Failed to load camera calibration! Using default profile...");
  } else {
    ROS_INFO_STREAM("Camera Calibration loaded!");
  }
}

void Detector::loadConfigFromParams() {
  bool ld = true;

  ros::NodeHandle *nh = &nh_private_;

  // Hough 1 Config
  ld = ld && nh->getParam("hough1/angular_resolution", hough1_config_.angular_resolution);
  ld = ld && nh->getParam("hough1/radial_resolution", hough1_config_.radial_resolution);
  ld = ld && nh->getParam("hough1/window_size", hough1_config_.window_size);
  ld = ld && nh->getParam("hough1/threshold", hough1_config_.threshold);
  ld = ld && nh->getParam("hough1/nms_radius", hough1_config_.nms_radius);
  ld = ld && nh->getParam("hough1/angle_range/min", hough1_config_.angle_min);
  ld = ld && nh->getParam("hough1/angle_range/max", hough1_config_.angle_max);

  // Detector Config
  ld = ld && nh->getParam("detector/evt_subsample_fac", detector_config_.evt_subsample_fac);
  ld = ld && nh->getParam("detector/buffer_size_s", detector_config_.buffer_size_s);
  ld = ld && nh->getParam("detector/tsteps_per_msg", detector_config_.tsteps_per_msg);
  ld = ld && nh->getParam("detector/msg_per_window", detector_config_.msg_per_window);
  ld = ld && nh->getParam("detector/centroid_find_window", detector_config_.centroid_find_window);
  ld = ld && nh->getParam("detector/centroid_find_thresh", detector_config_.centroid_find_thresh);
  ld = ld && nh->getParam("detector/triangulation_sv_thresh", detector_config_.triangln_sv_thresh);
  ld = ld && nh->getParam("detector/min_baseline_dist", detector_config_.min_baseline_dist);
  ld = ld && nh->getParam("detector/max_reproj_err_px", detector_config_.max_reproj_err_px);
  ld = ld && nh->getParam("detector/dist_thresh/min", detector_config_.dist_thresh_min);
  ld = ld && nh->getParam("detector/dist_thresh/max", detector_config_.dist_thresh_max);

  // Tracker Manager Config
  ld = ld && nh->getParam("tracker_mgr/centr_buffer_l", tracker_mgr_config_.centroid_buffer_size);
  ld = ld && nh->getParam("tracker_mgr/spawn_thresh", tracker_mgr_config_.tracker_spawn_threshold);
  ld = ld && nh->getParam("tracker_mgr/dxdt_cluster_tol_px", tracker_mgr_config_.dx_cluster_tol);
  ld = ld && nh->getParam("tracker_mgr/max_dx_allowed", tracker_mgr_config_.max_dx_allowed);
  ld = ld && nh->getParam("tracker_mgr/maturity_age", tracker_mgr_config_.maturity_age);
  ld = ld && nh->getParam("tracker_mgr/dxdt_range/min", tracker_mgr_config_.min_dx_dt);
  ld = ld && nh->getParam("tracker_mgr/dxdt_range/max", tracker_mgr_config_.max_dx_dt);

  // Output Config
  ld = ld && nh->getParam("output/map_file", output_config_.map_file);
  ld = ld && nh->getParam("output/rviz", output_config_.rviz);

  // Tracker Config
  ld = ld && nh->getParam("tracker/linearity_window",
                          tracker_mgr_config_.tracker_config.linearity_window);
  ld = ld && nh->getParam("tracker/linearity_tol_px",
                          tracker_mgr_config_.tracker_config.linearity_tol_px);

  // Debug message
  if (!ld) {
    ROS_WARN_STREAM("Failed to load params! Using defaults");
  } else {
    ROS_INFO_STREAM("All Params loaded!");
  }
}

void Detector::computeUndistortionMapping() {
  // Setup camera intrinsics from calibration file.
  cv::Mat camera_matrix =
      (cv::Mat1d(3, 3) << cam_config_.intrinsics[0], 0, cam_config_.intrinsics[2], 0,
       cam_config_.intrinsics[1], cam_config_.intrinsics[3], 0, 0, 1);
  cv::Mat distortionCoefficients =
      (cv::Mat1d(1, 4) << cam_config_.dist_coeffs[0], cam_config_.dist_coeffs[1],
       cam_config_.dist_coeffs[2], cam_config_.dist_coeffs[3]);

  cam_config_.undist_map_x.resize(cam_config_.cam_res_height, cam_config_.cam_res_width);
  cam_config_.undist_map_y.resize(cam_config_.cam_res_height, cam_config_.cam_res_width);

  // Compute undistortion mapping.
  for (int i = 0; i < cam_config_.cam_res_width; i++) {
    for (int j = 0; j < cam_config_.cam_res_height; j++) {
      cv::Mat_<cv::Point2f> points(1, 1);
      points(0) = cv::Point2f(i, j);
      cv::Mat dst;

      cv::undistortPoints(points, dst, camera_matrix, distortionCoefficients);
      const float u = cam_config_.intrinsics[0] * dst.at<float>(0, 0) + cam_config_.intrinsics[2];
      const float v = cam_config_.intrinsics[1] * dst.at<float>(0, 1) + cam_config_.intrinsics[3];

      CHECK_GT(u, 0.0 - cam_config_.acceptable_dist)
          << "Horizontal undistortion is larger than expected";
      CHECK_LT(u, cam_config_.cam_res_width + cam_config_.acceptable_dist)
          << "Horizontal undistortion is larger than expected";
      CHECK_GT(v, 0.0 - cam_config_.acceptable_dist)
          << "Vertical undistortion is larger than expected";
      CHECK_LT(v, cam_config_.cam_res_height + cam_config_.acceptable_dist)
          << "Vertical undistortion is larger than expected";

      cam_config_.undist_map_x(j, i) = u;
      cam_config_.undist_map_y(j, i) = v;
    }
  }
}

void Detector::poseCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr &msg) {
  // Add to pose buffer
  pose_buffer_.push_back(msg);

  // TODO: Time correction offset must be done in the source
  const double kAlignedTimestamp = msg->header.stamp.toSec();

  // Clean up buffer
  while (kAlignedTimestamp - pose_buffer_.front()->header.stamp.toSec() >
         detector_config_.buffer_size_s) {
    pose_buffer_.pop_front();
  }

  // Viz pose buffer path
  if (output_config_.rviz) {
    pose_buffer_path_.header.stamp = msg->header.stamp;
    pose_buffer_path_.poses.clear();

    for (int i = 0; i < pose_buffer_.size(); i++) {
      geometry_msgs::PoseStamped ps_;
      ps_.header = pose_buffer_[i]->header;
      ps_.pose = pose_buffer_[i]->pose.pose;
      ps_.pose.position.z = 0;
      pose_buffer_path_.poses.push_back(ps_);
    }

    pose_buffer_pub_.publish(pose_buffer_path_);
  }
}

void Detector::eventCallback(const dvs_msgs::EventArray::ConstPtr &msg) {
  const auto kStartTime = std::chrono::high_resolution_clock::now();

  feature_msg_.header = msg->header;
  feature_msg_.width = msg->width;
  feature_msg_.height = msg->height;

  // If initialized then make sure the last FLAGS_hough_1_window_size events
  // are prepended to the current list of events and remove older events.
  if (feature_msg_.events.size() > hough1_config_.window_size) {
    std::copy(feature_msg_.events.end() - hough1_config_.window_size, feature_msg_.events.end(),
              feature_msg_.events.begin());
    feature_msg_.events.resize(hough1_config_.window_size);
  }

  // Reshaping the event array into an Eigen matrix.
  Eigen::MatrixXf points;
  eventPreProcessing(msg, points);

  // Number of events after filtering and subsampling.
  int num_events = feature_msg_.events.size();
  CHECK_GE(num_events, 1);

  // Check there are enough events for our window size. This is only relevant
  // during initialization.
  if (num_events <= hough1_config_.window_size) {
    return;
  }

  // Computing all the radii for each theta hypothesis. This parameter pair
  // forms the Hough space. This is done all at once for each event.
  Eigen::MatrixXi radii;
  radii.resize(hough1_config_.angular_resolution, num_events);
  radii = (polar_param_mapping_1_ * points).cast<int>();

  // Total Hough Space at NMS Intervals
  // kNmsBatchCount is the reduced number of iterations. This is basically
  // the number of sub-batches that will be processed in parallel
  CHECK_GE(kNumThreads, 1);
  int nms_recompute_window =
      std::ceil(float(num_events - hough1_config_.window_size) / kNumThreads);
  nms_recompute_window = std::max(nms_recompute_window, hough1_config_.window_size);
  CHECK_GT(nms_recompute_window, 0);
  const int kNmsBatchCount =
      std::ceil(float(num_events - hough1_config_.window_size) / nms_recompute_window);

  // Initializing total Hough spaces. Total means the Hough Space for a full
  // current window, rather than the Hough Space of an individual event. It is
  // therefore the sum of all the Hough Spaces of the events in the current
  // window.
  std::vector<Eigen::MatrixXi> total_hough_spaces_neg(kNmsBatchCount);

  // At this point we are starting the parallelisation scheme of this
  // pipeline. As events have to be processed sequentially, the sequence is
  // split into parts. The beginning of each part depends on the end of the
  // previous one. This beginning state is computed using a full HT and full
  // NMS.

  // Resetting the accumulator cells of all Hough spaces for the beginning of
  // all batches.
#pragma omp parallel for
  for (int i = 0; i < kNmsBatchCount; i++) {
    total_hough_spaces_neg[i].resize(hough1_config_.radial_resolution,
                                     hough1_config_.angular_resolution);
    total_hough_spaces_neg[i].setZero();
  }

  // Computing total Hough space every N steps, so for the beginning of each
  // parallelisation batch. This depends on the last FLAGS_hough_1_window_size
  // of the previous batch.
#pragma omp parallel for
  for (int i = 0; i < kNmsBatchCount; i++) {
    computeFullHoughTransform(i, nms_recompute_window, total_hough_spaces_neg[i], radii);
  }

  // Each event is treated as a timestep. For each of these timesteps we keep
  // the active set of maxima in the Hough Space. These are basically the line
  // detections at each timestep. This whole storage is pre-initialized to
  // make it ready for parallelizing the whole process.
  std::vector<std::vector<hough2map::HoughLine>> maxima_list(num_events);

  // As we compute a full HT at the beginning of each batch, we also need a
  // full NMS. This is computed here.
#pragma omp parallel for
  for (int k = 0; k < kNmsBatchCount; k++) {
    computeFullNMS(k, nms_recompute_window, total_hough_spaces_neg[k], maxima_list);
  }

  // Within each parallelised NMS batch, we can now perform the rest of the
  // computations iterativels, processing the events in their correct
  // sequence. This is done in parallel for all batches.
#pragma omp parallel for
  for (int k = 0; k < kNmsBatchCount; k++) {
    itterativeNMS(k, nms_recompute_window, total_hough_spaces_neg[k], maxima_list, radii);
  }
  // If visualizations are turned on display them in the video stream.
  if (output_config_.rviz) {
    visualizeCurrentLineDetections(maxima_list);
  }

  // Run the second Hough Transform for spatio-temporal tracking.
  heuristicTrack(maxima_list);

  // Publish events that were part of the Hough transform (because they were
  // not filtered out).
  feature_pub_.publish(feature_msg_);

  if (num_events > 0) {
    std::chrono::duration<double, std::micro> duration_us =
        std::chrono::high_resolution_clock::now() - kStartTime;

    profiling_.total_events_timing_us += duration_us.count();
    profiling_.total_msgs_timing_ms += duration_us.count() / 1000.0;

    profiling_.total_events += num_events;
    profiling_.total_msgs++;

    LOG(INFO) << detector_name_ << std::fixed << std::setprecision(2) << std::setfill(' ')
              << " speed: " << std::setw(6)
              << profiling_.total_events_timing_us / profiling_.total_events << " us/event | "
              << std::setw(6) << profiling_.total_msgs_timing_ms / profiling_.total_msgs
              << " ms/msg | " << std::setw(6) << num_events << " e/msg";
  }
}

// Visualizing the current line detections of the first Hough Transform.
// This function only visualizes vertical lines, for other orientations it needs
// to be adjusted.
void Detector::visualizeCurrentLineDetections(
    const std::vector<std::vector<hough2map::HoughLine>> &cur_maxima_list) {
  int num_events = feature_msg_.events.size();

  int negative_detections[cam_config_.cam_res_width] = {0};

  // Getting the horizontal positions of all vertical line detections.
  for (int i = 0; i < num_events; i++) {
    const dvs_msgs::Event &e = feature_msg_.events[i];
    for (auto &maxima : cur_maxima_list[i]) {
      negative_detections[maxima.r] = 1;
    }
  }

  cv::Mat cur_frame = cur_greyscale_img_;

  // Plottin current line detections.
  for (int i = 0; i < cam_config_.cam_res_width; i++) {
    if (negative_detections[i] == 1) {
      cv::line(cur_frame, cv::Point(i, 0), cv::Point(i, cam_config_.cam_res_height),
               cv::Scalar(0, 0, 255), 2, 8);
    }
  }

  hough1_img_pub_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", cur_frame).toImageMsg());
}

// Performing itterative Non-Maximum suppression on the current batch of
// events based on a beginning Hough Space.
void Detector::itterativeNMS(const int time_step, const int nms_recompute_window,
                             Eigen::MatrixXi &total_hough_space_neg,
                             std::vector<std::vector<hough2map::HoughLine>> &cur_maxima_list,
                             const Eigen::MatrixXi &radii) {
  std::vector<hough2map::HoughLine> new_maxima;
  std::vector<int> new_maxima_value;
  int num_events = feature_msg_.events.size();

  /* Iterative Hough transform */

  // Itterating over all events which are part of this current batch. These
  // will be added and removed through the iteration process.
  const int left = hough1_config_.window_size + time_step * nms_recompute_window + 1;
  const int right = std::min(left + nms_recompute_window - 1, num_events);
  CHECK_GE(right, left);

  for (int l = left; l < right; l++) {
    // Getting the event that gets added right now.
    const dvs_msgs::Event &event = feature_msg_.events[l];
    const double kTimestamp = event.ts.toSec();
    CHECK_GE(l - 1, 0);

    // Establishing the lists of maxima for this timestep and the ones of the
    // previous timestep
    std::vector<hough2map::HoughLine> &current_maxima = cur_maxima_list[l];
    std::vector<hough2map::HoughLine> &previous_maxima = cur_maxima_list[l - 1];

    // Incrementing the accumulator cells for the current event.
    updateHoughSpaceVotes(true, l, radii, total_hough_space_neg);

    // Find the oldest event in the current window and get ready to remove it.
    const int kLRemove = l - hough1_config_.window_size;
    CHECK_GE(l - hough1_config_.window_size, 0);
    const dvs_msgs::Event &event_remove = feature_msg_.events[kLRemove];

    // Decrement the accumulator cells for the event to be removed.
    updateHoughSpaceVotes(false, kLRemove, radii, total_hough_space_neg);

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
    for (int i = 0; i < hough1_config_.angular_resolution; ++i) {
      const int kRadius = radii(i, l);

      if ((kRadius >= 0) && (kRadius < hough1_config_.radial_resolution)) {
        updateIncrementedNMS(kTimestamp, i, kRadius, total_hough_space_neg, previous_maxima,
                             discard, new_maxima, new_maxima_value);
      }
    }

    // For accumulator cells that got decremented.
    for (int i = 0; i < hough1_config_.angular_resolution; i++) {
      const int kRadius = radii(i, kLRemove);

      if ((kRadius >= 0) && (kRadius < hough1_config_.radial_resolution)) {
        updateDecrementedNMS(kTimestamp, i, kRadius, total_hough_space_neg, previous_maxima,
                             discard, new_maxima, new_maxima_value);
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
          const hough2map::HoughLine &kMaximum = previous_maxima[i];

          new_maxima.push_back(kMaximum);

          new_maxima_value.push_back(total_hough_space_neg(kMaximum.r, kMaximum.theta_idx));
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
            const hough2map::HoughLine &kPreviousMaximum = previous_maxima[i];

            bool found = false;
            for (const hough2map::HoughLine &current_maximum : current_maxima) {
              if ((current_maximum.r == kPreviousMaximum.r) &&
                  (current_maximum.theta_idx == kPreviousMaximum.theta_idx)) {
                found = true;
                break;
              }
            }

            if (!found) {
              discard[i] = true;

              addMaximaInRadius(kPreviousMaximum.theta_idx, kPreviousMaximum.r,
                                total_hough_space_neg, hough1_config_.threshold, kTimestamp,
                                &new_maxima, &new_maxima_value, true);
            }
          }
        }

        if (kNumMaximaCandidates < new_maxima.size()) {
          applySuppressionRadius(new_maxima, new_maxima_value, &current_maxima);
        } else {
          break;
        }
      }

      for (int i = 0; i < previous_maxima.size(); i++) {
        const hough2map::HoughLine &kPreviousMaximum = previous_maxima[i];
        for (const hough2map::HoughLine &current_maximum : current_maxima) {
          if (!((current_maximum.r == kPreviousMaximum.r) &&
                (current_maximum.theta_idx == kPreviousMaximum.theta_idx))) {
            maxima_updates_.push_back(current_maximum);
          }
        }
      }
    }
  }
}

// Updating the iterative Non-Maximum suppression for decremented events.
void Detector::updateDecrementedNMS(const double kTimestamp, const int kAngle, const int kRadius,
                                    const Eigen::MatrixXi &hough_space,
                                    const std::vector<hough2map::HoughLine> &previous_maxima,
                                    std::vector<int> &discard,
                                    std::vector<hough2map::HoughLine> &new_maxima,
                                    std::vector<int> &new_maxima_value) {
  // If decremented accumulator cell was previously a maximum, remove it. If
  // it's still a maximum, we will deal with it later.
  int k = 0;
  bool skip_neighborhood = false;
  for (const hough2map::HoughLine &maximum : previous_maxima) {
    if ((kRadius == maximum.r) && (kAngle == maximum.theta_idx)) {
      // Mark as discarded since we will added already
      // in the next step if it still is above the threshold.
      discard[k] = true;

      // Re-add to list of possible maxima for later pruning.
      addMaximaInRadius(kAngle, kRadius, hough_space, hough1_config_.threshold, kTimestamp,
                        &new_maxima, &new_maxima_value);

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
    const int m_r = std::min(kAngle + 1, hough1_config_.angular_resolution - 1);
    const int n_l = std::max(kRadius - 1, 0);
    const int n_r = std::min(kRadius + 1, hough1_config_.radial_resolution - 1);
    for (int m = m_l; m <= m_r; m++) {
      for (int n = n_l; n <= n_r; n++) {
        // The center is a separate case.
        if ((m == kAngle) && (n == kRadius)) {
          continue;
        }

        // Any neighbor points now larger and a maximum?
        if ((hough_space(kRadius, kAngle) + 1 == hough_space(n, m)) &&
            (hough_space(n, m) > hough1_config_.threshold) && isLocalMaxima(hough_space, m, n)) {
          // Add to temporary storage.
          new_maxima.push_back(addMaxima(m, n, kTimestamp));
          new_maxima_value.push_back(hough_space(n, m));
        }
      }
    }
  }
}

// Updating the iterative Non-Maximum suppression for incrementing events.
bool Detector::updateIncrementedNMS(const double kTimestamp, const int kAngle, const int kRadius,
                                    const Eigen::MatrixXi &hough_space,
                                    const std::vector<hough2map::HoughLine> &previous_maxima,
                                    std::vector<int> &discard,
                                    std::vector<hough2map::HoughLine> &new_maxima,
                                    std::vector<int> &new_maxima_value) {
  // If any of the surrounding ones are equal the center
  // for sure is not a local maximum.
  bool skip_center = false;

  // Iterate over neighbourhood to check if we might have
  // supressed a surrounding maximum by growing.
  const int m_l = std::max(kAngle - 1, 0);
  const int m_r = std::min(kAngle + 1, hough1_config_.angular_resolution - 1);
  const int n_l = std::max(kRadius - 1, 0);
  const int n_r = std::min(kRadius + 1, hough1_config_.radial_resolution - 1);
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
        for (const hough2map::HoughLine &maximum : previous_maxima) {
          if ((n == maximum.r) && (m == maximum.theta_idx)) {
            // We need to discard an old maximum.
            discard[k] = true;

            // And add a new one.
            addMaximaInRadius(m, n, hough_space, hough1_config_.threshold, kTimestamp, &new_maxima,
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
  if ((hough_space(kRadius, kAngle) > hough1_config_.threshold) &&
      isLocalMaxima(hough_space, kAngle, kRadius)) {
    bool add_maximum = true;
    // Check if it was a maximum previously.
    for (const auto &maximum : previous_maxima) {
      if ((kRadius == maximum.r) && (kAngle == maximum.theta_idx)) {
        add_maximum = false;
        break;
      }
    }

    // If required, add it to the list.
    if (add_maximum) {
      new_maxima.push_back(addMaxima(kAngle, kRadius, kTimestamp));
      new_maxima_value.push_back(hough_space(kRadius, kAngle));
    }
  }
  return false;
}

// Computing a full Non-Maximum Suppression for a given current Hough Space.
void Detector::computeFullNMS(const int time_step, const int nms_recompute_window,
                              const Eigen::MatrixXi &total_hough_space_neg,
                              std::vector<std::vector<hough2map::HoughLine>> &cur_maxima_list) {
  // Index of the current event in the frame of all events of the current
  // message with carry-over from previous message.
  const int kNmsIndex = hough1_config_.window_size + time_step * nms_recompute_window;
  std::vector<hough2map::HoughLine> &current_maxima = cur_maxima_list[kNmsIndex];

  // New detected maxima and their value.
  std::vector<hough2map::HoughLine> new_maxima;
  std::vector<int> new_maxima_value;

  // Checking every angle and radius hypothesis.
  for (int i = 0; i < hough1_config_.angular_resolution; i++) {
    for (int j = 0; j < hough1_config_.radial_resolution; j++) {
      // Get the current events for a current time stamp.
      const dvs_msgs::Event &event = feature_msg_.events[kNmsIndex];

      // Checking Hough space, whether it is larger than threshold
      // and larger than neighbors.
      if (total_hough_space_neg(j, i) > hough1_config_.threshold) {
        if (isLocalMaxima(total_hough_space_neg, i, j)) {
          // Add as a possible maximum to the list.
          new_maxima.push_back(addMaxima(i, j, event.ts.toSec()));
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
void Detector::computeFullHoughTransform(const int time_step, const int nms_recompute_window,
                                         Eigen::MatrixXi &total_hough_space_neg,
                                         const Eigen::MatrixXi &radii) {
  // Looping over all events that have an influence on the current total
  // Hough space, so the past 300.
  const int kRight = hough1_config_.window_size + time_step * nms_recompute_window;
  const int kLeft = kRight - hough1_config_.window_size;
  CHECK_GT(kRight, kLeft);

  for (int j = kRight; j > kLeft; j--) {
    // Looping over all confirmed hypothesis and adding them to the Hough
    // space.
    updateHoughSpaceVotes(true, j, radii, total_hough_space_neg);
  }
}

// Incrementing a HoughSpace for a certain event.
void Detector::updateHoughSpaceVotes(const bool increment, const int event_idx,
                                     const Eigen::MatrixXi &radii,
                                     Eigen::MatrixXi &hough_space_neg) {
  // Looping over all confirmed hypothesis and adding or removing them from the
  // Hough space.
  for (int k = 0; k < hough1_config_.angular_resolution; k++) {
    const int kRadius = radii(k, event_idx);
    // making sure the parameter set is within the domain of the HS.
    if ((kRadius >= 0) && (kRadius < hough1_config_.radial_resolution)) {
      // Incrementing or decrement the accumulator cells.
      if (increment) {
        hough_space_neg(kRadius, k)++;
      } else {
        hough_space_neg(kRadius, k)--;
      }
    }
  }
}

void Detector::heuristicTrack(
    const std::vector<std::vector<hough2map::HoughLine>> &cur_maxima_list) {
  int num_events = feature_msg_.events.size();

  // Second Hough tranform. x-t-space
  // Track the obtained maxima over time.

  // Step 1: Reshape maxima list into x-t data.
  // Step 2: Discretizing maxima into timesteps.

  Eigen::MatrixXi tracked_maxima;
  tracked_maxima.resize(hough1_config_.radial_resolution, detector_config_.tsteps_per_msg);
  tracked_maxima.setZero();

  const double kTimestampMsgBegin = feature_msg_.events[0].ts.toSec();

  CHECK_GT(detector_config_.tsteps_per_msg, 0);
  const double kColumnLengthInSec =
      (1.0 / cam_config_.evt_arr_frequency) / (detector_config_.tsteps_per_msg);

  double tracker_last_t = tracker_mgr_.getLatestTime();
  std::vector<int> prev_maxima_px_list;
  std::vector<PointTX> new_points;

  for (auto &&max_up : maxima_updates_) {
    PointTX p = {max_up.time, max_up.r};
    if (max_up.time > tracker_last_t && max_up.r < cam_config_.cam_res_width) {
      PointTX p = {max_up.time, max_up.r};
      new_points.push_back(p);
      if (!output_config_.map_file.empty() && map_file_.is_open()) {
        map_file_ << std::fixed << p.t << ',' << p.x << std::endl;
      }
    }
  }

  maxima_updates_.clear();

  for (int i = 0; i < num_events; i++) {
    const dvs_msgs::Event &e = feature_msg_.events[i];
    for (auto &maxima : cur_maxima_list[i]) {
      // Not every message is exactly 33ms long. To deal with this, we currently
      // squeeze longer messages down to be 33ms long.

      const double t = e.ts.toSec();

      const int time_idx = std::min((int)std::floor((t - kTimestampMsgBegin) / kColumnLengthInSec),
                                    detector_config_.tsteps_per_msg - 1);

      CHECK_LT(time_idx, detector_config_.tsteps_per_msg)
          << ": Something wrong with the time index!" + time_idx;

      // This discretizes the maxima. Many will be same or super close, so we
      // accumulate them.
      tracked_maxima(maxima.r, time_idx)++;

      // Add new point to list only if:
      // - The line is found after last buffer time
      // - The line is not beyond camera width
      // - The maxima already doesn't exist in prev list

      // &&
      //     std::find(prev_maxima_px_list.begin(), prev_maxima_px_list.end(), maxima.r) ==
      //         prev_maxima_px_list.end()

      // if (t > tracker_last_t && maxima.r < cam_config_.cam_res_width) {
      //   PointTX p = {t, maxima.r};
      //   new_points.push_back(p);
      // }
    }

    // Update prev maxima list
    prev_maxima_px_list.resize(cur_maxima_list[i].size());
    for (size_t j = 0; j < cur_maxima_list[i].size(); j++) {
      prev_maxima_px_list[j] = cur_maxima_list[i][j].r;
    }
  }

  tracker_mgr_.track(new_points);

  const double kWindowSizeInSec = detector_config_.msg_per_window / cam_config_.evt_arr_frequency;
  const double kWindowEndTime = feature_msg_.events[num_events - 1].ts.toSec();

  // Window for data storage.
  houghout_queue_.push_back(tracked_maxima);
  houghout_queue_last_t =
      kTimestampMsgBegin + kColumnLengthInSec * (detector_config_.tsteps_per_msg - 1);

  // Removing old stuff when the window is full.
  if (houghout_queue_.size() > detector_config_.msg_per_window) {
    houghout_queue_.pop_front();
  }

  // Get centroids
  for (int i = 0; i < tracked_maxima.cols(); i++) {
    // Generate timestamp
    double t = kTimestampMsgBegin + i * kColumnLengthInSec;

    // Get list of centroids
    std::vector<int> centroids = getClusteringCentroids(tracked_maxima.col(i));
    cluster_centroids_.push_back(centroids);

    // Track centroids
    // tracker_mgr_.track(t, centroids);
  }

  // Crop the cluster centroids deque
  while (cluster_centroids_.size() >
         detector_config_.tsteps_per_msg * detector_config_.msg_per_window) {
    cluster_centroids_.pop_front();
  }

  // Get finished Trackers
  std::vector<Tracker> trackers = tracker_mgr_.getFinishedTrackers(houghout_queue_last_t);
  for (auto &&tracker : trackers) {
    triangulateTracker(tracker);
    if (output_config_.rviz) {
      viz_trackers_.push_back(tracker);
    }
  }

  // Plot the trackers.
  if (output_config_.rviz) {
    visualizeTracker();
  }
}

void Detector::triangulateTracker(Tracker tracker) {
  auto tracker_points = tracker.getPoints();

  TrackerPole new_pole;

  new_pole.ID = pole_count_;
  new_pole.first_observed = tracker_points.front().x;

  std::vector<Eigen::Vector2d> pixel_pos;
  std::vector<Eigen::Affine2d> transformation_mats;

  for (auto &&tracker_pt : tracker_points) {
    if (tracker_pt.t >= pose_buffer_.front()->header.stamp.toSec() &&
        tracker_pt.t <= pose_buffer_.back()->header.stamp.toSec()) {
      // Get the latest odometry.

      auto cur_pose = queryPoseAtTime(tracker_pt.t);

      // Assemble train transformation matrix.
      Eigen::Affine3d T_body_to_world;
      tf2::fromMsg(cur_pose.pose.pose, T_body_to_world);

      Eigen::Affine3d T_cam_to_world = T_body_to_world * cam_config_.T_cam_to_body;

      // Invert train transformation matrix to get world to camera
      Eigen::Affine3d T_world_to_cam = T_cam_to_world.inverse();

      // Reduce 3D tf to 2D tf
      Eigen::Affine2d T_world_to_cam_reduced;
      T_world_to_cam_reduced = Eigen::Matrix3d::Identity();

      // NOTE: In camera frame, yaw is about Y axis..
      // ==> We ideally delete [Row 1] and [Col 2]. (index 0)
      // Hacky atan2 of averages in case of skew matrices
      auto yaw_mat_ = T_world_to_cam.matrix();
      double yaw_ = atan2(yaw_mat_(2, 0) - yaw_mat_(0, 1), yaw_mat_(0, 0) + yaw_mat_(2, 1));
      Eigen::Rotation2Dd R_world_to_cam_reduced(yaw_);

      T_world_to_cam_reduced.matrix().block<2, 2>(0, 0) = R_world_to_cam_reduced.matrix();
      T_world_to_cam_reduced(0, 2) = T_world_to_cam.translation()(0);
      T_world_to_cam_reduced(1, 2) = T_world_to_cam.translation()(2);

      // Everything needed for a DLT trianguaiton.
      Eigen::Vector2d pixel_position(tracker_pt.x, 1);
      pixel_pos.push_back(pixel_position);
      transformation_mats.push_back(T_world_to_cam_reduced);
    }
  }
  // Use a Singular Value Decomposition (SVD) to perform the triangulation.
  // This is also known as a Direct Linear Transform (DLT).
  int num_rows = transformation_mats.size();

  // At least two observations are required for a triangulation.
  if (num_rows > 2) {
    // Assemble matrix A for DLT.
    Eigen::MatrixXd A;
    A.resize(num_rows, 3);
    for (int i = 0; i < num_rows; i++) {
      // Convert pixel frame to cam frame.
      double position = ((pixel_pos[i][0] - cam_config_.intrinsics[2]) / cam_config_.intrinsics[0]);
      A.row(i) = position * transformation_mats[i].matrix().row(1) -
                 transformation_mats[i].matrix().row(0);
    }
    // Singular Value decomposition.
    Eigen::BDCSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Get the last column.
    int n_ = svd.matrixV().cols() - 1;
    Eigen::Vector3d x = svd.matrixV().col(n_);
    // double minSV = svd.singularValues()(n_);

    // Filters meta variables
    bool acceptable = true;
    Eigen::Vector2d x_world(x(0) / x(2), x(1) / x(2));

    // Check for baseline
    double baseline_dist =
        (transformation_mats.front().translation() - transformation_mats.back().translation())
            .norm();
    acceptable = (baseline_dist > detector_config_.min_baseline_dist);

    for (int i = 0; i < num_rows; i++) {
      if (!acceptable) continue;

      // Init vars
      Eigen::Affine2d tf_mat = transformation_mats[i];
      double x_px = pixel_pos[i][0];

      // Compute x_cam
      Eigen::Vector2d x_cam = tf_mat * x_world;

      // Check if distance range is satisfied
      if (x_cam(1) >= detector_config_.dist_thresh_min &&
          x_cam(1) <= detector_config_.dist_thresh_max) {
        // If so, measure reprojection error
        double x_px_reproj =
            cam_config_.intrinsics[0] * (x_cam(0) / x_cam(1)) + cam_config_.intrinsics[2];
        acceptable = (std::abs(x_px_reproj - x_px) < detector_config_.max_reproj_err_px);
      } else {
        acceptable = false;
      }
    }

    if (acceptable) {
      // Normalize homogenous coordinates.
      new_pole.pos_x = x[0] / x[2];
      new_pole.pos_y = x[1] / x[2];

      // Store new map point in file.
      // if (!output_config_.map_file.empty() && map_file_.is_open()) {
      //   map_file_ << std::fixed << new_pole.ID << ","
      //             << "pole"
      //             << "," << new_pole.first_observed << "," << new_pole.pos_x << ","
      //             << new_pole.pos_y << ","
      //             << "0"
      //             << ","
      //             << "0" << std::endl;
      // }

      // && (new_pole.ID - 1) % 1 == 0
      if (output_config_.rviz) {
        const ros::Time ts_ = ros::Time();

        // Poles
        pole_marker_.header.stamp = ts_;
        pole_marker_.id = new_pole.ID;
        pole_marker_.pose.position.x = new_pole.pos_x;
        pole_marker_.pose.position.y = new_pole.pos_y;
        pole_viz_pub_.publish(pole_marker_);

        // Calculate cam position
        Eigen::Affine2d w_T_c = transformation_mats.front().inverse();

        Eigen::Vector2d w_t_c = w_T_c.translation();

        // Cam marker
        cam_marker_.header.stamp = ts_;
        cam_marker_.id = new_pole.ID;
        cam_marker_.color.b = 1;
        cam_marker_.color.g = 0;
        cam_marker_.pose.position.x = w_t_c(0);
        cam_marker_.pose.position.y = w_t_c(1);
        cam_viz_pub_.publish(cam_marker_);
      }

      // Increment TrackerPole Counter
      pole_count_ += 1;
    }
  }
}

// Event preprocessing prior to first HT.
void Detector::eventPreProcessing(const dvs_msgs::EventArray::ConstPtr &orig_msg,
                                  Eigen::MatrixXf &points) {
  int num_events = orig_msg->events.size();
  // Filtering for dead pixels and subsampling the leftover events.
  //
  // TODO: Could be parallelized by splitting into two steps, one that counts
  // second one that does the actual shuffle in memory, if done correctly
  // with some caching overhead would be very small.
  //
  // Also would be faster to just preallocate the eigen matrix to max
  // size and write directly into it and afterwards resize.
  for (int i = 0; i < num_events; i += detector_config_.evt_subsample_fac) {
    const dvs_msgs::Event &e = orig_msg->events[i];

    // Seemingly broken pixels in the DVS (millions of exactly equal events at
    // once at random). This needs to be adapted if you use another device.
    // if (((e.x != 19) || (e.y != 18)) && ((e.x != 43) || (e.y != 72)) &&
    //     ((e.x != 89) || (e.y != 52)) && ((e.x != 25) || (e.y != 42)) &&
    //     ((e.x != 61) || (e.y != 71)) && ((e.x != 37) || (e.y != 112))) {
    //   feature_msg_.events.push_back(e);
    // }

    feature_msg_.events.push_back(e);
  }

  // Number of events after filtering and subsampling.
  num_events = feature_msg_.events.size();

  // Check there are enough events for our window size. This is only relevant
  // during initialization.
  if (num_events <= hough1_config_.window_size) {
    return;
  }

  // Reshaping the event array into an Eigen matrix.
  points.resize(2, num_events);
  points.setZero();

  // Add points from the actual message.
  const auto ptr = points.data();
  CHECK_NOTNULL(ptr);

  if (cam_config_.perform_undist) {
#pragma omp parallel for
    for (int i = 0; i < num_events; i++) {
      const dvs_msgs::Event &event = feature_msg_.events[i];

      *(ptr + 2 * i) = cam_config_.undist_map_x(event.y, event.x);
      *(ptr + 2 * i + 1) = cam_config_.undist_map_y(event.y, event.x);
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

void Detector::visualizeTracker() {
  int num_cols = detector_config_.msg_per_window * detector_config_.tsteps_per_msg;
  cv::Mat line_space_neg(hough1_config_.radial_resolution, num_cols, CV_8UC1, 1);

#pragma omp parallel for
  for (int i = 0; i < houghout_queue_.size(); i++) {
    for (int j = 0; j < houghout_queue_[i].rows(); j++) {
      for (int k = 0; k < detector_config_.tsteps_per_msg; k++) {
        line_space_neg.at<uchar>(j, i * detector_config_.tsteps_per_msg + k, 0) =
            houghout_queue_[i](j, k);
      }
    }
  }

  cv::cvtColor(line_space_neg, line_space_neg, cv::COLOR_GRAY2BGR);

  for (int i = 0; i < cluster_centroids_.size(); i++) {
    for (int j = 0; j < cluster_centroids_[i].size(); j++) {
      cv::drawMarker(line_space_neg, cv::Point(i, cluster_centroids_[i][j]),
                     cv::Scalar(255, 0, 255), cv::MARKER_SQUARE, 0.5);
    }
  }

  // Clean up tracker viz
  while (viz_trackers_.size() >= 5) {
    viz_trackers_.pop_front();
  }

  // Draw Tracker Lines
  const double kColumnLengthInSec =
      (1.0 / cam_config_.evt_arr_frequency) / (detector_config_.tsteps_per_msg);

  for (auto &&tracker : viz_trackers_) {
    if (tracker.length() > 5) {
      auto tracked_points = tracker.getPoints();
      // for (int i = 0; i < 1; i++) {
      for (int i = 0; i < tracker.length() - 1; i++) {
        if (tracked_points[i + 1].t - tracked_points[i].t > kColumnLengthInSec) {
          int p1_t_index =
              num_cols - 1 -
              (int)((houghout_queue_last_t - tracked_points[i].t) / kColumnLengthInSec);
          int p2_t_index =
              num_cols - 1 -
              (int)((houghout_queue_last_t - tracked_points[i + 1].t) / kColumnLengthInSec);
          CHECK_LT(p1_t_index, p2_t_index);
          if (p1_t_index >= 0 and p2_t_index < num_cols) {
            // Plot only if lines visible in time buffer space
            cv::Point p1(p1_t_index, tracked_points[i].x);
            cv::Point p2(p2_t_index, tracked_points[i + 1].x);
            cv::line(line_space_neg, p1, p2, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
          }
        };
      }
    }
  }

  // Flip image for nicer viewing.
  cv::flip(line_space_neg, line_space_neg, 0);
  cv::rotate(line_space_neg, line_space_neg, cv::ROTATE_90_CLOCKWISE);

  xt_img_pub_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", line_space_neg).toImageMsg());
}

void Detector::drawPolarCorLine(cv::Mat &image_space, float rho, float theta, cv::Scalar color) {
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

void Detector::addMaximaInRadius(int i, int radius, const Eigen::MatrixXi &total_hough_space,
                                 int local_threshold, double timestamp,
                                 std::vector<hough2map::HoughLine> *new_maxima,
                                 std::vector<int> *new_maxima_value, bool skip_center) {
  int m_l = std::max(i - hough1_config_.nms_radius, 0);
  int m_r = std::min(i + hough1_config_.nms_radius + 1, hough1_config_.angular_resolution);
  int n_l = std::max(radius - hough1_config_.nms_radius, 0);
  int n_r = std::min(radius + hough1_config_.nms_radius + 1, hough1_config_.radial_resolution);

  for (int m = m_l; m < m_r; m++) {
    for (int n = n_l; n < n_r; n++) {
      if (skip_center && (n == radius) && (m == i)) {
        continue;
      }

      if ((total_hough_space(n, m) > local_threshold) && isLocalMaxima(total_hough_space, m, n)) {
        new_maxima->push_back(addMaxima(m, n, timestamp));
        new_maxima_value->push_back(total_hough_space(n, m));
      }
    }
  }
}

void Detector::applySuppressionRadius(const std::vector<hough2map::HoughLine> &new_maxima,
                                      const std::vector<int> &new_maxima_value,
                                      std::vector<hough2map::HoughLine> *current_maxima) {
  // Create an index of all known maxima.
  std::vector<int> new_maxima_index(new_maxima_value.size());
  for (int i = 0; i < new_maxima_value.size(); i++) {
    new_maxima_index[i] = i;
  }

  // Sort the index of all currently known maxima. Sort them by: 1. value; 2.
  // rho value; 3. theta value.
  std::stable_sort(new_maxima_index.begin(), new_maxima_index.end(),
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
    const hough2map::HoughLine &new_maximum = new_maxima[new_maxima_index[i]];

    bool add_maximum = true;
    // Compare to all other maxima in the output buffer.
    for (int j = 0; j < current_maxima->size(); j++) {
      const hough2map::HoughLine &cur_maximum = (*current_maxima)[j];

      // If no maximum in the output buffer is of the same polarity and within
      // the radius, the current maximum is kept and added to the output
      // buffer.
      // Suppression radius.
      float distance = (cur_maximum.r - new_maximum.r) * (cur_maximum.r - new_maximum.r) +
                       (cur_maximum.theta_idx - new_maximum.theta_idx) *
                           (cur_maximum.theta_idx - new_maximum.theta_idx);

      if (distance < hough1_config_.nms_radius * hough1_config_.nms_radius) {
        add_maximum = false;
        break;
      }
    }

    // Adding accepted maxima to the output buffer.
    if (add_maximum) {
      current_maxima->push_back(new_maximum);
    }
  }
}

// Check if the center value is a maxima.
bool Detector::isLocalMaxima(const Eigen::MatrixXi &hough_space, int i, int radius) {
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

std::vector<int> Detector::getClusteringCentroids(Eigen::VectorXi detections) {
  std::vector<int> cluster_centroids;
  std::deque<int> last_n_centroids;

  int weighted_sum = 0;
  int value_sum = 0;

  for (int i = 0; i < detections.size(); i++) {
    value_sum += detections(i);
    weighted_sum += detections(i) * i;
    if (value_sum >= detector_config_.centroid_find_thresh) {
      // Calculate current centroid and append to last_n_centroids
      int current_centroid = (int)(weighted_sum / value_sum);
      last_n_centroids.push_back(current_centroid);
      if (last_n_centroids.size() > detector_config_.centroid_find_window) {
        last_n_centroids.pop_front();
      }
      // Check if the centroid has not moved
      if (value_sum >= detector_config_.centroid_find_thresh &&
          last_n_centroids.size() == detector_config_.centroid_find_window &&
          (last_n_centroids.back() - last_n_centroids.front()) <= 1) {
        // If so, add a cluster centroid
        int add_centroid = last_n_centroids.front();
        CHECK_GE(add_centroid, 0);
        CHECK_LT(add_centroid, detections.size());
        cluster_centroids.push_back(add_centroid);
        // Also reset the accumulators
        weighted_sum = 0;
        value_sum = 0;
      }
    }
  }

  return cluster_centroids;
}

// Just a funciton for creating new line structs.
HoughLine Detector::addMaxima(int angle, int rad, double cur_time) {
  hough2map::HoughLine new_line;

  new_line.ID = 0;
  new_line.r = rad;
  new_line.theta = thetas_1_(angle);
  new_line.theta_idx = angle;
  new_line.time = cur_time;

  return new_line;
}

geometry_msgs::PoseWithCovarianceStamped Detector::queryPoseAtTime(const double query_time) {
  auto lower_it =
      std::upper_bound(
          pose_buffer_.begin(), pose_buffer_.end(), query_time,
          [](double lhs, geometry_msgs::PoseWithCovarianceStampedConstPtr &rhs) -> bool {
            return lhs < rhs->header.stamp.toSec();
          }) -
      1;

  auto upper_it =
      std::lower_bound(pose_buffer_.begin(), pose_buffer_.end(), query_time,
                       [](geometry_msgs::PoseWithCovarianceStampedConstPtr &lhs,
                          double rhs) -> bool { return lhs->header.stamp.toSec() < rhs; });

  geometry_msgs::PoseWithCovarianceStamped interpolatedPose;

  // Get interpolation factor (f)a + (1-f)b
  const double kInterpFactor =
      std::abs(query_time - (*lower_it)->header.stamp.toSec()) /
      std::abs((*upper_it)->header.stamp.toSec() - (*lower_it)->header.stamp.toSec());

  // Interpolate timestamp
  interpolatedPose.header.stamp.fromSec(query_time);
  interpolatedPose.header.frame_id = (*lower_it)->header.frame_id;

  // Interpolate position
  interpolatedPose.pose.pose.position.x = (*lower_it)->pose.pose.position.x * kInterpFactor +
                                          (*lower_it)->pose.pose.position.x * (1 - kInterpFactor);
  interpolatedPose.pose.pose.position.y = (*lower_it)->pose.pose.position.y * kInterpFactor +
                                          (*lower_it)->pose.pose.position.y * (1 - kInterpFactor);
  interpolatedPose.pose.pose.position.z = (*lower_it)->pose.pose.position.z * kInterpFactor +
                                          (*lower_it)->pose.pose.position.z * (1 - kInterpFactor);

  // Interpolate quaternion SLERP
  tf2::Quaternion qLower;
  tf2::Quaternion qUpper;
  tf2::fromMsg((*lower_it)->pose.pose.orientation, qLower);
  tf2::fromMsg((*upper_it)->pose.pose.orientation, qUpper);
  interpolatedPose.pose.pose.orientation = tf2::toMsg(tf2::slerp(qLower, qUpper, kInterpFactor));

  return interpolatedPose;
}

}  // namespace hough2map
