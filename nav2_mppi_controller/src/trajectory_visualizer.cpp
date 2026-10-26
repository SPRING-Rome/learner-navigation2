// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "nav2_mppi_controller/tools/trajectory_visualizer.hpp"

namespace mppi
{

void TrajectoryVisualizer::on_configure(
  rclcpp_lifecycle::LifecycleNode::WeakPtr parent, const std::string & name,
  const std::string & frame_id, ParametersHandler * parameters_handler)
{
  auto node = parent.lock();
  logger_ = node->get_logger();
  frame_id_ = frame_id;
  trajectories_publisher_ =
    node->create_publisher<visualization_msgs::msg::MarkerArray>("~/candidate_trajectories", 1);
  transformed_path_pub_ = node->create_publisher<nav_msgs::msg::Path>(
    "~/transformed_global_plan", 1);
  optimal_path_pub_ = node->create_publisher<nav_msgs::msg::Path>("~/optimal_path", 1);
  parameters_handler_ = parameters_handler;

  auto getParam = parameters_handler->getParamGetter(name + ".TrajectoryVisualizer");

  getParam(trajectory_step_, "trajectory_step", 5);
  getParam(time_step_, "time_step", 3);
  getParam(lethal_cost_, "lethal_cost", 10000.0f);

  reset();
}

void TrajectoryVisualizer::on_cleanup()
{
  trajectories_publisher_.reset();
  transformed_path_pub_.reset();
  optimal_path_pub_.reset();
}

void TrajectoryVisualizer::on_activate()
{
  trajectories_publisher_->on_activate();
  transformed_path_pub_->on_activate();
  optimal_path_pub_->on_activate();
}

void TrajectoryVisualizer::on_deactivate()
{
  trajectories_publisher_->on_deactivate();
  transformed_path_pub_->on_deactivate();
  optimal_path_pub_->on_deactivate();
}

void TrajectoryVisualizer::add(
  const Eigen::ArrayXXf & trajectory,
  const std::string & marker_namespace,
  const builtin_interfaces::msg::Time & cmd_stamp)
{
  size_t size = trajectory.rows();
  if (!size) {
    return;
  }

  auto add_marker = [&](auto i) {
      float component = static_cast<float>(i) / static_cast<float>(size);

      auto pose = utils::createPose(trajectory(i, 0), trajectory(i, 1), 0.06);
      auto scale =
        i != size - 1 ?
        utils::createScale(0.03, 0.03, 0.07) :
        utils::createScale(0.07, 0.07, 0.09);
      auto color = utils::createColor(0, component, component, 1);
      auto marker = utils::createMarker(
        marker_id_++, pose, scale, color, frame_id_, marker_namespace);
      points_->markers.push_back(marker);

      // populate optimal path
      geometry_msgs::msg::PoseStamped pose_stamped;
      pose_stamped.header.frame_id = frame_id_;
      pose_stamped.pose = pose;

      tf2::Quaternion quaternion_tf2;
      quaternion_tf2.setRPY(0., 0., trajectory(i, 2));
      pose_stamped.pose.orientation = tf2::toMsg(quaternion_tf2);

      optimal_path_->poses.push_back(pose_stamped);
    };

  optimal_path_->header.stamp = cmd_stamp;
  optimal_path_->header.frame_id = frame_id_;
  for (size_t i = 0; i < size; i++) {
    add_marker(i);
  }
}

void TrajectoryVisualizer::add(
  const models::Trajectories & trajectories,
  const Eigen::ArrayXf & costs,
  const builtin_interfaces::msg::Time & stamp)
{
  if (trajectories_publisher_->get_subscription_count() == 0) {
    return;
  }

  // Clear markers of previous cycles, e.g. critics that were not evaluated this time
  visualization_msgs::msg::Marker clear_marker;
  clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
  points_->markers.push_back(clear_marker);

  addCostColoredTrajectories(trajectories, costs, "Total Cost", stamp);
}

void TrajectoryVisualizer::addCriticCosts(
  const models::Trajectories & trajectories,
  const std::vector<std::pair<std::string, Eigen::ArrayXf>> & critic_costs,
  const builtin_interfaces::msg::Time & stamp)
{
  if (trajectories_publisher_->get_subscription_count() == 0) {
    return;
  }

  for (const auto & [name, costs] : critic_costs) {
    addCostColoredTrajectories(trajectories, costs, name, stamp);
  }
}

void TrajectoryVisualizer::addCostColoredTrajectories(
  const models::Trajectories & trajectories,
  const Eigen::ArrayXf & costs,
  const std::string & marker_namespace,
  const builtin_interfaces::msg::Time & stamp)
{
  using visualization_msgs::msg::Marker;
  const size_t n_rows = trajectories.x.rows();
  const size_t n_cols = trajectories.x.cols();
  if (n_rows == 0 || static_cast<size_t>(costs.size()) < n_rows) {
    return;
  }

  // Per-critic costs are float differences (after - before), so allow for rounding
  const float lethal_tolerance = 0.05f;

  // Normalize over non-lethal trajectories only, so lethal costs don't wash out the gradient
  float min_val = std::numeric_limits<float>::max();
  float max_val = std::numeric_limits<float>::lowest();
  for (size_t k = 0; k < n_rows; ++k) {
    if (costs(k) < lethal_cost_ - lethal_tolerance) {
      min_val = std::min(min_val, costs(k));
      max_val = std::max(max_val, costs(k));
    }
  }
  const float range = max_val - min_val;

  points_->markers.reserve(points_->markers.size() + n_rows / trajectory_step_ + 1);
  for (size_t i = 0; i < n_rows; i += trajectory_step_) {
    Marker marker;
    marker.header.frame_id = frame_id_;
    marker.header.stamp = stamp;
    marker.ns = marker_namespace;
    marker.id = marker_id_++;
    marker.type = Marker::LINE_STRIP;
    marker.action = Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.01;  // line width
    if (std::abs(costs(i) - lethal_cost_) <= lethal_tolerance) {
      // Marked lethal by a critic
      marker.color = utils::createColor(0.0f, 0.0f, 0.0f, 1.0f);
    } else if (costs(i) > lethal_cost_) {
      // Lethal by accumulated cost: grey (non-zero blue distinguishes it from the black/colormap)
      marker.color = utils::createColor(0.5f, 0.5f, 0.5f, 1.0f);
    } else {
      marker.color = costToColor(range > 0.0f ? (costs(i) - min_val) / range : 0.0f);
    }

    marker.points.reserve(n_cols / time_step_ + 1);
    for (size_t j = 0; j < n_cols; j += time_step_) {
      geometry_msgs::msg::Point pt;
      pt.x = trajectories.x(i, j);
      pt.y = trajectories.y(i, j);
      pt.z = 0.03;
      marker.points.push_back(pt);
    }

    points_->markers.push_back(std::move(marker));
  }
}

std_msgs::msg::ColorRGBA TrajectoryVisualizer::costToColor(float normalized)
{
  // Green (0) -> Yellow (0.5) -> Red (1.0)
  normalized = std::clamp(normalized, 0.0f, 1.0f);
  float r, g;
  if (normalized < 0.5f) {
    r = 2.0f * normalized;
    g = 1.0f;
  } else {
    r = 1.0f;
    g = 2.0f * (1.0f - normalized);
  }
  return utils::createColor(r, g, 0.0f, 0.8f);
}

void TrajectoryVisualizer::reset()
{
  marker_id_ = 0;
  points_ = std::make_unique<visualization_msgs::msg::MarkerArray>();
  optimal_path_ = std::make_unique<nav_msgs::msg::Path>();
}

void TrajectoryVisualizer::visualize(const nav_msgs::msg::Path & plan)
{
  if (trajectories_publisher_->get_subscription_count() > 0) {
    trajectories_publisher_->publish(std::move(points_));
  }

  if (optimal_path_pub_->get_subscription_count() > 0) {
    optimal_path_pub_->publish(std::move(optimal_path_));
  }

  reset();

  if (transformed_path_pub_->get_subscription_count() > 0) {
    auto plan_ptr = std::make_unique<nav_msgs::msg::Path>(plan);
    transformed_path_pub_->publish(std::move(plan_ptr));
  }
}

}  // namespace mppi
