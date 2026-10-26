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

#include "nav2_mppi_controller/critic_manager.hpp"

namespace mppi
{

void CriticManager::on_configure(
  rclcpp_lifecycle::LifecycleNode::WeakPtr parent, const std::string & name,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros, ParametersHandler * param_handler)
{
  parent_ = parent;
  costmap_ros_ = costmap_ros;
  name_ = name;
  auto node = parent_.lock();
  logger_ = node->get_logger();
  parameters_handler_ = param_handler;

  getParams();
  loadCritics();
}

void CriticManager::getParams()
{
  auto node = parent_.lock();
  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(critic_names_, "critics", std::vector<std::string>{}, ParameterType::Static);
  getParam(publish_critics_stats_, "publish_critics_stats", false, ParameterType::Static);
}

void CriticManager::loadCritics()
{
  if (!loader_) {
    loader_ = std::make_unique<pluginlib::ClassLoader<critics::CriticFunction>>(
      "nav2_mppi_controller", "mppi::critics::CriticFunction");
  }

  auto node = parent_.lock();
  if (publish_critics_stats_) {
    critics_effect_pub_ = node->create_publisher<nav2_msgs::msg::CriticsStats>(
    "~/critics_stats", 10);
    critics_effect_pub_->on_activate();
  }

  critics_.clear();
  for (auto name : critic_names_) {
    std::string fullname = getFullName(name);
    auto instance = std::unique_ptr<critics::CriticFunction>(
      loader_->createUnmanagedInstance(fullname));
    critics_.push_back(std::move(instance));
    critics_.back()->on_configure(
      parent_, name_, name_ + "." + name, costmap_ros_,
      parameters_handler_);
    RCLCPP_INFO(logger_, "Critic loaded : %s", fullname.c_str());
  }
}

std::string CriticManager::getFullName(const std::string & name)
{
  return "mppi::critics::" + name;
}

void CriticManager::evalTrajectoriesScores(
  CriticData & data)
{
  if (collect_critic_info_) {
    critic_costs_.clear();
    critic_costs_.reserve(critics_.size());
    critic_durations_ms_.clear();
    critic_durations_ms_.reserve(critics_.size());
    critic_debug_info_.clear();
    critic_debug_info_.reserve(critics_.size());
  }
  data.collect_debug_info = collect_critic_info_;

  // A flag left over from a failed attempt means no critic runs: keep the attribution
  if (!data.fail_flag) {
    failed_critic_.clear();
    fail_reason_.clear();
    data.fail_reason.clear();
  }

  std::unique_ptr<nav2_msgs::msg::CriticsStats> stats_msg;
  if (publish_critics_stats_) {
    stats_msg = std::make_unique<nav2_msgs::msg::CriticsStats>();
    stats_msg->critics.reserve(critics_.size());
    stats_msg->changed.reserve(critics_.size());
    stats_msg->costs_sum.reserve(critics_.size());
    stats_msg->costs_normalized.reserve(critics_.size());
  }

  for (size_t i = 0; i < critics_.size(); ++i) {
    if (data.fail_flag) {
      break;
    }

    // Store costs before critic evaluation
    Eigen::ArrayXf costs_before;
    if (publish_critics_stats_ || collect_critic_info_) {
      costs_before = data.costs;
    }

    data.debug_info.clear();
    const auto start = std::chrono::steady_clock::now();
    critics_[i]->score(data);
    const auto end = std::chrono::steady_clock::now();

    if (data.fail_flag) {
      failed_critic_ = critic_names_[i];
      fail_reason_ = data.fail_reason;
    }

    if (!(publish_critics_stats_ || collect_critic_info_)) {
      continue;
    }

    // Costs added by this individual critic
    Eigen::ArrayXf cost_diff = data.costs - costs_before;

    // Calculate statistics if publishing is enabled
    if (publish_critics_stats_) {
      stats_msg->critics.push_back(critic_names_[i]);

      float costs_sum = cost_diff.sum();
      stats_msg->costs_sum.push_back(costs_sum);
      stats_msg->changed.push_back(costs_sum != 0.0f);
      
      // Calculate normalized cost contribution for this critic
      // Normalize the cost_diff array and sum it
      if (cost_diff.size() > 0) {
        auto cost_diff_normalized = cost_diff - cost_diff.minCoeff();
        float normalized_sum = cost_diff_normalized.sum();
        stats_msg->costs_normalized.push_back(normalized_sum);
      } else {
        stats_msg->costs_normalized.push_back(0.0f);
      }
    }

    if (collect_critic_info_) {
      critic_costs_.emplace_back(critic_names_[i], std::move(cost_diff));
      critic_durations_ms_.push_back(
        std::chrono::duration<float, std::milli>(end - start).count());
      critic_debug_info_.push_back(std::move(data.debug_info));
      data.debug_info.clear();
    }
  }

  // Publish statistics if enabled
  if (critics_effect_pub_) {
    auto node = parent_.lock();
    stats_msg->stamp = node->get_clock()->now();
    critics_effect_pub_->publish(std::move(stats_msg));
  }
}

}  // namespace mppi
