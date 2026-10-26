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

#ifndef NAV2_MPPI_CONTROLLER__CRITIC_MANAGER_HPP_
#define NAV2_MPPI_CONTROLLER__CRITIC_MANAGER_HPP_

#include <Eigen/Dense>

#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <pluginlib/class_loader.hpp>

#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav2_msgs/msg/critics_stats.hpp"

#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"

#include "nav2_mppi_controller/tools/parameters_handler.hpp"
#include "nav2_mppi_controller/tools/utils.hpp"
#include "nav2_mppi_controller/critic_data.hpp"
#include "nav2_mppi_controller/critic_function.hpp"

namespace mppi
{

/**
 * @class mppi::CriticManager
 * @brief Manager of objective function plugins for scoring trajectories
 */
class CriticManager
{
public:
  typedef std::vector<std::unique_ptr<critics::CriticFunction>> Critics;
  /**
    * @brief Constructor for mppi::CriticManager
    */
  CriticManager() = default;


  /**
    * @brief Virtual Destructor for mppi::CriticManager
    */
  virtual ~CriticManager() = default;

  /**
    * @brief Configure critic manager on bringup and load plugins
    * @param parent WeakPtr to node
    * @param name Name of plugin
    * @param costmap_ros Costmap2DROS object of environment
    * @param dynamic_parameter_handler Parameter handler object
    */
  void on_configure(
    rclcpp_lifecycle::LifecycleNode::WeakPtr parent, const std::string & name,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS>, ParametersHandler *);

  /**
    * @brief Score trajectories by the set of loaded critic functions
    * @param CriticData Struct of necessary information to pass to the critic functions
    */
  void evalTrajectoriesScores(CriticData & data);

  /**
    * @brief Enable or disable storing per-critic costs and run times (visualization, debug topic)
    * @param collect Whether to collect per-critic information
    */
  void setCollectCriticInfo(bool collect) {collect_critic_info_ = collect;}

  /**
    * @brief Get stored per-critic costs from last evaluation (filled when collection is enabled)
    * @return Vector of (critic_name, cost_array) pairs
    */
  const std::vector<std::pair<std::string, Eigen::ArrayXf>> & getCriticCosts() const
  {
    return critic_costs_;
  }

  /**
    * @brief Get per-trajectory debug text of each critic from the last evaluation (filled when
    * collection is enabled)
    * @return One vector per critic, same order as getCriticCosts(): empty if the critic gave no
    * text, otherwise one string per trajectory
    */
  const std::vector<std::vector<std::string>> & getCriticDebugInfo() const
  {
    return critic_debug_info_;
  }

  /**
    * @brief Get per-critic run times of the last evaluation (filled when collection is enabled)
    * @return Run time of each critic score() call in milliseconds, same order as getCriticCosts()
    */
  const std::vector<float> & getCriticDurationsMs() const {return critic_durations_ms_;}

  /**
    * @brief Get the name of the critic that set the fail flag in the last evaluation
    * @return Critic name, empty if no critic failed
    */
  const std::string & getFailedCritic() const {return failed_critic_;}

  /**
    * @brief Get the reason given by the critic that set the fail flag
    * @return Reason, empty if no critic failed or the critic gave none
    */
  const std::string & getFailReason() const {return fail_reason_;}

protected:
  /**
    * @brief Get parameters (critics to load)
    */
  void getParams();

  /**
    * @brief Load the critic plugins
    */
  virtual void loadCritics();

  /**
    * @brief Get full-name namespaced critic IDs
    */
  std::string getFullName(const std::string & name);

protected:
  rclcpp_lifecycle::LifecycleNode::WeakPtr parent_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  std::string name_;

  ParametersHandler * parameters_handler_;
  std::vector<std::string> critic_names_;
  std::unique_ptr<pluginlib::ClassLoader<critics::CriticFunction>> loader_;
  Critics critics_;

  rclcpp_lifecycle::LifecyclePublisher<nav2_msgs::msg::CriticsStats>::SharedPtr
    critics_effect_pub_;

  bool publish_critics_stats_;
  bool collect_critic_info_{false};
  std::vector<std::pair<std::string, Eigen::ArrayXf>> critic_costs_;
  std::vector<std::vector<std::string>> critic_debug_info_;
  std::vector<float> critic_durations_ms_;
  std::string failed_critic_;
  std::string fail_reason_;

  rclcpp::Logger logger_{rclcpp::get_logger("MPPIController")};
};

}  // namespace mppi

#endif  // NAV2_MPPI_CONTROLLER__CRITIC_MANAGER_HPP_
