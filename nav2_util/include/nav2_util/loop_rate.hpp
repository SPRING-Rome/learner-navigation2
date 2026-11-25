// Copyright (c) 2025 Learner
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

#ifndef NAV2_UTIL__LOOP_RATE_HPP_
#define NAV2_UTIL__LOOP_RATE_HPP_

#include <chrono>
#include <memory>

#include "rclcpp/clock.hpp"
#include "rclcpp/duration.hpp"
#include "rclcpp/rate.hpp"
#include "rclcpp/time.hpp"

namespace nav2_util
{

/**
 * @class nav2_util::LoopRate
 * @brief Loop rate that follows the simulation clock when the node uses sim time.
 *
 * Without use_sim_time it behaves exactly like rclcpp::WallRate. With use_sim_time the period
 * is measured on the node clock (/clock), so a simulation running faster or slower than real time
 * still gives the configured number of iterations per simulated second.
 */
class LoopRate
{
public:
  LoopRate(double frequency, const rclcpp::Clock::SharedPtr & clock)
  : clock_(clock),
    sim_time_(clock->get_clock_type() == RCL_ROS_TIME && clock->ros_time_is_active()),
    period_(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(1.0 / frequency)))
  {
    if (sim_time_) {
      last_ = clock_->now();
    } else {
      wall_ = std::make_unique<rclcpp::WallRate>(frequency);
    }
  }

  bool is_sim_time() const {return sim_time_;}

  /// @brief Sleep until the next period. Returns false if the period was already over.
  bool sleep()
  {
    if (!sim_time_) {
      return wall_->sleep();
    }
    const rclcpp::Duration period(period_);
    auto now = clock_->now();
    if (now < last_) {
      last_ = now;  // simulation time jumped backwards
    }
    auto next = last_ + period;
    last_ = next;
    if (now >= next) {
      if (now > next + period) {
        last_ = now + period;  // missed more than a whole cycle
      }
      return false;
    }
    clock_->sleep_until(next);
    return true;
  }

private:
  rclcpp::Clock::SharedPtr clock_;
  bool sim_time_;
  std::chrono::nanoseconds period_;
  rclcpp::Time last_{0, 0, RCL_ROS_TIME};
  std::unique_ptr<rclcpp::WallRate> wall_;
};

}  // namespace nav2_util

#endif  // NAV2_UTIL__LOOP_RATE_HPP_
