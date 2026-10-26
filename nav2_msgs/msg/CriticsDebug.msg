# Debug information of one attempt of an MPPI optimization cycle (last iteration), not meant for RViz.
# Contains the sampled trajectories and, for each critic, the cost of every trajectory.
# Only published while it has subscribers. Failed attempts are published too, before the
# optimizer is reset.

uint8 RESULT_SUCCESS=0
uint8 RESULT_RETRY=1             # An attempt failed (fail flag set by a critic), a retry follows
uint8 RESULT_NO_VALID_CONTROL=2  # An attempt failed with no retry left: NoValidControl is thrown

# stamp: stamp of the velocity command; frame_id: frame of the trajectories and poses
std_msgs/Header header

# Control cycle (one per computeVelocityCommands call, starting at 1) and attempt in that cycle:
# 0 for the first one, 1 for the first retry, ...
uint32 cycle_id
uint32 retry
uint8 result

# Critic that set the fail flag and the reason it gave (if any). Empty on success. On a retry
# attempt no critic is evaluated and the failure of the previous attempt is reported.
string failed_critic
string fail_reason

# Number of sampled trajectories
uint32 batch_size

# Number of points of each trajectory
uint32 time_steps

# Time between two trajectory points, in seconds
float32 model_dt

# Per-critic costs, in evaluation order. Critics skipped after a failure are absent.
CriticCosts[] critics

# Cost added by the optimizer itself (control sequence regularization), one per trajectory
float32[] control_costs

# Final cost of each trajectory: sum of all critics and control_costs (size: batch_size)
float32[] total_costs

# Softmax weight of each trajectory in the control update, sums to 1 (size: batch_size)
float32[] weights

# Sampled trajectories, row-major [batch_size x time_steps]: element (i, j) is at i * time_steps + j
float32[] x
float32[] y
float32[] yaw

# Robot state and goal used in this cycle
geometry_msgs/Pose robot_pose
geometry_msgs/Twist robot_speed
geometry_msgs/Pose goal
