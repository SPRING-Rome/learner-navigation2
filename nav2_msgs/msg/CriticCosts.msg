# Cost added by one critic to every sampled trajectory

# Critic name, as listed in the controller "critics" parameter
string name

# Cost added by this critic to each trajectory (size: CriticsDebug.batch_size)
float32[] costs

# Optional text of this critic for each trajectory, explaining its cost (e.g. the worst case it
# found or why the trajectory is lethal). Empty if the critic gives none, otherwise one string per
# trajectory (size: CriticsDebug.batch_size), empty for the trajectories without anything to report
string[] debug_info

# Time spent in the critic score() function, in milliseconds
float32 duration_ms
