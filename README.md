# ur-arm-utils

Some simple ROS utils for getting the UR5 to move. The main modes of movement are either pose planning, tool-frame velocity commands, or prerecorded trajectories.

## Pose planning

Pose planning is taken care of in scripts/ur_waypoint_handler.py. To plan a pose, one can either pass in a set of joints to go to, or a desired pose. Note that if you pass in a pose, you are not guaranteed a given joint configuration, which means that you may end up in a configuration which has low manipulability. For this reason, it is generally better to pass in joints when possible. 

The relevant ROS services to call are plan_joints and plan_pose. Each takes in the corresponding target message (as a JointState for joints or as a PoseStamped for pose) as well as a Boolean flag for whether or not you want to actually execute the planned trajectory. The services return the planned trajectory, as well as a status message for whether planning/execution succeeded or not.

## Tool-frame velocity commands

Sometimes, it is desirable to move the end effector with a desired velocity without having to resort to planning to individual waypoints, which can be slow and result in jerky movements. To do this, you should send a velocity command as a Vector3Stamped. The script which handles velocity commands is ur_servoing.py.

1. First, you must call the /servo_activate service to enable velocity commands to be interpreted. Otherwise the node will simply ignore all sent velocity commands. 
2. Next, you should publish your velocity commands to the vel_command topic, which will send the desired velocities to the robot.
3. When done, you should call the /servo_stop command to disable the system from reading velocity commands.
4. Sometimes, it is useful to rewind the robot to where it was before it started servoing. For this, you can call the /servo_rewind service after calling /servo_stop. This will play the joint states recorded during the movement backwards. 

## Prerecorded trajectories

If you need the robot to do a preplanned movement, it is sometimes best to record a trajectory and play it back. The script that handles playing back trajectories is ur_trajectory_handler.py. Simply call the /execute_trajectory service with your JointTrajectory message.
