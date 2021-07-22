#!/usr/bin/env python

import rospy
from contextlib import contextmanager
from std_srvs.srv import Empty, Trigger
from ur_dashboard_msgs.srv import Load
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
import numpy as np
from arm_utils.srv import ExecuteTrajectory, HandleJointPlan, HandlePosePlan
from sensor_msgs.msg import JointState
import moveit_commander


def create_joint_mapping(source_names, target_names):
    source_names = list(source_names)
    idx = [source_names.index(target_name) for target_name in target_names]
    return np.array(idx)

def handle_joint_plan(req):
    joint = req.joint_state
    joint_vals = np.array(joint.position)
    if joint.name and joint.name[0]:
        idx_mapping = create_joint_mapping(joint.name, move_group.get_active_joints())
        joint_vals = joint_vals[idx_mapping]

    planned_traj = move_group.plan(joint_vals).joint_trajectory
    if req.execute:
        execute_traj_srv(planned_traj)

    return planned_traj

if __name__ == '__main__':
    rospy.init_node('ur_waypoint_handler')

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group_name = rospy.get_param('move_group_name', 'manipulator')
    move_group = moveit_commander.MoveGroupCommander(group_name)

    execute_traj_srv = rospy.ServiceProxy('execute_trajectory', ExecuteTrajectory)
    rospy.Service('plan_joints', HandleJointPlan, handle_joint_plan)

    rospy.spin()

