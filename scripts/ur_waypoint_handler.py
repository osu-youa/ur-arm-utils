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
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_pose

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
    success = bool(planned_traj.points)
    if success and req.execute:
        execute_traj_srv(planned_traj)

    return planned_traj, success

def handle_pose_plan(req):

    pose = req.pose
    stamp = pose.header.stamp

    base_frame = move_group.get_planning_frame().lstrip('/')
    success = tf_buffer.can_transform(base_frame, pose.header.frame_id, stamp, timeout=rospy.Duration(0.5))
    if not success:
        raise Exception("Failed to retrieve TF!")
    tf = tf_buffer.lookup_transform(base_frame, pose.header.frame_id, stamp)
    pose_base = do_transform_pose(pose, tf).pose

    move_group.set_pose_target(pose_base)
    planned_traj = move_group.plan().joint_trajectory
    success = bool(planned_traj.points)
    if success and req.execute:
        execute_traj_srv(planned_traj)

    return planned_traj, success


if __name__ == '__main__':
    rospy.init_node('ur_waypoint_handler')

    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer)

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group_name = rospy.get_param('move_group_name', 'manipulator')
    move_group = moveit_commander.MoveGroupCommander(group_name)

    execute_traj_srv = rospy.ServiceProxy('execute_trajectory', ExecuteTrajectory)
    rospy.Service('plan_joints', HandleJointPlan, handle_joint_plan)
    rospy.Service('plan_pose', HandlePosePlan, handle_pose_plan)

    rospy.spin()

