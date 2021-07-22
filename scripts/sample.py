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
from tf2_geometry_msgs import do_transform_pose, do_transform_point
from geometry_msgs.msg import PoseStamped, Quaternion, Point, PointStamped, Vector3Stamped, Vector3, TransformStamped
from copy import deepcopy

def tf_to_pose(tf):
    frame_id = tf.header.frame_id
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    tr = tf.transform.translation
    rt = tf.transform.rotation
    pose.pose.position = Point(tr.x, tr.y, tr.z)
    pose.pose.orientation = Quaternion(rt.x, rt.y, rt.z, rt.w)

    return pose

if __name__ == '__main__':
    # Be sure to launch the base utils and the appropriate UR moveit package
    # roslaunch ur_robot_driver ur5e_bringup.launch robot_ip:=YOUR_IP
    # roslaunch ur5_e_moveit_config move_group.launch
    # roslaunch arm_utils launch_base.launch

    rospy.init_node('sample_script')

    # Initialize connections to required services/topics
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer)
    plan_pose_srv = rospy.ServiceProxy('/plan_pose', HandlePosePlan)
    plan_joints_srv = rospy.ServiceProxy('/plan_joints', HandleJointPlan)
    servo_activate = rospy.ServiceProxy('/servo_activate', Empty)
    servo_stop = rospy.ServiceProxy('/servo_stop', Empty)
    servo_rewind = rospy.ServiceProxy('/servo_rewind', Empty)
    vel_pub = rospy.Publisher('/vel_command', Vector3Stamped)

    rospy.sleep(0.5)

    base_frame = 'base_link'
    tool_frame = 'ee_link'

    # Retrieve the starting pose of the end effector

    t = rospy.Time.now()
    success = tf_buffer.can_transform(base_frame, tool_frame, t, timeout=rospy.Duration(0.5))
    if not success:
        raise Exception("Failed to retrieve TF!")
    tf = tf_buffer.lookup_transform(base_frame, tool_frame, t)
    start_pose = tf_to_pose(tf)
    joints_start = rospy.wait_for_message('/joint_states', JointState, timeout=0.5)

    # Move the arm around close to the starting position and perform forward servoing
    N_SAMPLES = 3
    RADIUS = 0.05
    angles = np.linspace(0, 2*np.pi, num=N_SAMPLES, endpoint=False)
    for i, angle in enumerate(angles, start=1):
        print('Moving to target {}'.format(i))
        pt_tool = PointStamped()
        pt_tool.header.frame_id = tool_frame
        pt_tool.point = Point(0.0, RADIUS * np.cos(angle), RADIUS * np.sin(angle))
        pt_base = do_transform_point(pt_tool, tf)
        target_pose = deepcopy(start_pose)
        target_pose.pose.position = pt_base.point

        response = plan_pose_srv(target_pose, True)
        if not response.success:
            rospy.logwarn('Planning to target {} failed, moving to next target'.format(i))
            continue

        servo_activate()
        rate = rospy.Rate(10)
        start = rospy.Time.now()
        rospy.loginfo('Servoing...')
        while (rospy.Time.now() - start).to_sec() < 2.0:
            vel_msg = Vector3Stamped()
            vel_msg.header.stamp = rospy.Time.now()
            vel_msg.header.frame_id = tool_frame
            vel_msg.vector = Vector3(0.03, 0.0, 0.0)
            vel_pub.publish(vel_msg)
            rate.sleep()
        servo_stop()
        rospy.loginfo('Done servoing!')
        raw_input('Press Enter when ready to rewind arm...')
        servo_rewind()

    # Move back to start
    print('Moving back to start!')
    plan_joints_srv(joints_start, True)

