#!/usr/bin/env python

import rospy
import sys
from geometry_msgs.msg import Vector3Stamped
from tf2_geometry_msgs import do_transform_vector3
from std_msgs.msg import String
from tf2_ros import TransformListener, Buffer
from std_srvs.srv import Empty
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
from arm_utils.srv import ExecuteTrajectory



IS_ACTIVE = False
LAST_VEL = None
LAST_JOINT = None
JOINT_HISTORY = []


def convert_vel_to_msg(vec, a=0.5):
    if isinstance(vec, Vector3Stamped):
        vec = vec.vector
    msg = 'speedl([{:.4f}, {:.4f}, {:.4f}, 0, 0, 0], {:.4f}, 0.05)'.format(vec.x, vec.y, vec.z, a)
    return msg

def update_joints(msg):
    global LAST_JOINT
    LAST_JOINT = msg

def handle_vel(vec_stamped):

    if not IS_ACTIVE:
        rospy.logwarn_throttle(1.0, 'Velocity command was received, but the servoing has not been activated!')
        return

    stamp = vec_stamped.header.stamp
    success = tf_buffer.can_transform('base', vec_stamped.header.frame_id, stamp, timeout=rospy.Duration(0.5))
    if not success:
        raise Exception("Failed to retrieve TF!")
    tf = tf_buffer.lookup_transform('base', vec_stamped.header.frame_id, stamp)
    vec_base = do_transform_vector3(vec_stamped, tf)

    global LAST_VEL
    LAST_VEL = vec_base

def servo_activate(*_, **__):

    global LAST_VEL
    global JOINT_HISTORY
    global IS_ACTIVE

    LAST_VEL = None
    JOINT_HISTORY = []
    IS_ACTIVE = True
    rospy.loginfo('Servoing activated!')

    return []

def servo_stop(*_, **__):
    global LAST_VEL
    global IS_ACTIVE
    rospy.loginfo('Servoing stopped!')

    LAST_VEL = None
    IS_ACTIVE = False
    return []

def servo_rewind(*_, **__):

    if IS_ACTIVE:
        rospy.logerr('Cannot rewind while seroving is active!')
        return []

    if not JOINT_HISTORY:
        rospy.logwarn('No joint history to replay!')
        return []

    last_pos = np.array(JOINT_HISTORY[-1].position)
    curr_pos = np.array(LAST_JOINT.position)
    if np.abs(last_pos - curr_pos).sum() > np.radians(5.0):
        rospy.logerr('The current joints are more than 5 degrees off from the last recorded joint! Not replaying...')
        return []

    # Convert history to joint trajectory
    last_time = JOINT_HISTORY[-1].header.stamp
    traj = JointTrajectory()
    traj.joint_names = JOINT_HISTORY[-1].name
    for joint_state in JOINT_HISTORY[::-1]:
        traj_point = JointTrajectoryPoint()
        traj_point.positions = joint_state.position
        traj_point.time_from_start = last_time - joint_state.header.stamp
        traj.points.append(traj_point)

    traj_srv(traj)

    return []

if __name__ == '__main__':
    rospy.init_node('ur_servoing')


    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer)
    rospy.sleep(0.5)

    rospy.Service('servo_activate', Empty, servo_activate)
    rospy.Service('servo_stop', Empty, servo_stop)
    rospy.Service('servo_rewind', Empty, servo_rewind)
    traj_srv = rospy.ServiceProxy('execute_trajectory', ExecuteTrajectory)
    vel_sub = rospy.Subscriber('vel_command', Vector3Stamped, handle_vel, queue_size=1)
    joint_sub = rospy.Subscriber('/joint_states', JointState, update_joints, queue_size=1)

    urscript_topic = '/ur_hardware_interface/script_command'
    urscript_pub = rospy.Publisher(urscript_topic, String, queue_size=1)


    # MAIN LOOP
    rate = rospy.Rate(50)
    tol = np.radians(2.0)

    while not rospy.is_shutdown():
        if IS_ACTIVE and LAST_VEL is not None:
            msg = convert_vel_to_msg(LAST_VEL)
            urscript_pub.publish(msg)
            if not JOINT_HISTORY and LAST_JOINT is not None:
                JOINT_HISTORY.append(LAST_JOINT)
            else:
                last_pos = np.array(JOINT_HISTORY[-1].position)
                curr_pos = np.array(LAST_JOINT.position)
                if np.abs(last_pos - curr_pos).sum() > tol:
                    JOINT_HISTORY.append(LAST_JOINT)

        rate.sleep()
