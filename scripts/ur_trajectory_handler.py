#!/usr/bin/env python

import rospy
from contextlib import contextmanager
from std_srvs.srv import Empty, Trigger
from ur_dashboard_msgs.srv import Load
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
import numpy as np
from arm_utils.srv import ExecuteTrajectory

class URProgramManager:
    def __init__(self):
        self.program_load_srv = rospy.ServiceProxy('/ur_hardware_interface/dashboard/load_program', Load)
        self.program_play_srv = rospy.ServiceProxy('/ur_hardware_interface/dashboard/play', Trigger)
        self.program_stop_srv = rospy.ServiceProxy('/ur_hardware_interface/dashboard/stop', Trigger)

    @contextmanager
    def enable_external_control(self):
        self.program_load_srv(rospy.get_param('extcontrol', 'extcontrol.urp'))
        rospy.loginfo('External control enabled!')
        self.program_play_srv()
        rospy.sleep(0.25)
        try:
            yield
        finally:
            self.program_stop_srv()
            rospy.sleep(0.25)
            rospy.loginfo('External control disabled!')

    def external_control_decorator(self, func):
        def ret_func(*args, **kwargs):
            with self.enable_external_control():
                return func(*args, **kwargs)

        return ret_func

def execute(req):
    traj = req.traj
    goal = FollowJointTrajectoryGoal()
    goal.trajectory = traj
    client.send_goal(goal)
    client.wait_for_result()

    return []

if __name__ == '__main__':
    rospy.init_node('ur_trajectory_handler')

    ur_manager = URProgramManager()
    rospy.Service('execute_trajectory', ExecuteTrajectory, ur_manager.external_control_decorator(execute))

    client_name = 'scaled_pos_joint_traj_controller/follow_joint_trajectory'
    client = actionlib.SimpleActionClient(client_name, FollowJointTrajectoryAction)
    if not client.wait_for_server(rospy.Duration(5.0)):
        raise Exception("Could not connect to arm client!")

    rospy.spin()