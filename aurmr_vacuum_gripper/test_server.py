#!/usr/bin/env python3
import os
from control_msgs.msg import GripperCommandAction, GripperCommandGoal
import actionlib
from actionlib import SimpleActionClient
# from vacuum_gripper_control.msg import vacuum_gripper_set, vacuum_gripper_get
import rospy
import numpy as np

GRIPPER_ACTION_SERVER = '/gripper_controller/gripper_cmd'

class VacuumEjector:
    def __init__(self):
        self.client = actionlib.SimpleActionClient(GRIPPER_ACTION_SERVER, GripperCommandAction)
        self.client.wait_for_server()
        self.flag = [True, True, True, True]

    def send_goal(self, ejectors:list=[5,1,1,0,0]):
        if self.client.wait_for_server(timeout=rospy.Duration(5)):  # Wait for 5 seconds
            rospy.loginfo(f"Action server is running! Sending goal {ejectors}")
            goal = GripperCommandGoal()
            goal.command.position = int(''.join(map(str, ejectors)))
            goal.command.max_effort = 1
            self.client.send_goal(goal)
            print(f'Server response: {self.client.wait_for_result()}')
        else:
            rospy.logwarn("Action server not available!")

    def check_action_server(self):

        # client = actionlib.SimpleActionClient(GRIPPER_ACTION_SERVER, GripperCommandAction)

        rospy.loginfo("Waiting for action server to start...")
        if self.client.wait_for_server(timeout=rospy.Duration(5)):  # Wait for 5 seconds
            rospy.loginfo("Action server is running!")
        else:
            rospy.logwarn("Action server not available!")


if __name__ == '__main__':
    rospy.init_node('reactive_vacuum')
    vacuum_ejector = VacuumEjector()
    vacuum_ejector.check_action_server()
    vacuum_ejector.send_goal([5,1,1,1,1])
    rospy.sleep(5)
    vacuum_ejector.send_goal([5,0,0,0,0])