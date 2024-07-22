#! /usr/bin/env python3

import rospy

import actionlib

from control_msgs.msg import GripperCommandAction, GripperCommandGoal, GripperCommandResult, GripperCommandFeedback
# from robotiq_2f_gripper_control.msg import Robotiq2FGripper_robot_input, Robotiq2FGripper_robot_output
from robotiq_2f_gripper_control.msg import finger_gripper_input, finger_gripper_output, vacuum_gripper_input, vacuum_gripper_output

class GripperActionServer(object):
    _feedback = GripperCommandFeedback()
    _result = GripperCommandResult()
    def __init__(self, name, gripper_type):
        # create messages that are used to publish feedback/result
        self.gripper_type = gripper_type
        if(gripper_type == "vacuum"):
            self.input_msg = vacuum_gripper_input
            self.output_msg = vacuum_gripper_output
        elif(gripper_type == "finger"):
            self.input_msg = finger_gripper_input
            self.output_msg = finger_gripper_output

        self._status = self.input_msg()


        self._action_name = name
        self._as = actionlib.SimpleActionServer(self._action_name, GripperCommandAction, execute_cb=self.execute_cb, auto_start = False)
        self._as.start()
        self.command_pub = rospy.Publisher("/gripper_control/command", self.output_msg, queue_size=5)
        self.status_listener = rospy.Subscriber("/gripper_control/status", self.input_msg, queue_size=5,
                                                callback=self.update_status)

    def update_status(self, msg):
        self._status = msg
      
    def execute_cb(self, goal: GripperCommandGoal):

        # helper variables
        r = rospy.Rate(10)
        success = True
        
        # publish info to the console for the user
        rospy.loginfo('%s: Moving gripper to %f with effort %f' % (
        self._action_name, goal.command.position, goal.command.max_effort))
        if self.gripper_type == "finger":
            command = self.output_msg()
            command.rPR = int(goal.command.position * 255)
            command.rSP = 100
            command.rFR = int(goal.command.max_effort * 255)
        elif self.gripper_type == "vacuum":
            command = self.output_msg()
            print(goal.command.position)
            if goal.command.position != 0:
                command.rPR = 0
            else:
                command.rPR = 200
        print(command)
        self.command_pub.publish(command)

        # TODO(Jack, 6/21): Find a better way around the sleep to check when the gripper has actually recieved the command
        rospy.sleep(1)

        # start executing the action
        while not rospy.is_shutdown():
            # check that preempt has not been requested by the client
            if self._as.is_preempt_requested():
                rospy.loginfo('%s: Preempted' % self._action_name)
                self._as.set_preempted()
                success = False
                break
            elif self._status.gOBJ != 0: # The gripper either detected an obstacle or is finished moving
                self._feedback.reached_goal == True
                break
            # publish the feedback
            self._feedback.position = self._status.gPO
            self._as.publish_feedback(self._feedback)
            # this step is not necessary, the sequence is computed at 1 Hz for demonstration purposes
            r.sleep()
          
        if success:
            self._result.position = self._feedback.position
            self._result.reached_goal = success
            rospy.loginfo('%s: Succeeded' % self._action_name)
            self._as.set_succeeded(self._result)
        
if __name__ == '__main__':
    rospy.init_node('gripper_action_server')
    gripper_type = rospy.get_param("~gripper_type", "finger")
    server = GripperActionServer("gripper_controller/gripper_cmd", gripper_type)
    rospy.spin()
