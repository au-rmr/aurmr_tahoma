#!/usr/bin/env python
import math
from multiprocessing.connection import wait
from os import waitstatus_to_exitcode
from std_msgs.msg import Header, _String
import matplotlib.pyplot as plt

from typing import List
import rospy
from robotiq_2f_gripper_control.msg import vacuum_gripper_input
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from tf_conversions import transformations
import numpy as np

global forces
forces = []

def callback_feedback(data):
    global forces
    input_force = data.gPO
    print(input_force)
    if(input_force < 99):
        forces.append(input_force)
    


def main():
    rospy.init_node("suction_gripper_force_graph")
    rospy.Subscriber("/gripper_control/status", vacuum_gripper_input, callback_feedback)
    while not rospy.is_shutdown():
    # do some work or nothing
        continue
    name = input("Name of the plot")
    plt.plot(forces)
    plt.xlabel('timestep')
    plt.ylabel('force magnitude')
    # plt.show()
    plt.savefig("/home/aurmr/workspaces/py39_ws_clone/src/aurmr_tahoma/aurmr_tasks/graphs_suction_gripper/"+name+".png")
    # if(data.)


if __name__ == '__main__':
    main()


