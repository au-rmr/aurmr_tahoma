#!/usr/bin/env python
import rospy
import numpy as np
import math
from geometry_msgs.msg import PoseStamped, WrenchStamped
from robotiq_2f_gripper_control.msg import vacuum_gripper_input
from std_msgs.msg import Byte, Int16
import time
import pandas as pd

import json
from datetime import datetime

global reaction_force, reaction_torque, vaccum_flow, recording_status

reaction_force = np.array([0.0, 0.0, 0.0])
reaction_torque = np.array([0.0, 0.0, 0.0])
vaccum_flow = 0
recording_status = 0

def wrench_cb(msg):
    global reaction_force, reaction_torque
    reaction_force = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
    reaction_torque = np.array([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])

def vaccum_gripper_cb(msg):
    global vaccum_flow
    vaccum_flow = msg.gPO

def suction_cup_properties_cb(msg):
    global recording_status
    recording_status = msg.data

def main():
    rospy.init_node("suction_cup_deformation_reaction_force_and_vaccum_flow_data_collection")
    wrench_listener = rospy.Subscriber("/wrench", WrenchStamped, wrench_cb)
    vaccum_flow_listener = rospy.Subscriber("/gripper_control/status", vacuum_gripper_input, vaccum_gripper_cb)
    recording_status_listener = rospy.Subscriber("/suction_cup_properties", Byte, suction_cup_properties_cb)

    data_id = input("Enter the index of the data point generation: ")
    data_id = int(data_id)
    data_id_pub = rospy.Publisher('/data_id_topic', Int16, queue_size=10, latch=True)
    data_id_pub.publish(data_id)

    suction_cup_properties_reaction_force = np.array([])
    suction_cup_properties_vaccum_flow = np.array([])
    real_time_counter = 0
    while not rospy.is_shutdown():
        if(recording_status == 1):
            suction_cup_properties_reaction_force = np.vstack([suction_cup_properties_reaction_force, reaction_force]) if suction_cup_properties_reaction_force.size else reaction_force
            suction_cup_properties_vaccum_flow = np.hstack([suction_cup_properties_vaccum_flow, int(vaccum_flow)])
            real_time_counter = 1
            # print(reaction_force, vaccum_flow)
        else:
            if(real_time_counter == 1):
                datetime_string = datetime.now().isoformat().replace(":","")[:-7]

                real_time_counter = 0

                suction_cup_properties_reaction_force = np.transpose(suction_cup_properties_reaction_force)
                # print(suction_cup_properties_reaction_force)
                # dict_values = {"Reaction Force x" : suction_cup_properties_reaction_force[0], "Reaction Force y" : suction_cup_properties_reaction_force[1], "Reaction Force z" : suction_cup_properties_reaction_force[2], "Vaccum Flow" : suction_cup_properties_vaccum_flow.tolist(), "Suction Scores" : suction_score_array.tolist()}
                if(recording_status == 0):
                    dict_values = {"Reaction Force x" : suction_cup_properties_reaction_force[0].tolist(), "Reaction Force y" : suction_cup_properties_reaction_force[1].tolist(), "Reaction Force z" : suction_cup_properties_reaction_force[2].tolist(), "Vaccum Flow" : suction_cup_properties_vaccum_flow.tolist(), "success" : False}
                elif(recording_status == 2):
                    dict_values = {"Reaction Force x" : suction_cup_properties_reaction_force[0].tolist(), "Reaction Force y" : suction_cup_properties_reaction_force[1].tolist(), "Reaction Force z" : suction_cup_properties_reaction_force[2].tolist(), "Vaccum Flow" : suction_cup_properties_vaccum_flow.tolist(), "success" : True}
                print("dict values", dict_values)
                new_dir_name = str(data_id)
                data_path = "/home/aurmr/workspaces/scenario_based_grasp_point_selection/src/real_world_data/"
                save_dir_json = data_path+"/control_json_data_"+new_dir_name+"_"+datetime_string+".json"
                with open(save_dir_json, 'w') as json_file:
                    json.dump(dict_values, json_file)
                data_id += 1
                data_id_pub.publish(data_id)

                suction_cup_properties_reaction_force = np.array([])
                suction_cup_properties_vaccum_flow = np.array([])
        rospy.sleep(0.005)


if __name__ == '__main__':
    main()