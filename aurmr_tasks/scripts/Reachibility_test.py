#!/usr/bin/env python

import smach_ros
import std_msgs
from std_msgs.msg import Header
from smach import State, StateMachine
import math
import rospy

from aurmr_perception.util import vec_to_quat_msg, I_QUAT
from aurmr_tasks.common import states
from aurmr_tasks.common.tahoma import Tahoma
import aurmr_tasks.common.control_flow as cf
from aurmr_tasks.common import motion
import numpy as np
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from tf_conversions import transformations

def main():
    rospy.loginfo('Getting robot resources')
    rospy.init_node("Reacheability_Test")
    simulation = rospy.get_param("/use_sim_time", False)
    State.simulation = simulation
    robot = Tahoma(simulation)

    # align_to_bin_orientation = transformations.quaternion_from_euler(-1.57, 0, 0)
    # align_to_bin_quat = Quaternion(x=align_to_bin_orientation[0], y=align_to_bin_orientation[1], z=align_to_bin_orientation[2], w=align_to_bin_orientation[3])
    # for i in range(4):
    #     for letter in "abcdefghijklm":
    #         frame_name = f"pod_bin_{i+1}{letter}"
    #         BIN_APPROACH_POSES.append(PoseStamped(header=std_msgs.msg.Header(frame_id=frame_name), pose=Pose(position=Point(x=.125, y=-.25,z=.05), orientation=align_to_bin_quat)))
    # print("Determining approachable bins...")
    # REACHABLE_BIN_APPROACHES = []
    # for i, pose in enumerate(BIN_APPROACH_POSES):
    #     solution = robot.compute_ik(pose, rospy.Duration(.1))
    #     if solution:
    #         REACHABLE_BIN_APPROACHES.append(pose)

    # print(f"{len(REACHABLE_BIN_APPROACHES)} of {len(BIN_APPROACH_POSES)} bins are approachable")

    pick_sm = StateMachine(["succeeded", "preempted", "aborted"],
                           input_keys=[],
                           output_keys=[])
    
    
    WAYPOINTS = []
    # INIT_POSE = PoseStamped(header=Header(frame_id="base_link"), pose=Pose(position=Point(x=0.3, y=-0.5,z=1.4), orientation=quat_obj))
    # WAYPOINTS.append(INIT_POSE)
    # PRE_GRASP_POSE = PoseStamped(header=Header(frame_id="base_link"), pose=Pose(position=Point(x=0.5, y=-0.2,z=1.25), orientation=Quaternion(x=0, y=0.707, z=0, w=0.707)))
    # WAYPOINTS.append(PRE_GRASP_POSE)
    # PRE_GRASP_POSE = PoseStamped(header=Header(frame_id="base_link"), pose=Pose(position=Point(x=0.5, y=0.3,z=1.25), orientation=quat_obj))
    # WAYPOINTS.append(PRE_GRASP_POSE)
    # PRE_GRASP_POSE = PoseStamped(header=Header(frame_id="base_link"), pose=Pose(position=Point(x=0.5, y=-0.2,z=1.45), orientation=quat_obj))
    # WAYPOINTS.append(PRE_GRASP_POSE)
    # print(REACHABLE_BIN_APPROACHES)
    # for i in range(5):
    RETAIN_SPACE = 0.1
    for j in range(4):
        
        PRE_GRASP_POSE = PoseStamped(header=Header(frame_id="base_link"), pose=Pose(position=Point(x=0.54-RETAIN_SPACE, y=-0.4+j*0.24,z=1.565), orientation=Quaternion(x=0, y=0.707, z=0, w=0.707)))
        WAYPOINTS.append(PRE_GRASP_POSE)
        PRE_GRASP_POSE = PoseStamped(header=Header(frame_id="base_link"), pose=Pose(position=Point(x=0.54-RETAIN_SPACE, y=-0.4+j*0.24,z=1.445), orientation=Quaternion(x=0, y=0.707, z=0, w=0.707)))
        WAYPOINTS.append(PRE_GRASP_POSE)
        PRE_GRASP_POSE = PoseStamped(header=Header(frame_id="base_link"), pose=Pose(position=Point(x=0.54-RETAIN_SPACE, y=-0.4+j*0.24,z=1.25), orientation=Quaternion(x=0, y=0.707, z=0, w=0.707)))
        WAYPOINTS.append(PRE_GRASP_POSE)
        PRE_GRASP_POSE = PoseStamped(header=Header(frame_id="base_link"), pose=Pose(position=Point(x=0.54-RETAIN_SPACE, y=-0.4+j*0.24,z=1.1), orientation=Quaternion(x=0, y=0.707, z=0, w=0.707)))
        WAYPOINTS.append(PRE_GRASP_POSE)
        PRE_GRASP_POSE = PoseStamped(header=Header(frame_id="base_link"), pose=Pose(position=Point(x=0.54-RETAIN_SPACE, y=-0.4+j*0.24,z=0.95), orientation=Quaternion(x=0, y=0.707, z=0, w=0.707)))
        WAYPOINTS.append(PRE_GRASP_POSE)
    
    # DROP_POSE = PoseStamped(header=Header(frame_id="base_link"), pose=Pose(position=Point(x=0.0, y=-0.45,z=1.1), orientation=Quaternion(x=0, y=0.707, z=0, w=0.707)))
    INIT_POSE = PoseStamped(header=Header(frame_id="base_link"), pose=Pose(position=Point(x=0.0, y=-0.5, z=1.5), orientation=Quaternion(x=0, y=0.707, z=0, w=0.707)))
    
    align_to_tote_orientation = transformations.quaternion_from_euler(math.pi, 0, 0)
    align_to_bin_quat = vec_to_quat_msg(align_to_tote_orientation)
    DROP_POSE = PoseStamped(header=Header(frame_id="base_link"), pose=Pose(position=Point(x=0.0, y=-0.45,z=1.1), orientation=align_to_bin_quat ))
    
    with pick_sm:
        StateMachine.add_auto("CLEAR_SCENE", motion.ClearCollisionGeometry(robot), ["succeeded"])
        StateMachine.add_auto("SETUP_COLLISION_SCENE", motion.AddPodCollisionGeometry(robot), ["succeeded"])
        cf.inject_userdata_auto("LOAD_POSES_PICK", "poses_pick", WAYPOINTS)
        StateMachine.add("PICK_POSE", cf.IterateList("poses_pick", "pose"), {"repeat": "MOVE_TO_POSE_PICK", "done": "aborted"})
        StateMachine.add('MOVE_TO_POSE_PICK', motion.MoveEndEffectorToPose(robot), {"succeeded": "PAUSE_PICK", "aborted": "PAUSE_PICK"})
        # Clear the pod object so the robot can reach into the bins
        StateMachine.add("PAUSE_PICK", states.Wait(1), {"succeeded": "MOVE_TO_GRASP"})
        # StateMachine.add("CLEAR_SCENE_GRASP", motion.ClearCollisionGeometry(robot), {"succeeded": "MOVE_TO_GRASP"})
        StateMachine.add("MOVE_TO_GRASP", motion.MoveEndEffectorToOffset(robot, (0, 0, RETAIN_SPACE+0.05)), {"succeeded": "PAUSE_GRASP", "aborted": "PAUSE_GRASP"})
        StateMachine.add("PAUSE_GRASP", states.Wait(1), {'succeeded': "MOVE_UP"})
        # StateMachine.add("GRASP", motion.CloseGripper(robot), {'succeeded': "PAUSE_GRASP_2"})
        # StateMachine.add("PAUSE_GRASP_2", states.Wait(1), {'succeeded': "MOVE_UP"})
        StateMachine.add("MOVE_UP", motion.MoveEndEffectorToOffset(robot, (-0.02, 0.0, 0.0)), {"succeeded": "PAUSE_UP", "aborted": "PAUSE_UP"})
        StateMachine.add("PAUSE_UP", states.Wait(1), {"succeeded": "MOVE_BACK"})
        StateMachine.add("MOVE_BACK", motion.MoveEndEffectorToOffset(robot, (0, 0, -(RETAIN_SPACE+0.05))), {"succeeded": "MOVE_TO_POSE_DROP", "aborted": "MOVE_TO_POSE_DROP"})
        # StateMachine.add("SETUP_COLLISION_SCENE_PICK", motion.AddPodCollisionGeometry(robot), {"succeeded": "MOVE_TO_POSE_DROP"})
        StateMachine.add('MOVE_TO_POSE_DROP', motion.MoveEndEffectorToPose(robot, DROP_POSE), {"succeeded": "PICK_POSE", "aborted": "PICK_POSE"})

        # StateMachine.add("ASK_FOR_GRIPPER_OPEN_TOTE_RELEASE", motion.OpenGripper(robot), {'succeeded': "succeeded"})

        
        # StateMachine.add_auto("POKE_BIN", motion.MoveEndEffectorInLineInOut(robot), ['succeeded', 'preempted', 'aborted'],
        #                         {"succeeded": "PICK_POSE", "aborted": "PICK_POSE"})
     

    # Create top state machine
    sm = StateMachine(outcomes=['succeeded', "preempted", 'aborted'])

    with sm:
        StateMachine.add('PLAY_POSE_SEQUENCE', pick_sm, {"succeeded": "succeeded", "aborted": "aborted"})

    rospy.loginfo('Beginning pick SM')

    sis = smach_ros.IntrospectionServer('pick_sm', sm, '/pick')
    sis.start()

    outcome = sm.execute()

    # rospy.spin()
    sis.stop()


if __name__ == '__main__':
    main()
