#!/usr/bin/env python
import math
from std_msgs.msg import Header
import smach_ros

from smach import State, StateMachine

import rospy

from aurmr_tasks.common.tahoma import Tahoma
from aurmr_tasks.common import motion, perception
from aurmr_tasks import interaction
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from tf_conversions import transformations
import aurmr_tasks.common.control_flow as cf
from aurmr_tasks.util import formulate_ud_str_auto

BIN_IDS = []

for letter in "abcdefghijklm":
    for i in range(4):
        BIN_IDS.append(f"{i + 1}{letter}")


def load_sm(stows):

    sm = StateMachine(["succeeded", "preempted", "aborted"],
                           input_keys=[],
                           output_keys=[])

    with sm:
        cf.inject_userdata_auto("LOAD_STOWS", "stows", stows)
        StateMachine.add_auto("PRE_PERCEIVE", perception.CaptureEmptyBin(), ["succeeded"])
        StateMachine.add("ITERATE_STOWS", cf.IterateList("stows", "stow"), {"repeat": "ASK_FOR_BIN_LOAD", "done": "succeeded" })
        cf.splat_auto("SPLAT_STOW", "stow", ["bin_id", "object_id"])
        formulate_ud_str_auto("MAKE_PROMPT_STRING", "Load bin {} with the {}", ["bin_id", "object_id"], "prompt")
        StateMachine.add_auto("ASK_FOR_BIN_LOAD", interaction.AskForHumanAction(), ["succeeded"])
        StateMachine.add_auto("POST_PERCEIVE", perception.CaptureObject(), ["aborted"], {"succeeded": "ITERATE_STOWS"})

    return sm


