#!/usr/bin/env python

from smach import State, StateMachine

import rospy

from aurmr_tasks.common import motion, perception
from aurmr_tasks.common import hri

import aurmr_tasks.common.control_flow as cf
from aurmr_tasks.util import formulate_ud_str_auto

# HACK: Pulled from the database and hardcoded. Might want to query for these instead?
BIN_CODE_TO_BIN_ID = {
      "P-8-H119H546": "1A",
      "P-8-H119H547": "2A",
      "P-8-H119H548": "3A",
      "P-8-H119H549": "4A",
      "P-8-H208H542": "1B",
      "P-8-H208H543": "2B",
      "P-8-H208H544": "3B",
      "P-8-H208H545": "4B",
      "P-8-H336H066": "1C",
      "P-8-H336H067": "2C",
      "P-8-H336H068": "3C",
      "P-8-H336H069": "4C",
      "P-8-H470H838": "1D",
      "P-8-H470H839": "2D",
      "P-8-H470H840": "3D",
      "P-8-H470H841": "4D",
      "P-9-H099H586": "1I",
      "P-9-H099H587": "2I",
      "P-9-H099H588": "3I",
      "P-9-H099H589": "4I",
      "P-9-H276H282": "1J",
      "P-9-H276H283": "2J",
      "P-9-H276H284": "3J",
      "P-9-H276H285": "4J",
      "P-9-H373H538": "1K",
      "P-9-H373H539": "2K",
      "P-9-H373H540": "3K",
      "P-9-H373H541": "4K",
      "P-7-H866F302": "1L",
      "P-7-H866F303": "2L",
      "P-7-H866F304": "3L",
      "P-7-H866F305": "4L",
      "P-7-H218F218": "1M",
      "P-7-H218F219": "2M",
      "P-7-H218F220": "3M",
      "P-7-H218F221": "4M",
      "P-8-H588H170": "1E",
      "P-8-H588H171": "2E",
      "P-8-H588H172": "3E",
      "P-8-H588H173": "4E",
      "P-8-H758H650": "1F",
      "P-8-H758H651": "2F",
      "P-8-H758H652": "3F",
      "P-8-H758H653": "4F",
      "P-6-H835J238": "1G",
      "P-6-H835J239": "2G",
      "P-6-H835J240": "3G",
      "P-6-H835J241": "4G",
      "P-9-H051H030": "1H",
      "P-9-H051H031": "2H",
      "P-9-H051H032": "3H",
      "P-9-H051H033": "4H",
      "P-8-H588H494": "1E",
      "P-8-H588H495": "2E",
      "P-8-H588H496": "3E",
      "P-8-H588H497": "4E",
      "P-8-H758H974": "1F",
      "P-8-H758H975": "2F",
      "P-8-H758H976": "3F",
      "P-8-H758H977": "4F",
      "P-8-H888H454": "1G",
      "P-8-H888H455": "2G",
      "P-8-H888H456": "3G",
      "P-8-H888H457": "4G",
      "P-9-H051H354": "1H",
      "P-9-H051H355": "2H",
      "P-9-H051H356": "3H",
      "P-9-H051H357": "4H",
      "P-9-M095R783": "1C",
      "P-9-M095R784": "2C",
      "P-9-M095R785": "3C",
      "P-9-M223R307": "1D",
      "P-9-M223R308": "2D",
      "P-9-M223R309": "3D",
      "P-9-M503R831": "1E",
      "P-9-M503R832": "2E",
      "P-9-M503R833":  "3E",
      "L[9[D051D354": "1H",
        "L[9[D051D355": "2H",
        "L[9[D051D356": "3H",
        "L[9[D051D357": "4H",
        "L[8[D888D454": "1G",
        "L[8[D888D455": "2G",
        "L[8[D888D456": "3G",
        "L[8[D888D457": "4G",
        "L[8[D758D974": "1F",
        "L[8[D758D975": "2F",
        "L[8[D758D976": "3F",
        "L[8[D758D977": "4F",
        "L[8[D588D494": "1E",
        "L[8[D588D495": "2E",
        "L[8[D588D496": "3E",
        "L[8[D588D497": "4E",

}


BIN_IDS = []

for letter in "abcdefghijklm":
    for i in range(4):
        BIN_IDS.append(f"{i + 1}{letter}")


def load_sm(stows):
    sm = StateMachine(["succeeded", "preempted", "aborted"],
                           input_keys=[],
                           output_keys=[])
    print("STOWS:", stows)

    with sm:
        cf.inject_userdata_auto("LOAD_STOWS", "stows", stows)

        StateMachine.add("ITERATE_STOWS", cf.IterateList("stows", "stow"), {"repeat": "SPLAT_STOW", "done": "succeeded" })
        cf.splat_auto("SPLAT_STOW", "stow", ["target_bin_id", "target_object_id"])
        StateMachine.add_auto("PRE_PERCEIVE", perception.CaptureEmptyBin(), ["succeeded"])
        formulate_ud_str_auto("MAKE_PROMPT_STRING", "Load bin {} with the {}", ["target_bin_id", "target_object_id"], "prompt")
        StateMachine.add_auto("ASK_FOR_BIN_LOAD", hri.AskForHumanAction(), ["succeeded"])
        StateMachine.add_auto("POST_PERCEIVE", perception.StowObject(), ["aborted"], {"succeeded": "ITERATE_STOWS"})

    return sm
