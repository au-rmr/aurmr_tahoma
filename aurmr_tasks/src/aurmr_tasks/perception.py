import std_msgs.msg

from smach import State

from geometry_msgs.msg import (
    PoseStamped,

)

from tf_conversions import transformations

from aurmr_tasks.common.states import Wait


class PerceiveBin(Wait):
    def __init__(self):
        Wait.__init__(self, 10)


class GetObjectPoints(Wait):
    def __init__(self):
        Wait.__init__(self, 10)


class GetGraspPoseForPoints(Wait):
    def __init__(self):
        Wait.__init__(self, 10)