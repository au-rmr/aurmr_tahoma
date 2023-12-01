import rospy
from smach import State
from smach_ros import SimpleActionState
from std_msgs.msg import String
from std_srvs.srv import Trigger


class Wait(State):
    def __init__(self, amount):
        State.__init__(self, outcomes=["succeeded"])
        self.amount = amount

    def execute(self, ud):
        rospy.sleep(self.amount)
        return "succeeded"


class NoOp(Wait):
    def __init__(self):
        Wait.__init__(self, 0)

