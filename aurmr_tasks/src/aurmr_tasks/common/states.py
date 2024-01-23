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


class RecordTime(State):
    def __init__(self):
        State.__init__(self, outcomes=['succeeded'], output_keys=["start_time"])

    def execute(self, userdata):
        now = rospy.get_rostime()
        userdata["start_time"] = now.secs + now.nsecs*1E-9
        return "succeeded"
