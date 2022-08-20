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


class WaitForStart(State):
    def __init__(self):
        State.__init__(self, outcomes=['signalled', 'not_signalled'])
        self.trigger_service_client = rospy.ServiceProxy('wrist_trigger', Trigger)

    def execute(self, userdata):
        # Assume we've been tapped in simulation
        try:
            if State.simulation:
                return 'signalled'
        except AttributeError:
            pass

        try:
            rospy.wait_for_service('wrist_trigger', timeout=10)
        except rospy.ROSException:
            rospy.logerr("Couldn't obtain wrist_trigger service handle")
            return 'not_signalled'
        rospy.loginfo('Waiting for start signal...')
        try:
            response = self.trigger_service_client()
            rospy.loginfo(response.message)
            if response.success:
                return 'signalled'
            else:
                return 'not_signalled'
        except rospy.ServiceException():
            return 'not_signalled'

class WaitForKeyPress(State):
    def __init__(self):
        State.__init__(self, outcomes=['signalled', 'not_signalled', 'aborted'])

    def execute(self, userdata):
        try:
            user_input = input("Press any key to start\n")
            return 'signalled'
        except KeyboardInterrupt:
            return 'aborted'



