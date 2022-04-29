from smach import State
import rospy


class AskForHumanAction(State):
    def __init__(self, prompt):
        State.__init__(self, outcomes=['succeeded', 'aborted'])
        self.prompt = prompt

    def execute(self, userdata):
        try:
            valid_signal = False
            while not rospy.is_shutdown() and not valid_signal:
                user_input = input(f"{self.prompt}\nProceed [y/n]?")
                if user_input == "y":
                    return 'succeeded'
                elif user_input == "n":
                    return 'aborted'
        except KeyboardInterrupt:
            return 'aborted'