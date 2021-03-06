from smach import State
import rospy


def prompt_for_confirmation(prompt):
    valid_signal = False
    while not rospy.is_shutdown() and not valid_signal:
        user_input = input(f"{prompt}\nProceed [y/n]?")
        if user_input == "y":
            return True
        elif user_input == "n":
            return False


class AskForHumanAction(State):
    def __init__(self, default_prompt=None):
        State.__init__(self, outcomes=['succeeded', 'aborted'], input_keys=["prompt"])
        self.prompt = default_prompt

    def execute(self, userdata):
        prompt = self.prompt
        if not self.prompt:
            prompt = userdata["prompt"]
        confirmed = prompt_for_confirmation(prompt)
        if confirmed:
            return "succeeded"
        else:
            return "aborted"