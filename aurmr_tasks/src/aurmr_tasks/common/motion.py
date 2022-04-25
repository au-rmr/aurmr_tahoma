import std_msgs.msg

from smach import State

from geometry_msgs.msg import (
    PoseStamped,

)

from tf_conversions import transformations


class MoveToJointPositions(State):
    def __init__(self, robot, positions):
        State.__init__(self, outcomes=['succeeded', 'aborted'])
        self.robot = robot
        self.positions = positions

    def execute(self, ud):
        self.robot.move_to_joint_positions(self.positions)
        return "succeeded"


class MoveEndEffectorToPose(State):
    def __init__(self, robot, default_pose=None):
        State.__init__(self, input_keys=['pose'], outcomes=['succeeded', 'preempted', 'aborted'])
        self.robot = robot
        self.default_pose = default_pose

    def execute(self, userdata):
        if self.default_pose:
            pose = self.default_pose
        else:
            pose = userdata["pose"]

        success = self.robot.move_to_pose_goal(pose)
        if success:
            return "succeeded"
        else:
            return "aborted"


class MoveEndEffectorInLineInOut(State):
    def __init__(self, robot):
        State.__init__(self, input_keys=['pose'], outcomes=['succeeded', 'preempted', 'aborted'])
        self.robot = robot

    def execute(self, userdata):
        # FIXME(nickswalker): This is hacked together for basic demonstration. Fix this method later
        error = self.robot.straight_move_to_pose(None)
        if error is None:
            return "succeeded"
        else:
            print(error)
            return "aborted"
