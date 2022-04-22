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

        pose_stamped = PoseStamped(header=std_msgs.msg.Header(frame_id="base_link"))
        pose_stamped.pose.position.x = pose["xyz"][0]
        pose_stamped.pose.position.y = pose["xyz"][1]
        pose_stamped.pose.position.z = pose["xyz"][2]
        quaternion = transformations.quaternion_from_euler(*pose["rpy"])
        pose_stamped.pose.orientation.x = quaternion[0]
        pose_stamped.pose.orientation.y = quaternion[1]
        pose_stamped.pose.orientation.z = quaternion[2]
        pose_stamped.pose.orientation.w = quaternion[3]
        error = self.robot.move_to_pose_goal(pose_stamped)
        if error is None:
            return "succeeded"
        else:
            print(error)
            return "aborted"
