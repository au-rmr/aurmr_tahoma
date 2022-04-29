import geometry_msgs.msg
import rospy
import std_msgs.msg

from smach import State

from geometry_msgs.msg import (
    PoseStamped,

)
import rospy
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from smach import State
from smach_ros import SimpleActionState
from std_msgs.msg import String, Header
from std_srvs.srv import Trigger


from tf_conversions import transformations

I_QUAT = Quaternion(x=0, y=0, z=0, w=1)

TRIPOD_ORIENTATION = transformations.quaternion_from_euler(0, -.3,0)
TRIPOD_ORIENTATION = Quaternion(x=TRIPOD_ORIENTATION[0],y= TRIPOD_ORIENTATION[1],z=TRIPOD_ORIENTATION[2],w=TRIPOD_ORIENTATION[3])


class MoveToJointAngles(State):
    def __init__(self, robot, positions):
        State.__init__(self, outcomes=['succeeded', 'aborted'])
        self.robot = robot
        self.positions = positions

    def execute(self, ud):
        self.robot.move_to_joint_angles(self.positions, 100)
        return "succeeded"


class MoveEndEffectorToPose(State):
    def __init__(self, robot, default_pose=None):
        State.__init__(self, input_keys=['pose'], outcomes=['succeeded', 'preempted', 'aborted'])
        self.robot = robot
        self.default_pose = default_pose
        self.target_pose_visualizer = rospy.Publisher("end_effector_target", geometry_msgs.msg.PoseStamped, queue_size=1, latch=True)

    def execute(self, userdata):
        if self.default_pose:
            pose = self.default_pose
        else:
            pose = userdata["pose"]

        self.target_pose_visualizer.publish(pose)
        success = self.robot.move_to_pose_goal(pose)
        if success:
            return "succeeded"
        else:
            return "aborted"


class MoveEndEffectorInLine(State):
    def __init__(self, robot, to_point):
        State.__init__(self, input_keys=['pose'], outcomes=['succeeded', 'preempted', 'aborted'])
        self.robot = robot
        self.to_point = to_point

    def execute(self, userdata):
        offset = self.to_point
        current = self.robot.move_group.get_current_pose().pose
        current.position.x += offset[0]
        current.position.y += offset[1]
        current.position.z += offset[2]
        error = self.robot.straight_move_to_pose(current, avoid_collisions=False)
        if error is None:
            return "succeeded"
        else:
            print(error)
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


class ClearCollisionGeometry(State):
    def __init__(self, robot):
        State.__init__(self, outcomes=["succeeded", "aborted"])
        self.robot = robot

    def execute(self, ud):
        self.robot.scene.clear()
        start = rospy.get_time()
        seconds = rospy.get_time()
        timeout = 5.0
        while (seconds - start < timeout) and not rospy.is_shutdown():
            objects = self.robot.scene.get_objects()

            # Test if we are in the expected state
            if len(objects) == 0:
                return "succeeded"

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        return "aborted"


class AddInHandCollisionGeometry(State):
    def __init__(self, robot):
        State.__init__(self, outcomes=["succeeded", "aborted"])
        self.robot = robot

    def execute(self, ud):
        self.robot.scene.add_box("item", PoseStamped(header=Header(frame_id="arm_tool0"),
                                                pose=Pose(position=Point(x=0, y=0, z=0.13), orientation=Quaternion(x=0, y=0, z=0, w=1))),
                            (.06, .13, .06))
        self.robot.scene.attach_box("arm_tool0", "item")
        start = rospy.get_time()
        seconds = rospy.get_time()
        timeout = 5.0
        while (seconds - start < timeout) and not rospy.is_shutdown():
            # Test if the box is in attached objects
            attached_objects = self.robot.scene.get_attached_objects(["item"])
            is_attached = len(attached_objects.keys()) > 0

            # Test if the box is in the scene.
            # Note that attaching the box will remove it from known_objects
            is_known = "item" in self.robot.scene.get_known_object_names()

            # Test if we are in the expected state
            if is_attached:
                return "succeeded"

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        return "aborted"


class AddPodCollisionGeometry(State):
    def __init__(self, robot):
        State.__init__(self, outcomes=["succeeded", "aborted"])
        self.robot = robot

    def execute(self, ud):
        self.robot.scene.add_box("pod_top", PoseStamped(header=Header(frame_id="base_link"),
                                                   pose=Pose(position=Point(x=1.35, y=0.14, z=1.88),
                                                             orientation=I_QUAT)))

        self.robot.scene.add_box("pod_bottom", PoseStamped(header=Header(frame_id="base_link"),
                                                      pose=Pose(position=Point(x=1.35, y=0.14, z=.68),
                                                                orientation=I_QUAT)))
        self.robot.scene.add_box("pod_left", PoseStamped(header=Header(frame_id="base_link"),
                                                    pose=Pose(position=Point(x=1.1, y=.69, z=1.27),
                                                              orientation=I_QUAT)), (.5, .5, .3))
        self.robot.scene.add_box("pod_right", PoseStamped(header=Header(frame_id="base_link"),
                                                     pose=Pose(position=Point(x=1.1, y=-.05, z=1.27),
                                                               orientation=I_QUAT)), (.5, .5, .3))
        self.robot.scene.add_box("tripod", PoseStamped(header=Header(frame_id="base_link"),
                                                  pose=Pose(position=Point(x=-.3, y=0, z=.8),
                                                            orientation=TRIPOD_ORIENTATION)), (.5, .5, 1))
        start = rospy.get_time()
        seconds = rospy.get_time()
        timeout = 50.0
        while (seconds - start < timeout) and not rospy.is_shutdown():
            objects = self.robot.scene.get_objects()

            # Test if we are in the expected state
            if len(objects) == 5:
                return "succeeded"

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        return "aborted"

