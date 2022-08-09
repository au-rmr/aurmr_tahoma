import geometry_msgs.msg
from sensor_msgs.msg import JointState
from smach import State, StateMachine
import rospy
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from smach import State
from std_msgs.msg import Header
from aurmr_tasks.interaction import prompt_for_confirmation
from aurmr_tasks.util import apply_offset_to_pose
from aurmr_tasks import interaction
from aurmr_perception.util import I_QUAT


class MoveToJointAngles(State):
    def __init__(self, robot, default_position=None):
        State.__init__(self, input_keys=["position"], outcomes=['succeeded', 'aborted'])
        self.robot = robot
        self.position = default_position

    def execute(self, ud):
        if self.position:
            target = self.position
        else:
            target = ud["position"]
        to_log = target
        if isinstance(target, JointState):
            to_log = target.position
        rospy.loginfo(f"Moving to {to_log}")
        success = self.robot.move_to_joint_angles(target)
        if success:
            return "succeeded"
        else:
            return "aborted"


class MoveEndEffectorToPose(State):
    def __init__(self, robot, default_pose=None):
        State.__init__(self, input_keys=['pose'], outcomes=['succeeded', 'preempted', 'aborted'])
        self.robot = robot
        self.default_pose = default_pose
        self.target_pose_visualizer = rospy.Publisher("end_effector_target", geometry_msgs.msg.PoseStamped,
                                                      queue_size=1, latch=True)

    def execute(self, userdata):
        if self.default_pose:
            pose = self.default_pose
        else:
            pose = userdata["pose"]

        self.target_pose_visualizer.publish(pose)
        success = self.robot.move_to_pose(pose)
        if success:
            return "succeeded"
        else:
            return "aborted"


class MoveEndEffectorToPoseLinear(State):
    def __init__(self, robot, to_pose):
        State.__init__(self, input_keys=['pose'], outcomes=['succeeded', 'preempted', 'aborted'])
        self.robot = robot
        self.to_pose = to_pose

    def execute(self, userdata):
        target_pose = self.to_pose
        if target_pose is None:
            target_pose = userdata["pose"]
        success = self.robot.straight_move_to_pose(target_pose, avoid_collisions=False)
        if success:
            return "succeeded"
        else:
            return "aborted"


class MoveEndEffectorToOffset(State):
    def __init__(self, robot, offset, frame=None, use_force=False, use_gripper=False):
        State.__init__(self, input_keys=['offset'], outcomes=['succeeded', 'preempted', 'aborted'])
        self.robot = robot
        self.offset = offset
        self.offset_frame = frame
        self.use_force = use_force
        self.use_gripper = use_gripper

    def execute(self, userdata):
        offset = self.offset
        if offset is None:
            offset = userdata["offset"]
        offset_frame = self.offset_frame
        if offset_frame is None:
            offset_frame = 'arm_tool0'
        # In base_link by default
        current = self.robot.move_group.get_current_pose()
        target_pose = apply_offset_to_pose(current, offset, offset_frame, self.robot.tf2_buffer)
        succeeded = self.robot.straight_move_to_pose(target_pose, avoid_collisions=False, jump_threshold=10.0, use_force=self.use_force, use_gripper=self.use_gripper)
        
        if succeeded:
            return "succeeded"
        else:
            return "aborted"


class ServoEndEffectorToPose(State):
    def __init__(self, robot, to_pose, pos_tolerance=.01, angular_tolerance=.1, frame=None):
        State.__init__(self, input_keys=['pose'], outcomes=['succeeded', 'preempted', 'aborted'])
        self.robot = robot
        self.to_pose = to_pose
        self.frame = frame
        self.pos_tolerance = pos_tolerance
        self.angular_tolerance = angular_tolerance

    def execute(self, userdata):
        target = self.to_pose
        if target is None:
            target = userdata["pose"]

        succeeded = self.robot.servo_to_pose(target, self.pos_tolerance, self.angular_tolerance, avoid_collisions=True)
        if succeeded:
            return "succeeded"
        else:
            return "aborted"


class ServoEndEffectorToOffset(State):
    def __init__(self, robot, offset, pos_tolerance=0.01, angular_tolerance=0.1, frame=None):
        State.__init__(self, input_keys=['offset'], outcomes=['succeeded', 'preempted', 'aborted'])
        self.robot = robot
        self.offset = offset
        self.offset_frame = frame
        self.pos_tolerance = pos_tolerance
        self.angular_tolerance = angular_tolerance

    def execute(self, userdata):
        offset = self.offset
        if offset is None:
            offset = userdata["offset"]
        offset_frame = self.offset_frame
        if offset_frame is None:
            offset_frame = 'arm_tool0'
        # In base_link by default
        current = self.robot.move_group.get_current_pose()
        target_pose = apply_offset_to_pose(current, offset, offset_frame, self.robot.tf2_buffer)
        succeeded = self.robot.servo_to_pose(target_pose, pos_tolerance=self.pos_tolerance, angular_tolerance=self.angular_tolerance, avoid_collisions=True)
        if succeeded:
            return "succeeded"
        else:
            return "aborted"


def robust_move_to_offset(robot, offset, frame=None):
    sm = StateMachine(["succeeded", "preempted", "aborted"])
    if frame is None:
        frame = "arm_tool0"
    with sm:
        StateMachine.add_auto("TRY_CARTESIAN_MOVE", MoveEndEffectorToOffset(robot, offset, frame, use_force=True, use_gripper=False), ["aborted"],
                              transitions={"succeeded": "succeeded"})
        # Generous allowances here to try and get some motion
        # StateMachine.add_auto("TRY_SERVO_MOVE", ServoEndEffectorToOffset(robot, offset, 0.02, 0.3, frame=frame), ["aborted"],
        #                       transitions={"succeeded": "succeeded"})
        # # HACK: Servo's underlying controller freaks out when reengaged after teach pendent intervention. Switch to follow_traj with a nonce goal
        # StateMachine.add_auto("SWITCH_TO_TRAJ_CONTROLLER", MoveEndEffectorToOffset(robot, (0,0,0), frame), ["succeeded", "aborted"],
        #                       transitions={"succeeded": "succeeded"})
        StateMachine.add_auto("ASK_HUMAN_TO_MOVE", interaction.AskForHumanAction(
            f"Please move the end effector by {offset} in the {frame} frame"), ["succeeded", "aborted"])
    return sm

def grasp_move_to_offset(robot, offset, fame=None):
    sm = StateMachine(["succeeded", "preempted", "aborted"])
    if frame is None:
        frame = "arm_tool0"
    with sm:
        StateMachine.add_auto("TRY_CARTESIAN_MOVE", MoveEndEffectorToOffset(robot, offset, frame, use_force=True, use_gripper=True), ["aborted"],
                              transitions={"succeeded": "succeeded"})
        # Generous allowances here to try and get some motion
        # StateMachine.add_auto("TRY_SERVO_MOVE", ServoEndEffectorToOffset(robot, offset, 0.02, 0.3, frame=frame), ["aborted"],
        #                       transitions={"succeeded": "succeeded"})
        # # HACK: Servo's underlying controller freaks out when reengaged after teach pendent intervention. Switch to follow_traj with a nonce goal
        # StateMachine.add_auto("SWITCH_TO_TRAJ_CONTROLLER", MoveEndEffectorToOffset(robot, (0,0,0), frame), ["succeeded", "aborted"],
        #                       transitions={"succeeded": "succeeded"})
        StateMachine.add_auto("ASK_HUMAN_TO_MOVE", interaction.AskForHumanAction(
            f"Please move the end effector by {offset} in the {frame} frame"), ["succeeded", "aborted"])
    return sm

class MoveEndEffectorInLineInOut(State):
    def __init__(self, robot):
        State.__init__(self, input_keys=['pose'], outcomes=['succeeded', 'preempted', 'aborted'])
        self.robot = robot

    def execute(self, userdata):
        target = self.robot.move_group.get_current_pose()
        target.pose.position.x += .1
        success = self.robot.servo_to_pose(target)
        if not success:
            rospy.logerr("Failed to move forward. Going to attempt to move back to original pose")
        rospy.sleep(1)

        target.pose.position.x -= .1
        success = self.robot.servo_to_pose(target)
        if not success:
            return "aborted"
        else:
            return "succeeded"


class CloseGripper(State):
    def __init__(self, robot, return_before_done=False):
        State.__init__(self, input_keys=[], outcomes=['succeeded', 'preempted', 'aborted'])
        self.robot = robot
        self.return_before_done = return_before_done

    def execute(self, ud):
        self.robot.close_gripper(return_before_done=self.return_before_done)
        return "succeeded"


class OpenGripper(State):
    def __init__(self, robot, return_before_done=False):
        State.__init__(self, input_keys=[], outcomes=['succeeded', 'preempted', 'aborted'])
        self.robot = robot
        self.return_before_done = return_before_done

    def execute(self, ud):
        self.robot.open_gripper(return_before_done=self.return_before_done)
        return "succeeded"



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
        self.robot.scene.attach_box("arm_tool0", "item", touch_links=["gripper_robotiq_arg2f_base_link", "gripper_left_distal_phalanx",
                                                 "gripper_left_proximal_phalanx", "gripper_right_proximal_phalanx",
                                                 "gripper_right_distal_phalanx", "gripper_left_bar", "gripper_right_bar"])
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


class AddCalibrationTargetInHandCollision(State):
    def __init__(self, robot):
        State.__init__(self, outcomes=["succeeded", "aborted"])
        self.robot = robot

    def execute(self, ud):
        self.robot.scene.add_box("calibration_target", PoseStamped(header=Header(frame_id="arm_tool0"),
                                                pose=Pose(position=Point(x=0, y=0, z=0.31), orientation=Quaternion(x=0, y=0, z=0, w=1))),
                            (.01, .18, .26))
        self.robot.scene.attach_box("arm_tool0", "calibration_target")
        start = rospy.get_time()
        seconds = rospy.get_time()
        timeout = 5.0
        while (seconds - start < timeout) and not rospy.is_shutdown():
            # Test if the box is in attached objects
            attached_objects = self.robot.scene.get_attached_objects(["calibration_target"])
            is_attached = len(attached_objects.keys()) > 0

            # Test if the box is in the scene.
            # Note that attaching the box will remove it from known_objects
            is_known = "calibration_target" in self.robot.scene.get_known_object_names()

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
        # FIXME: We should probably read this in from transforms or something
        POD_SIZE = .9398
        HALF_POD_SIZE = POD_SIZE / 2
        self.robot.scene.add_box("pod_top", PoseStamped(header=Header(frame_id="pod_base_link"),
                                                   pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=2.03),
                                                             orientation=I_QUAT)), (POD_SIZE, POD_SIZE, 1.5))

        self.robot.scene.add_box("pod_bottom", PoseStamped(header=Header(frame_id="pod_base_link"),
                                                      pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=.68),
                                                                orientation=I_QUAT)), (POD_SIZE, POD_SIZE, 1))
        self.robot.scene.add_box("pod_left", PoseStamped(header=Header(frame_id="pod_base_link"),
                                                    pose=Pose(position=Point(x=.75, y=.25, z=1.27),
                                                              orientation=I_QUAT)), (.5, .5, .3))
        self.robot.scene.add_box("pod_right", PoseStamped(header=Header(frame_id="pod_base_link"),
                                                     pose=Pose(position=Point(x=.25, y=.25, z=1.27),
                                                               orientation=I_QUAT)), (.5, .5, .3))

        start = rospy.get_time()
        seconds = rospy.get_time()
        timeout = 50.0
        while (seconds - start < timeout) and not rospy.is_shutdown():
            objects = self.robot.scene.get_objects()

            # Test if we are in the expected state
            if len(objects) == 4:
                return "succeeded"

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        return "aborted"
