import geometry_msgs.msg
import rospy
import math
from sensor_msgs.msg import JointState
from smach import State, StateMachine
from smach import State
from aurmr_tasks.util import apply_offset_to_pose

from robotiq_2f_gripper_control.msg import vacuum_gripper_input as VacuumGripperStatus
from moveit_msgs.msg import Constraints, JointConstraint
from aurmr_tasks.util import all_close, pose_dist


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
        if self.robot.close_to_joint_values(target, 0.01):
            rospy.loginfo("Skipping move attempt, as we are already close to the target configuration")
            return "succeeded"
        success = self.robot.move_to_joint_angles(target)
        if success:
            return "succeeded"
        else:
            return "aborted"

class MoveIntoJointLimits(State):
    def __init__(self, robot):
        State.__init__(self, input_keys=[], outcomes=['succeeded', 'aborted'])
        self.robot = robot

    def execute(self, ud):
        target_angle = math.pi*.9
        joint_angles = self.robot.get_current_joint_values()
        for i, angle in enumerate(joint_angles):
            if angle > math.pi:
                joint_angles[i] = target_angle
            elif angle < -math.pi:
                joint_angles[i] = -target_angle

        constraints = self.robot.move_group.get_path_constraints()
        print("constraints", constraints)
        constraints.joint_constraints = []
        for name in self.robot.commander.get_active_joint_names():
            if name == "arm_elbow_joint":
                continue
            new_constraint = JointConstraint()
            new_constraint.joint_name = name
            new_constraint.position = 0
            new_constraint.tolerance_above =  1.5*math.pi
            new_constraint.tolerance_below = 1.5*math.pi
            constraints.joint_constraints.append(new_constraint)
        print(constraints)
        success = self.robot.move_to_joint_angles(joint_angles, path_constraints=constraints)
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
        success = self.robot.move_to_pose(
                          pose,
                          allowed_planning_time=20.0,
                          execution_timeout=15.0,
                          num_planning_attempts=40,
                          orientation_constraint=None,
                          replan=True,
                          replan_attempts=20,
                          tolerance=0.02)
        if success:
            return "succeeded"
        else:
            return "aborted"

class MoveEndEffectorToPoseManipulable(State):
    def __init__(self, robot, default_pose=None):
        State.__init__(self, input_keys=['pose'], outcomes=['succeeded', 'preempted', 'aborted'])
        self.robot = robot
        self.default_pose = default_pose

    def execute(self, userdata):
        if self.default_pose:
            pose = self.default_pose
        else:
            pose = userdata["pose"]

        success = self.robot.move_to_pose_manipulable(
                          pose,
                          allowed_planning_time=15.0,
                          execution_timeout=15.0,
                          num_planning_attempts=20,
                          orientation_constraint=None,
                          replan=True,
                          replan_attempts=8,
                          tolerance=0.01)
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
        succeeded = self.robot.straight_move_to_pose(target_pose, ee_step = 0.01, avoid_collisions=True, jump_threshold=6.0, use_force=self.use_force, use_gripper=self.use_gripper)

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

class AdjustRightIfColumn1(State):
    def __init__(self, robot, offset, frame=None):
        State.__init__(self, input_keys=["target_bin_id"], outcomes=['succeeded', 'aborted', 'pass'])
        self.robot = robot
        self.offset = offset
        self.frame = frame

    def execute(self, ud):
        column_1 = ['1E', '1F', '1G', '1H']
        if ud['target_bin_id'] in column_1:
            sm = robust_move_to_offset(self.robot, self.offset, self.frame)
            outcome = sm.execute()
            return 'succeeded' if outcome == "succeeded" else 'aborted'
        else:
            return 'pass'

class AdjustLeftIfColumn4(State):
    def __init__(self, robot, offset, frame=None):
        State.__init__(self, input_keys=["target_bin_id"], outcomes=['succeeded', 'aborted', 'pass'])
        self.robot = robot
        self.offset = offset
        self.frame = frame

    def execute(self, ud):
        column_4 = ['4E', '4F', '4G', '4H']
        if ud['target_bin_id'] in column_4:
            sm = robust_move_to_offset(self.robot, self.offset, self.frame)
            outcome = sm.execute()
            return 'succeeded' if outcome == "succeeded" else 'aborted'
        else:
            return 'pass'

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
        # StateMachine.add_auto("ASK_HUMAN_TO_MOVE", interaction.AskForHumanAction(
        #     f"Please move the end effector by {offset} in the {frame} frame"), ["succeeded", "aborted"])
    return sm

def grasp_move_to_offset(robot, offset, frame=None):
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
        # StateMachine.add_auto("ASK_HUMAN_TO_MOVE", interaction.AskForHumanAction(
        #     f"Please move the end effector by {offset} in the {frame} frame"), ["succeeded", "aborted"])
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

class MoveToBin(State):
    def __init__(self, robot):
        State.__init__(self, input_keys=["target_bin_id"], outcomes=['succeeded', 'aborted'])
        self.robot = robot

    def execute(self, ud):
        target = ud["target_bin_id"]
        if isinstance(target, JointState):
            to_log = target.position
        else:
            target = "pre_bin_" + target.lower()
        to_log = target
        rospy.loginfo(f"Moving to {to_log}")
        success = self.robot.move_to_joint_angles(target, return_before_done=False)
        if success:
            return "succeeded"
        else:
            return "aborted"

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

class BlowOffGripper(State):
    def __init__(self, robot, return_before_done=False):
        State.__init__(self, input_keys=[], outcomes=['succeeded', 'preempted', 'aborted'])
        self.robot = robot
        self.return_before_done = return_before_done

    def execute(self, ud):
        self.robot.blow_off_gripper(return_before_done=self.return_before_done)
        return "succeeded"


class CheckGripperItem(State):
    def __init__(self, robot):
        State.__init__(self, input_keys=[], output_keys=[], outcomes=['item_detected', 'no_item_detected'])
        self.robot = robot

    def execute(self, ud):
        status = 'item_detected' if self.robot.check_gripper_item() else 'no_item_detected'
        return status

class ReleaseGripperIfNoItem(State):
    def __init__(self, robot):
        State.__init__(self, input_keys=[], output_keys=[], outcomes=['succeeded'])
        self.robot = robot

    def execute(self, ud):
        if not self.robot.check_gripper_item():
            self.robot.open_gripper(return_before_done=True)
        return 'succeeded'
