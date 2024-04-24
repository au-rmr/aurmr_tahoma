import copy
import geometry_msgs.msg
import rospy
import math
import actionlib
from sensor_msgs.msg import JointState
from smach import State, StateMachine
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from smach import State
from std_msgs.msg import Header
from aurmr_tasks.interaction import prompt_for_confirmation
from aurmr_tasks.util import apply_offset_to_pose
from aurmr_tasks import interaction
from aurmr_perception.util import I_QUAT, ROT_90_Z_QUAT
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, Bool
from geometry_msgs.msg import PoseStamped, WrenchStamped
from robotiq_2f_gripper_control.msg import vacuum_gripper_input as VacuumGripperStatus
from control_msgs.msg import FollowJointTrajectoryAction, GripperCommandAction, GripperCommandGoal
from moveit_msgs.msg import Constraints, JointConstraint
class MoveEndEffectorToPose_Storm(State):
    def __init__(self, default_pose=None):
        State.__init__(self, input_keys=['pose'], outcomes=['succeeded', 'preempted', 'aborted'])
        self.default_pose = default_pose
        # self.target_pose_visualizer = rospy.Publisher("end_effector_target", geometry_msgs.msg.PoseStamped, queue_size=1, latch=True)
        self.goal_finished = False

        self.GOAL_POSE = '/goal_pose'
        self.STROM_RESULT = '/storm_info/result'
        self.ACTIVATE_CONTROL = '/activate_control'
        rospy.Subscriber(self.STROM_RESULT, Bool, self.callback)
        self.goal_pub = rospy.Publisher(self.GOAL_POSE, geometry_msgs.msg.PoseStamped, queue_size=1)
        self.AC_pub = rospy.Publisher(self.ACTIVATE_CONTROL, Bool, queue_size=1)


    def callback(self, msg):
        self.goal_finished = msg.data
        #print("Get STROM_RESULT msg", self.goal_finished )

    def execute(self, userdata):
        if self.default_pose:
            pose = self.default_pose
        else:
            pose = userdata["pose"]
        # print("Start executing storm")
        # self.target_pose_visualizer.publish(pose)
        self.goal_finished = False

        while not(self.goal_finished):
            # print("The robot is still moving to: ", pose)
            self.goal_pub.publish(pose)
            self.AC_pub.publish(Bool(data=True))
            # print("Goal has not reached yet")
            rospy.sleep(1)

        self.AC_pub.publish(Bool(data=False))
        # rospy.loginfo("Finish a waypoint", pose)
        success = self.goal_finished
        if success:
            return "succeeded"
        else:
            return "aborted"

class MoveEndEffectorInLine_Storm(State):
    def __init__(self, start_pose=None, goal_pose=None, use_force=False, use_gripper=False):
        State.__init__(self,outcomes=['succeeded', 'preempted', 'aborted'])
        self.start_pose = start_pose
        self.goal_pose = goal_pose
        self.use_force = use_force
        self.use_gripper = use_gripper
        self.goal_finished = False
        self.object_detected = False
        self.force_msg = 0
        self.torque_msg = 0

        self.GOAL_POSE = '/goal_pose'
        self.STROM_RESULT = '/storm_info/result'
        self.ACTIVATE_CONTROL = '/activate_control'
        self.GRIPPER_ACTION_SERVER = '/gripper_controller/gripper_cmd'
        self.CURRENT_POSE = '/goal_pose/storm'

        self._gripper_client = actionlib.SimpleActionClient(self.GRIPPER_ACTION_SERVER, GripperCommandAction)
        self.goal_pub = rospy.Publisher(self.GOAL_POSE, PoseStamped, queue_size=1)
        self.AC_pub = rospy.Publisher(self.ACTIVATE_CONTROL, Bool, queue_size=1)
        self.goal_listener = rospy.Subscriber(self.STROM_RESULT, Bool, self.goal_finish_cb)
        self.wrench_listener = rospy.Subscriber("/wrench", WrenchStamped, self.wrench_cb)
        self.gripper_status_listener = rospy.Subscriber("/gripper_control/status", VacuumGripperStatus, self.gripper_status_cb)
        self.curr_pose_listener = rospy.Subscriber(self.CURRENT_POSE, PoseStamped, self.curr_pose_cb)

    def curr_pose_cb(self, msg: PoseStamped):
        if self.start_pose is None:
            self.start_pose = msg

    def close_gripper(self, return_before_done=False):
        goal = GripperCommandGoal()
        goal.command.position = 0.83
        goal.command.max_effort = 1
        self._gripper_client.send_goal(goal)
        # rospy.loginfo("Waiting for gripper" + str(return_before_done))
        if not return_before_done:
            rospy.loginfo("Waiting for gripper. \n")
            self._gripper_client.wait_for_result()

    def open_gripper(self, return_before_done=False):
        goal = GripperCommandGoal()
        goal.command.position = 0
        goal.command.max_effort = 1
        self._gripper_client.send_goal(goal)
        if not return_before_done:
            self._gripper_client.wait_for_result()

    def wrench_cb(self, msg: WrenchStamped):
        self.force_msg = math.sqrt(msg.wrench.force.x**2 + msg.wrench.force.y**2+ msg.wrench.force.z**2)
        self.torque_msg = math.sqrt(msg.wrench.torque.x**2 + msg.wrench.torque.y**2+ msg.wrench.torque.z**2)

    def gripper_status_cb(self, msg: VacuumGripperStatus):
        self.object_detected = (msg.gPO < 95)

    def goal_finish_cb(self, msg):
        self.goal_finished = msg.data
        #print("Get STROM_RESULT msg", self.goal_finished )

    def execute(self, userdata):
        if self.start_pose is None:
            print('Waiting for start_pose')
            rospy.sleep(0.5)
        poses = []
        diff_x = self.goal_pose.pose.position.x - self.start_pose.pose.position.x
        diff_y = self.goal_pose.pose.position.y - self.start_pose.pose.position.y
        diff_z = self.goal_pose.pose.position.z - self.start_pose.pose.position.z
        segment_num = int(((diff_x**2 + diff_y**2 + diff_z**2)**(1/2))/0.02)
        # print("number of dividen:", segment_num)
        for i in range(segment_num):
            pose = copy.deepcopy(self.start_pose)
            pose.pose.position.x += (i/segment_num)*diff_x
            pose.pose.position.y += (i/segment_num)*diff_y
            pose.pose.position.z += (i/segment_num)*diff_z
            poses.append(pose)
        poses.append(self.goal_pose)

        time_out = 5
        force_limit = 50
        if self.use_gripper:
            self.close_gripper(return_before_done=True)
        for pose in poses:
            self.goal_finished = False
            steps = 0.0
            while not(self.goal_finished):
                self.goal_pub.publish(pose)
                self.AC_pub.publish(Bool(data=True))
                if self.use_force and self.force_msg > force_limit:
                    self.AC_pub.publish(Bool(data=False))
                    rospy.loginfo("Stopping movement due to force feedback")
                    return "succeeded"
                elif self.use_gripper and self.object_detected:
                    self.AC_pub.publish(Bool(data=False))
                    rospy.loginfo("Stopping movement due to object detection")
                    return "succeeded"
                rospy.sleep(0.05)
                steps = steps + 0.05
                if steps>time_out:
                    # self.AC_pub.publish(Bool(data=False))
                    rospy.loginfo("Time_out in cartesian movement")
                    # early_stop = True
                    break

            self.AC_pub.publish(Bool(data=False))
            # print("Finish waypoint: ", pose)
        # if self.use_gripper:
        #     self.open_gripper(return_before_done=True)

        if self.goal_finished:
            return "succeeded"
        else:
            return "aborted"


class CloseGripperStorm(State):
    def __init__(self, return_before_done=False):
        State.__init__(self, outcomes=['succeeded', 'preempted', 'aborted'])
        self.return_before_done = return_before_done
        self.GRIPPER_ACTION_SERVER = '/gripper_controller/gripper_cmd'
        self._gripper_client = actionlib.SimpleActionClient(self.GRIPPER_ACTION_SERVER, GripperCommandAction)

    def close_gripper(self, return_before_done=False):
        goal = GripperCommandGoal()
        goal.command.position = 0.83
        goal.command.max_effort = 1
        self._gripper_client.send_goal(goal)
        # rospy.loginfo("Waiting for gripper" + str(return_before_done))
        if not return_before_done:
            rospy.loginfo("Waiting for gripper. \n")
            self._gripper_client.wait_for_result()


    def execute(self, ud):
        self.close_gripper(return_before_done=self.return_before_done)
        return "succeeded"


class OpenGripperStorm(State):
    def __init__(self, return_before_done=False):
        State.__init__(self, outcomes=['succeeded', 'preempted', 'aborted'])
        self.return_before_done = return_before_done
        self.GRIPPER_ACTION_SERVER = '/gripper_controller/gripper_cmd'
        self._gripper_client = actionlib.SimpleActionClient(self.GRIPPER_ACTION_SERVER, GripperCommandAction)

    def open_gripper(self, return_before_done=False):
        goal = GripperCommandGoal()
        goal.command.position = 0
        goal.command.max_effort = 1
        self._gripper_client.send_goal(goal)
        if not return_before_done:
            self._gripper_client.wait_for_result()

    def execute(self, ud):
        self.open_gripper(return_before_done=self.return_before_done)
        return "succeeded"

class MoveToBinHome(State):
    def __init__(self, robot):
        State.__init__(self, input_keys=["target_bin_id"], outcomes=['succeeded', 'aborted', 'pass'])
        self.robot = robot
        self.join_config_1f = 'home_1f'
        self.join_config_2f = 'home_2f'
        self.join_config_3f = 'home_3f'
        self.join_config_4f = 'home_4f'
        self.join_config_1g = 'home_1g'
        self.join_config_2g = 'home_2g'
        self.join_config_3g = 'home_3g'
        self.join_config_1h = 'home_1h'
        self.join_config_2h = 'home_2h'
        self.join_config_3h = 'home_3h'
        self.join_config_4h = 'home_4h'
        self.join_config_4g = 'home_4g'
        self.join_config_1e = 'home_1e'
        self.join_config_2e = 'home_2e'
        self.join_config_3e = 'home_3e'
        self.join_config_4e = 'home_4e'



    def execute(self, ud):
        if ud['target_bin_id'] == '1F':
            success = self.robot.move_to_joint_angles(self.join_config_1f)
            return 'succeeded' if success else 'aborted'
        elif ud['target_bin_id'] == '2F':
            success = self.robot.move_to_joint_angles(self.join_config_2f)
            return 'succeeded' if success else 'aborted'
        elif ud['target_bin_id'] == '3F':
            success = self.robot.move_to_joint_angles(self.join_config_3f)
            return 'succeeded' if success else 'aborted'
        elif ud['target_bin_id'] == '4F':
            success = self.robot.move_to_joint_angles(self.join_config_4f)
            return 'succeeded' if success else 'aborted'
        elif ud['target_bin_id'] == '1E':
            success = self.robot.move_to_joint_angles(self.join_config_1e)
            return 'succeeded' if success else 'aborted'
        elif ud['target_bin_id'] == '2E':
            success = self.robot.move_to_joint_angles(self.join_config_2e)
            return 'succeeded' if success else 'aborted'
        elif ud['target_bin_id'] == '3E':
            success = self.robot.move_to_joint_angles(self.join_config_3e)
            return 'succeeded' if success else 'aborted'
        elif ud['target_bin_id'] == '4E':
            success = self.robot.move_to_joint_angles(self.join_config_4e)
            return 'succeeded' if success else 'aborted'
        elif ud['target_bin_id'] == '1G':
            success = self.robot.move_to_joint_angles(self.join_config_1g)
            return 'succeeded' if success else 'aborted'
        elif ud['target_bin_id'] == '2G':
            success = self.robot.move_to_joint_angles(self.join_config_2g)
            return 'succeeded' if success else 'aborted'
        elif ud['target_bin_id'] == '3G':
            success = self.robot.move_to_joint_angles(self.join_config_3g)
            return 'succeeded' if success else 'aborted'
        elif ud['target_bin_id'] == '4G':
            success = self.robot.move_to_joint_angles(self.join_config_4g)
            return 'succeeded' if success else 'aborted'
        elif ud['target_bin_id'] == '1H':
            success = self.robot.move_to_joint_angles(self.join_config_1h)
            return 'succeeded' if success else 'aborted'
        elif ud['target_bin_id'] == '2H':
            success = self.robot.move_to_joint_angles(self.join_config_2h)
            return 'succeeded' if success else 'aborted'
        elif ud['target_bin_id'] == '3H':
            success = self.robot.move_to_joint_angles(self.join_config_3h)
            return 'succeeded' if success else 'aborted'
        elif ud['target_bin_id'] == '4H':
            success = self.robot.move_to_joint_angles(self.join_config_4h)
            return 'succeeded' if success else 'aborted'
        else:
            return 'pass'


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

class MoveIntoJointLimits(State):
    def __init__(self, robot):
        State.__init__(self, input_keys=[], outcomes=['succeeded', 'aborted'])
        self.robot = robot

    def execute(self, ud):
        target_angle = math.pi*.9
        joint_angles = self.robot.move_group.get_current_joint_values()
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

        self.target_pose_visualizer.publish(pose)
        success = self.robot.move_to_pose(
                          pose,
                          allowed_planning_time=15.0,
                          execution_timeout=15.0,
                          num_planning_attempts=20,
                          orientation_constraint=None,
                          replan=True,
                          replan_attempts=8,
                          tolerance=0.01)
        # input('check planning frame!!!!!!!!!!!!!!')
        if success:
            return "succeeded"
        else:
            return "aborted"

class MoveEndEffectorToPoseManipulable(State):
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
        State.__init__(self, input_keys=[], output_keys=["status"], outcomes=['item_detected', 'no_item_detected'])
        self.robot = robot

    def execute(self, ud):
        status = 'item_detected' if self.robot.check_gripper_item() else 'no_item_detected'
        print(status)
        ud["status"] = status
        return status

class ReleaseGripperIfNoItem(State):
    def __init__(self, robot):
        State.__init__(self, input_keys=[], output_keys=[], outcomes=['succeeded'])
        self.robot = robot

    def execute(self, ud):
        if not self.robot.check_gripper_item():
            self.robot.open_gripper(return_before_done=True)
        return 'succeeded'

class ClearCollisionGeometry(State):
    def __init__(self, robot):
        State.__init__(self, outcomes=["succeeded", "aborted"])
        self.robot = robot

    def execute(self, ud):
        print("CLEARING")
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
        self.robot.scene.add_box("item", PoseStamped(header=Header(frame_id="gripper_equilibrium_grasp"),
                                                pose=Pose(position=Point(x=0, y=0, z=0.03), orientation=Quaternion(x=0, y=0, z=0, w=1))),
                            (.03, .03, .08))
        self.robot.scene.attach_box("gripper_equilibrium_grasp", "item", touch_links=["gripper_robotiq_arg2f_base_link", "gripper_left_distal_phalanx",
                                                 "gripper_left_proximal_phalanx", "gripper_right_proximal_phalanx",
                                                 "gripper_right_distal_phalanx", "gripper_left_bar", "gripper_right_bar", "gripper_base_link", "epick_end_effector"])
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

# Original
# class AddPodCollisionGeometry(State):
#     def __init__(self, robot):
#         State.__init__(self, outcomes=["succeeded", "aborted"])
#         self.robot = robot

#     def execute(self, ud):
#         # FIXME: We should probably read this in from transforms or something
#         POD_SIZE = .9398
#         HALF_POD_SIZE = POD_SIZE / 2
#         self.robot.scene.add_box("pod_top", PoseStamped(header=Header(frame_id="pod_base_link"),
#                                                    pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=2.03),
#                                                              orientation=I_QUAT)), (POD_SIZE, POD_SIZE, 1.5))

#         self.robot.scene.add_box("pod_bottom", PoseStamped(header=Header(frame_id="pod_base_link"),
#                                                       pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=.68),
#                                                                 orientation=I_QUAT)), (POD_SIZE, POD_SIZE, 1))
#         self.robot.scene.add_box("pod_left", PoseStamped(header=Header(frame_id="pod_base_link"),
#                                                     pose=Pose(position=Point(x=.75, y=.25, z=1.27),
#                                                               orientation=I_QUAT)), (.5, .5, .3))
#         self.robot.scene.add_box("pod_right", PoseStamped(header=Header(frame_id="pod_base_link"),
#                                                      pose=Pose(position=Point(x=.25, y=.25, z=1.27),
#                                                                orientation=I_QUAT)), (.5, .5, .3))

#         start = rospy.get_time()
#         seconds = rospy.get_time()
#         timeout = 50.0
#         while (seconds - start < timeout) and not rospy.is_shutdown():
#             objects = self.robot.scene.get_objects()

#             # Test if we are in the expected state
#             if len(objects) == 4:
#                 return "succeeded"

#             # Sleep so that we give other threads time on the processor
#             rospy.sleep(0.1)
#             seconds = rospy.get_time()

#         # If we exited the while loop without returning then we timed out
#         return "aborted"


class AddPodCollisionGeometry(State):
    def __init__(self, robot):
        State.__init__(self, outcomes=["succeeded", "aborted"])
        self.robot = robot

    def execute(self, ud):
        # FIXME: We should probably read this in from transforms or something
        POD_SIZE = .9398
        HALF_POD_SIZE = POD_SIZE / 2
        WALL_WIDTH = 0.003
        # self.robot.scene.add_box("pod_top", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                            pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=2.03),
        #                                                      orientation=I_QUAT)), (POD_SIZE, POD_SIZE, 1.5))
        self.robot.scene.add_box("col_01", PoseStamped(header=Header(frame_id="pod_base_link"),
                                                      pose=Pose(position=Point(x=0.045, y=HALF_POD_SIZE, z=1.45),
                                                                orientation=I_QUAT)), (WALL_WIDTH, POD_SIZE, 2.3))
        # self.robot.scene.add_box("col_02", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=0.5*HALF_POD_SIZE, y=HALF_POD_SIZE, z=1.45),
        #                                                         orientation=I_QUAT)), (WALL_WIDTH, POD_SIZE, 2.3))
        # self.robot.scene.add_box("col_03", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=1.45),
        #                                                         orientation=I_QUAT)), (WALL_WIDTH, POD_SIZE, 2.3))
        # self.robot.scene.add_box("col_04", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=1.5*HALF_POD_SIZE, y=HALF_POD_SIZE, z=1.45),
        #                                                         orientation=I_QUAT)), (WALL_WIDTH, POD_SIZE, 2.3))
        self.robot.scene.add_box("col_05", PoseStamped(header=Header(frame_id="pod_base_link"),
                                                      pose=Pose(position=Point(x=POD_SIZE-0.045, y=HALF_POD_SIZE, z=1.45),
                                                                orientation=I_QUAT)), (WALL_WIDTH, POD_SIZE, 2.3))
        # self.robot.scene.add_box("row_01", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=0.34),
        #                                                         orientation=I_QUAT)), (POD_SIZE, POD_SIZE, WALL_WIDTH))
        # self.robot.scene.add_box("row_02", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=0.56),
        #                                                         orientation=I_QUAT)), (POD_SIZE, POD_SIZE, WALL_WIDTH))
        # self.robot.scene.add_box("row_03", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=0.70),
        #                                                         orientation=I_QUAT)), (POD_SIZE, POD_SIZE, WALL_WIDTH))
        # self.robot.scene.add_box("row_04", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=0.89),
        #                                                         orientation=I_QUAT)), (POD_SIZE, POD_SIZE, WALL_WIDTH))
        # self.robot.scene.add_box("row_05", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=1.04),
        #                                                         orientation=I_QUAT)), (POD_SIZE, POD_SIZE, WALL_WIDTH))
        # self.robot.scene.add_box("row_06", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=1.16),
        #                                                         orientation=I_QUAT)), (POD_SIZE, POD_SIZE, WALL_WIDTH))
        # self.robot.scene.add_box("row_07", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=1.38),
        #                                                         orientation=I_QUAT)), (POD_SIZE, POD_SIZE, WALL_WIDTH))
        # self.robot.scene.add_box("row_08", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=1.51),
        #                                                         orientation=I_QUAT)), (POD_SIZE, POD_SIZE, WALL_WIDTH))
        # self.robot.scene.add_box("row_09", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=1.66),
        #                                                         orientation=I_QUAT)), (POD_SIZE, POD_SIZE, WALL_WIDTH))
        # self.robot.scene.add_box("row_10", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=1.79),
        #                                                         orientation=I_QUAT)), (POD_SIZE, POD_SIZE, WALL_WIDTH))
        # self.robot.scene.add_box("row_11", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=2.01),
        #                                                         orientation=I_QUAT)), (POD_SIZE, POD_SIZE, WALL_WIDTH))
        # self.robot.scene.add_box("row_12", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=2.14),
        #                                                         orientation=I_QUAT)), (POD_SIZE, POD_SIZE, WALL_WIDTH))
        # self.robot.scene.add_box("row_13", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=2.29),
        #                                                         orientation=I_QUAT)), (POD_SIZE, POD_SIZE, WALL_WIDTH))
        # self.robot.scene.add_box("row_14", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=2.57),
        #                                                         orientation=I_QUAT)), (POD_SIZE, POD_SIZE, WALL_WIDTH))
        self.robot.scene.add_box("back_frame", PoseStamped(header=Header(frame_id="base_link"),
                                                    pose=Pose(position=Point(x=-0.28, y=0, z=1.27),
                                                              orientation=ROT_90_Z_QUAT)), (1.45, .2, 1.5))


        start = rospy.get_time()
        seconds = rospy.get_time()
        timeout = 50.0
        while (seconds - start < timeout) and not rospy.is_shutdown():
            objects = self.robot.scene.get_objects()

            # Test if we are in the expected state
            if len(objects) == 3:
                return "succeeded"

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        return "aborted"


class AddFullPodCollisionGeometryDropHide(State):
    def __init__(self, robot):
        State.__init__(self, outcomes=['succeeded', 'aborted'])
        self.robot = robot

    def execute(self, ud):
        # FIXME: We should probably read this in from transforms or something
        POD_SIZE = .9148
        HALF_POD_SIZE = POD_SIZE / 2
        WALL_WIDTH = 0.003
        SIDE_WALL_WIDTH = 0.033
        
        self.robot.scene.add_box("front_frame", PoseStamped(header=Header(frame_id="pod_base_link"),
                                                    pose=Pose(position=Point(x=POD_SIZE/2, y=0., z=1.34),
                                                              orientation=I_QUAT)), (1.8, .05, 3.0))
        
        self.robot.scene.add_box("left_side_frame", PoseStamped(header=Header(frame_id="base_link"),
                                                    pose=Pose(position=Point(x=0.25, y=1.00, z=1.34),
                                                              orientation=I_QUAT)), (2.00, .05, 3.0))
        
        self.robot.scene.add_box("right_side_frame", PoseStamped(header=Header(frame_id="base_link"),
                                                    pose=Pose(position=Point(x=0.25, y=-1.00, z=1.34),
                                                              orientation=I_QUAT)), (2.00, .05, 3.0))

        
        number_collision_box = 3

        start = rospy.get_time()
        seconds = rospy.get_time()
        timeout = 5.0
        while (seconds - start < timeout) and not rospy.is_shutdown():
            objects = self.robot.scene.get_objects()

            # Test if we are in the expected state
            if len(objects) == number_collision_box:
                return "succeeded"

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        return "aborted"


class AddFullPodCollisionGeometry(State):
    def __init__(self, robot,):
        State.__init__(self, input_keys=["target_bin_id"], outcomes=['succeeded', 'aborted'])
        self.robot = robot

    def execute(self, ud):
        # FIXME: We should probably read this in from transforms or something
        POD_SIZE = .9148
        HALF_POD_SIZE = POD_SIZE / 2
        WALL_WIDTH = 0.003
        SIDE_WALL_WIDTH = 0.033
        
        self.robot.scene.add_box("front_frame", PoseStamped(header=Header(frame_id="pod_base_link"),
                                                    pose=Pose(position=Point(x=POD_SIZE/2, y=0., z=1.34),
                                                              orientation=I_QUAT)), (1.8, .05, 3.0))
        
        self.robot.scene.add_box("left_side_frame", PoseStamped(header=Header(frame_id="base_link"),
                                                    pose=Pose(position=Point(x=0.25, y=1.00, z=1.34),
                                                              orientation=I_QUAT)), (2.00, .05, 3.0))
        
        self.robot.scene.add_box("right_side_frame", PoseStamped(header=Header(frame_id="base_link"),
                                                    pose=Pose(position=Point(x=0.25, y=-1.00, z=1.34),
                                                              orientation=I_QUAT)), (2.00, .05, 3.0))

        
        number_collision_box = 3
        print("target bin id", ud['target_bin_id'])
        try:
            import tf
            listener = tf.TransformListener()
            (trans, rot) = listener.lookupTransform('base_link', 'pod_bin_4h', rospy.Time())
            print("############################################################################", trans)
        except:
            try:
                bin_id = ud['target_bin_id']
                self.bin_2A = ['C', 'D', 'E']
                if(bin_id[1] in self.bin_2A):
                    self.bin_2A_heights = {'C':1.15, 'D':1.4, 'E':1.65}
                    self.bin_2A_widths = {'1':0.28, '2':-0.03, '3':-0.34}
                    number_collision_box = 6
                    z_coordinate = self.bin_2A_heights[bin_id[1]]
                    y_coordinate = self.bin_2A_widths[bin_id[0]]
                    # self.robot.scene.add_box("horizontal_plane", PoseStamped(header=Header(frame_id="base_link"),
                    #                                             pose=Pose(position=Point(x=0.75, y=0.0, z=z_coordinate),
                    #                                                     orientation=I_QUAT)), (0.5, 1.5, 0.02))
                    
                    # self.robot.scene.add_box("vertical_plane_1", PoseStamped(header=Header(frame_id="base_link"),
                    #                                             pose=Pose(position=Point(x=0.72, y=y_coordinate+0.18, z=1.2),
                    #                                                     orientation=I_QUAT)), (0.5, 0.01, 3.0))
                    
                    # self.robot.scene.add_box("vertical_plane_2", PoseStamped(header=Header(frame_id="base_link"),
                    #                                             pose=Pose(position=Point(x=0.72, y=y_coordinate-0.18, z=1.2),
                    #                                                     orientation=I_QUAT)), (0.5, 0.01, 3.0))
            except Exception as e:
                print(e)

        start = rospy.get_time()
        seconds = rospy.get_time()
        timeout = 5.0
        while (seconds - start < timeout) and not rospy.is_shutdown():
            objects = self.robot.scene.get_objects()

            # Test if we are in the expected state
            if len(objects) == number_collision_box:
                return "succeeded"

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        return "aborted"

class AddPartialPodCollisionGeometry(State):
    def __init__(self, robot):
        State.__init__(self, outcomes=["succeeded", "aborted"])
        self.robot = robot

    def execute(self, ud):
        # FIXME: We should probably read this in from transforms or something
        POD_SIZE = .9398
        HALF_POD_SIZE = POD_SIZE / 2
        WALL_WIDTH = 0.003
        SIDE_WALL_WIDTH = 0.033
        # self.robot.scene.add_box("pod_top", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                            pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=2.03),
        #                                                      orientation=I_QUAT)), (POD_SIZE, POD_SIZE, 1.5))
        self.robot.scene.add_box("col_01", PoseStamped(header=Header(frame_id="pod_base_link"),
                                                      pose=Pose(position=Point(x=0.0, y=HALF_POD_SIZE, z=1.45),
                                                                orientation=I_QUAT)), (SIDE_WALL_WIDTH, POD_SIZE, 2.3))

        self.robot.scene.add_box("col_05", PoseStamped(header=Header(frame_id="pod_base_link"),
                                                      pose=Pose(position=Point(x=POD_SIZE, y=HALF_POD_SIZE, z=1.45),
                                                                orientation=I_QUAT)), (SIDE_WALL_WIDTH, POD_SIZE, 2.3))

        self.robot.scene.add_box("back_frame", PoseStamped(header=Header(frame_id="base_link"),
                                                    pose=Pose(position=Point(x=-0.28, y=0, z=1.27),
                                                              orientation=ROT_90_Z_QUAT)), (1.45, .2, 1.5))

        # self.robot.scene.add_box("upper_pods_outer", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE-0.1, z=1.66+0.5),
        #                                                         orientation=I_QUAT)), (POD_SIZE, POD_SIZE, 1))

        start = rospy.get_time()
        seconds = rospy.get_time()
        timeout = 50.0
        while (seconds - start < timeout) and not rospy.is_shutdown():
            objects = self.robot.scene.get_objects()

            # Test if we are in the expected state
            if len(objects) == 3:
                return "succeeded"

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        return "aborted"
