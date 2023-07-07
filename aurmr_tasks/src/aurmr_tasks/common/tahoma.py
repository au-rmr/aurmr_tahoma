import copy
import math
import sys
import numpy as np
from functools import wraps
from turtle import pos
import std_msgs

from actionlib import SimpleActionClient
from actionlib_msgs.msg import GoalStatusArray, GoalStatus
from control_msgs.msg import FollowJointTrajectoryAction, GripperCommandAction, GripperCommandGoal, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
import tf2_ros

#import tf2_geometry_msgs

import actionlib
import control_msgs.msg
import trajectory_msgs.msg
import rospy

from tf2_geometry_msgs import from_msg_msg
from geometry_msgs.msg import PoseStamped, WrenchStamped, Quaternion, Pose, Point
# from robotiq_2f_gripper_control.msg import vacuum_gripper_input arStatus
from aurmr_tasks.util import all_close, pose_dist
from moveit_msgs.msg import MoveItErrorCodes, MoveGroupAction, DisplayTrajectory, RobotState
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from sensor_msgs.msg import JointState

import moveit_commander
from controller_manager_msgs.srv import ListControllers, SwitchController
from tahoma_moveit_config.msg import ServoToPoseAction, ServoToPoseGoal



ARM_GROUP_NAME = 'manipulator'
JOINT_ACTION_SERVER = '/pos_joint_traj_controller/follow_joint_trajectory'
JOINT_GROUP_CONTROLLER = 'joint_group_pos_controller'
JOINT_TRAJ_CONTROLLER = 'scaled_pos_joint_traj_controller'
JOINT_TRAJ_CONTROLLER_SIM = 'pos_joint_traj_controller'
GRIPPER_ACTION_SERVER = '/gripper_controller/gripper_cmd'
MOVE_GROUP_ACTION_SERVER = 'move_group'
TIME_FROM_START = 5

CONFLICTING_CONTROLLERS = [JOINT_TRAJ_CONTROLLER, JOINT_TRAJ_CONTROLLER_SIM, JOINT_GROUP_CONTROLLER]

MOVE_IT_ERROR_TO_STRING = {
    MoveItErrorCodes.SUCCESS: "SUCCESS",
    MoveItErrorCodes.FAILURE: 'FAILURE',
    MoveItErrorCodes.PLANNING_FAILED: 'PLANNING_FAILED',
    MoveItErrorCodes.INVALID_MOTION_PLAN: 'INVALID_MOTION_PLAN',
    MoveItErrorCodes.MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE: 'MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE',
    MoveItErrorCodes.CONTROL_FAILED: 'CONTROL_FAILED',
    MoveItErrorCodes.UNABLE_TO_AQUIRE_SENSOR_DATA: 'UNABLE_TO_AQUIRE_SENSOR_DATA',
    MoveItErrorCodes.TIMED_OUT: 'TIMED_OUT',
    MoveItErrorCodes.PREEMPTED: 'PREEMPTED',
    MoveItErrorCodes.START_STATE_IN_COLLISION: 'START_STATE_IN_COLLISION',
    MoveItErrorCodes.START_STATE_VIOLATES_PATH_CONSTRAINTS: 'START_STATE_VIOLATES_PATH_CONSTRAINTS',
    MoveItErrorCodes.GOAL_IN_COLLISION: 'GOAL_IN_COLLISION',
    MoveItErrorCodes.GOAL_VIOLATES_PATH_CONSTRAINTS: 'GOAL_VIOLATES_PATH_CONSTRAINTS',
    MoveItErrorCodes.GOAL_CONSTRAINTS_VIOLATED: 'GOAL_CONSTRAINTS_VIOLATED',
    MoveItErrorCodes.INVALID_GROUP_NAME: 'INVALID_GROUP_NAME',
    MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS: 'INVALID_GOAL_CONSTRAINTS',
    MoveItErrorCodes.INVALID_ROBOT_STATE: 'INVALID_ROBOT_STATE',
    MoveItErrorCodes.INVALID_LINK_NAME: 'INVALID_LINK_NAME',
    MoveItErrorCodes.INVALID_OBJECT_NAME: 'INVALID_OBJECT_NAME',
    MoveItErrorCodes.FRAME_TRANSFORM_FAILURE: 'FRAME_TRANSFORM_FAILURE',
    MoveItErrorCodes.COLLISION_CHECKING_UNAVAILABLE: 'COLLISION_CHECKING_UNAVAILABLE',
    MoveItErrorCodes.ROBOT_STATE_STALE: 'ROBOT_STATE_STALE',
    MoveItErrorCodes.SENSOR_INFO_STALE: 'SENSOR_INFO_STALE',
    MoveItErrorCodes.NO_IK_SOLUTION: 'NO_IK_SOLUTION',
}


def requires_controller(named):
    """
    Args:
        named:
    Returns: a decorator factory that replaces an invoked function
    with a wrapped function call, ensuring that the correct controller
    is activated before the call.
    """
    def decorator(function):
        # Use this functools helper to make docstrings of the passed in function get bubbled up
        @wraps(function)
        def wrapper(self, *args, **kwargs):
            if not self.is_controller_active(named):
                activated = self.activate_controller(named)
                if not activated:
                    return Falsecen
            return function(self, *args, **kwargs)
        return wrapper
    return decorator


def moveit_error_string(val):
    """Returns a string associated with a MoveItErrorCode.
    Args:
        val: The val field from moveit_msgs/MoveItErrorCodes.msg
    Returns: The string associated with the error value, 'UNKNOWN_ERROR_CODE'
        if the value is invalid.
    """
    return MOVE_IT_ERROR_TO_STRING.get(val, 'UNKNOWN_ERROR_CODE')


class Tahoma:
    def __init__(self, in_sim=False):
        self.in_sim = in_sim
        self.joint_state = None
        self.point_cloud = None

        self.tf2_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer)
        self.wrist_position = None
        self.lift_position = None

        # self.joint_states_subscriber = rospy.Subscriber('/joint_states', JointState, self.joint_states_callback)
        self._joint_traj_client = actionlib.SimpleActionClient(
            JOINT_ACTION_SERVER, control_msgs.msg.FollowJointTrajectoryAction)
        self._gripper_client = actionlib.SimpleActionClient(GRIPPER_ACTION_SERVER, GripperCommandAction)
        server_reached = self._joint_traj_client.wait_for_server(rospy.Duration(10))
        if not server_reached:
            print('Unable to connect to arm action server. Timeout exceeded. Exiting...')
            rospy.signal_shutdown('Unable to connect to arm action server. Timeout exceeded.')
            sys.exit()
        self._move_group_client = actionlib.SimpleActionClient(
            MOVE_GROUP_ACTION_SERVER, MoveGroupAction)
        self._move_group_client.wait_for_server(rospy.Duration(10))
        self._compute_ik = rospy.ServiceProxy('compute_ik', GetPositionIK)
        moveit_commander.roscpp_initialize(sys.argv)
        self.commander = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface(synchronous=True)
        self.move_group = moveit_commander.MoveGroupCommander(ARM_GROUP_NAME)
        self.MAX_VEL_FACTOR = .6
        self.MAX_ACC_FACTOR = .6
        self.move_group.set_max_velocity_scaling_factor(self.MAX_VEL_FACTOR)
        self.move_group.set_max_acceleration_scaling_factor(self.MAX_ACC_FACTOR)
        self.display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            DisplayTrajectory,
            queue_size=1,
            latch=True
        )
        self._controller_lister = rospy.ServiceProxy("/controller_manager/list_controllers", ListControllers)
        self._controller_switcher = rospy.ServiceProxy("/controller_manager/switch_controller", SwitchController)
        self.servo_to_pose_client = SimpleActionClient("/servo_server/servo_to_pose", ServoToPoseAction)

        self.grasp_pose_pub = rospy.Publisher("~grasp_pose", PoseStamped, queue_size=1)

        planning_frame = self.move_group.get_planning_frame()
        # print(f"\nplanning_frame: {planning_frame}")
        eef_link = self.move_group.get_end_effector_link() # "arm_tool0" by default, i.e. robot flange
        # print(f"eef_link: {eef_link}")
        self.move_group.set_end_effector_link("epick_end_effector")
        eef_link = self.move_group.get_end_effector_link()
        group_names = self.commander.get_group_names()

        self.wrench_listener = rospy.Subscriber("/wrench", WrenchStamped, self.wrench_cb)
        # self.gripper_status_listener = rospy.Subscriber("/gripper_control/status", VacuumGripperStatus, self.gripper_status_cb)
        self.traj_status_listener = rospy.Subscriber("/scaled_pos_joint_traj_controller/follow_joint_trajectory/status", GoalStatusArray, self.goal_status_cb)
        self.force_mag = 0
        self.torque_mag = 0
        self.object_detected = False
        self.goal_finished = True
        self.goal_stamp = 0
        # Misc variables
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names        

        self.active_controllers = None
        self.update_running_controllers()

        # Wrong pod1_bin_heights_inch! Where did you get them??? 
        # Official Amazon recipe has different numbers (in mm)!!!
        # pod1_bin_heights_inch = [8.75, 5.5, 7.5, 6, 4.5, 8.75, 5, 6, 5, 8.75, 5, 6, 11]
        
        # pod1_bin_heights_m are from pod1_recipe_790-00265-021_2.txt "bindDimensions" : A1-A13
        # "ligHeight" : 26, but in reality 32
        pod1_bin_heights_m = [0.223, 0.146, 0.197, 0.146, 0.121, 0.223, 0.121, 0.146, 0.121, 0.223, 0.121, 0.146, 0.261]

        # pod2_bin_heights_m are from pod2_recipe_790-00265-022_2.txt "bindDimensions" : A1-A8
        # Original Values (mm): [346, 295, 244, 219, 244, 269, 320, 257]
        # with minor updates to correspond the reality
        # "ligHeight" : 26, but in reality 32
        pod2_bin_heights_m = [0.346, 0.295, 0.250, 0.225, 0.250, 0.275, 0.325, 0.275]

        self.pod_sizes = dict(
            pod1_brace_frame_width = 0.033,
            pod1_base_frame_width = 0.041,
            pod1_base_to_brace_XY_offset = 0.010,
            pod1_base_width = 0.9398,
            pod1_brace_width = 0.9188,
            pod1_base_height  = 0.295, # Real value 0.257
            pod1_brace_height  = 2.311,
            pod1_brace_frame_thickness = 0.005,
            pod1_bin_heights = pod1_bin_heights_m,
            pod1_bin_depth = 0.152,
            pod1_bin_width = 0.9188 / 4,
            pod1_bin_wall_thickness = 0.002,
            pod1_bin_bottom_thickness = 0.007, # 5mm + 2mm
            pod1_bin_flap_height = 0.032,

            pod2_brace_frame_width = 0.033,
            pod2_base_frame_width = 0.041,
            pod2_base_to_brace_XY_offset = 0.010,
            pod2_base_width = 0.9398,
            pod2_brace_width = 0.9188,
            pod2_base_height  = 0.295, # Real value 0.257
            pod2_brace_height  = 2.311,
            pod2_brace_frame_thickness = 0.005,
            pod2_bin_heights = pod2_bin_heights_m,
            pod2_bin_depth = 0.356,
            pod2_bin_width = 0.9188 / 3,
            pod2_bin_wall_thickness = 0.002,
            pod2_bin_bottom_thickness = 0.009, # 7mm + 2mm
            pod2_bin_flap_height = 0.032
            )
        
        self.end_effector = dict(
            robotiq_epick_dimX = 0.0750,
            robotiq_epick_dimY = 0.0830,
            robotiq_epick_dimZ = 0.1047,
            robotiq_epick_finH = 0.0024,
            robotiq_epick_earH = 0.0125,
            robotiq_epick_earR = 0.0059,
            epick_cylinder_H   = 0.0900,
            epick_cylinder_R   = 0.0059,
            epick_end_effect_H = 0.0100,
            epick_end_effect_R = 0.0225,
            coupling_D         = 0.0700,
            coupling_H         = 0.0200,
            coupling_h         = 0.0030
            )
        

        # self.move_to_joint_angles(joints='tote_approach', startpoint='current')

        # result is the same as for self.commander.get_current_state(), which is more correct because 
        # self.commader has get_current_state() function, self.move_group not
        # still, the same results - strange...
        # self.home_state = self.move_group.get_current_state()

        self.home_state = RobotState() # 'tote_approach' - object dropping position
        self.home_state.joint_state.header.stamp = rospy.Time.now()
        self.home_state.joint_state.header.frame_id = 'base_link'
        # ['arm_shoulder_pan_joint', 'arm_shoulder_lift_joint', 'arm_elbow_joint', 'arm_wrist_1_joint', 'arm_wrist_2_joint', 'arm_wrist_3_joint']
        self.home_state.joint_state.name = self.move_group.get_active_joints()
        self.home_state.joint_state.position = [2.86, -1.47, 1.58, 4.75, 1.59, 4.83] # the order is the same as joint_state.name 
        # self.move_to_joint_angles(joints=self.home_state.joint_state, startpoint='current')


    def wrench_cb(self, msg: WrenchStamped):
        self.force_mag = math.sqrt(msg.wrench.force.x**2 + msg.wrench.force.y**2+ msg.wrench.force.z**2)
        self.torque_mag = math.sqrt(msg.wrench.torque.x**2 + msg.wrench.torque.y**2+ msg.wrench.torque.z**2)


    # def gripper_status_cb(self, msg: VacuumGripperStatus):
    #     self.object_detected = (msg.gPO < 95)


    def goal_status_cb(self, msg: GoalStatusArray):
        latest_time = 0
        latest_status = GoalStatus.SUCCEEDED
        for g in msg.status_list:
            new_stamp = g.goal_id.stamp.secs + g.goal_id.stamp.nsecs*10**(-9)
            if new_stamp > latest_time:
                latest_time = new_stamp
                latest_status = g.status
        self.goal_finished = latest_status != GoalStatus.PENDING and latest_status != GoalStatus.ACTIVE
        self.goal_stamp = latest_time
        # home_state_joint_positions = [-1.01, -0.46, -1.62, -1.05, -5.26, 3.14]


    def update_running_controllers(self):
        controllers_status = self._controller_lister().controller
        self.active_controllers = []
        for controller in controllers_status:
            if controller.state == "running":
                self.active_controllers.append(controller.name)


    def wait_for_controllers(self, timeout):
        rate = rospy.Rate(1)
        try_until = rospy.Time.now() + rospy.Duration(timeout)
        while not rospy.is_shutdown() and rospy.Time.now() < try_until:
            controllers_status = self._controller_lister().controller
            for controller in controllers_status:
                is_active = controller.state == "running"
                if not is_active:
                    continue
                if self.in_sim and controller.name == JOINT_TRAJ_CONTROLLER_SIM:
                    return True
                elif not self.in_sim and controller.name == JOINT_TRAJ_CONTROLLER:
                    return True
            rate.sleep()
        return False


    def activate_controller(self, named):
        # Hacked to work for what we need right now
        to_enable = named
        if named == JOINT_TRAJ_CONTROLLER and self.in_sim:
            to_enable = JOINT_TRAJ_CONTROLLER_SIM
        to_disable = []
        for candidate in CONFLICTING_CONTROLLERS:
            if candidate == named:
                continue
            # Only disable conflicting controllers that are enabled
            if candidate in self.active_controllers:
                to_disable.append(candidate)
        if named not in CONFLICTING_CONTROLLERS:
            raise RuntimeError(f"Asked to activate unknown controller {named}")
        rospy.loginfo(f"Activating {named} and disabling {to_disable}")
        ok = self._controller_switcher.call([to_enable], to_disable, 1, False, 2.0)
        if not ok:
            rospy.logerr("Couldn't activate controller")
            return False
            # FIXME(nickswalker): We should crash out here but this exception seems to get swallowed
            # raise RuntimeError(f"Failed to enable controller {named}")
        else:
            return True


    def is_controller_active(self, named):
        self.update_running_controllers()
        to_check = named
        if named == JOINT_TRAJ_CONTROLLER and self.in_sim:
            to_check = JOINT_TRAJ_CONTROLLER_SIM
        return to_check in self.active_controllers


    def open_gripper(self, return_before_done=False):
        goal = GripperCommandGoal()
        goal.command.position = 0
        goal.command.max_effort = 1
        self._gripper_client.send_goal(goal)
        if not return_before_done:
            self._gripper_client.wait_for_result()


    def check_gripper_item(self):
        return self.object_detected 


    def close_gripper(self, return_before_done=False):
        goal = GripperCommandGoal()
        goal.command.position = 0.83
        goal.command.max_effort = 1
        self._gripper_client.send_goal(goal)
        rospy.loginfo("Waiting for gripper" + str(return_before_done))
        if not return_before_done:
            rospy.loginfo("Waiting for gripper. \n")
            self._gripper_client.wait_for_result()


    @requires_controller(JOINT_TRAJ_CONTROLLER)
    def move_to_pose_unsafe(self, pose, return_before_done=False):
        joint_names = [key for key in pose]
        point = JointTrajectoryPoint()
        point.time_from_start = rospy.Duration(0, 1)

        trajectory_goal = FollowJointTrajectoryGoal()
        trajectory_goal.goal_time_tolerance = rospy.Time(1)
        trajectory_goal.trajectory.joint_names = joint_names

        joint_positions = [pose[key] for key in joint_names]
        point.positions = joint_positions
        trajectory_goal.trajectory.points = [point]

        trajectory_goal.trajectory.header.stamp = rospy.Time(0)
        self._joint_traj_client.send_goal(trajectory_goal)
        if not return_before_done:
            self._joint_traj_client.wait_for_result()
            # print('Received the following result:')
            # print(self.trajectory_client.get_result())


    @requires_controller(JOINT_TRAJ_CONTROLLER)
    def move_to_joint_angles_unsafe(self, joint_state):
        """
        Moves to an ArmJoints configuration
        :param joint_state: an ArmJoints instance to move to
        :return:
        """
        goal = control_msgs.msg.FollowJointTrajectoryGoal()
        goal.trajectory.joint_names.extend(self.move_group.get_active_joints())
        point = trajectory_msgs.msg.JointTrajectoryPoint()
        point.positions.extend(joint_state.values())
        point.time_from_start = rospy.Duration(TIME_FROM_START)
        goal.trajectory.points.append(point)
        self._joint_traj_client.send_goal(goal)
        self._joint_traj_client.wait_for_result(rospy.Duration(10))


    def get_joint_values_for_name(self, name):
        self.move_group.set_named_target(name)
        values = self.move_group.get_joint_value_target()
        return values


    @requires_controller(JOINT_TRAJ_CONTROLLER)
    def move_to_joint_angles(self,
                           joints,
                           allowed_planning_time=10.0,
                              # home_state_joint_positions = [-1.01, -0.46, -1.62, -1.05, -5.26, 3.14]
     execution_timeout=15.0,
                           num_planning_attempts=20,
                           plan_only=False,
                           replan=False,
                           replan_attempts=5,
                           joint_tolerance=0.01,
                           startpoint = 'home'):
        """ Moves the end-effector to a pose, using motion planning.
        
        Args:
            joints: dict("joint_name":joint_value)
                    list(joint_values)
                    JointState. 
                    string "name" of a stored configuration [self.move_group.remember_joint_values("name", values)]
                        values = self.move_group.get_current_joint_values() (= list of joint values) if no values provided
                    Target joint angles
            allowed_planning_time: float. 
                                   The maximum duration to wait for a planning result.
            execution_timeout: float. 
                               The maximum duration to wait for an arm motion to execute 
                               (or for planning to fail completely), in seconds.
            group_name: string.
            num_planning_attempts: int. 
                                   The number of times to compute the same plan. 
                                   The shortest path is ultimately used. 
                                   For random planners, this can help get shorter, less weird paths.
            plan_only: bool. 
                       If True, then this method does not execute the plan on the robot. 
                       Useful for determining whether this is likely to succeed.
            replan: bool. 
                    If True, then if an execution fails (while the arm is moving), then come up with a new plan and execute it.
            replan_attempts: int. 
                             How many times to replan if the execution fails.
            joint_tolerance: float. 
                             The goal joint tolerance, in radians.
        
        Returns:
            string describing the error if an error occurred, else None. """
        self.move_group.stop()

        self.move_group.set_num_planning_attempts(num_planning_attempts)
        self.move_group.allow_replanning(replan)
        self.move_group.set_goal_joint_tolerance(joint_tolerance)
        self.move_group.set_planning_time(allowed_planning_time)
        if startpoint == 'home':
            self.move_group.set_start_state(self.home_state)
        else:
            self.move_group.set_start_state_to_current_state()

        if isinstance(joints, str):
            joint_values = self.get_joint_values_for_name(joints)
            self.move_group.set_named_target(joints)
        elif isinstance(joints, JointState):
            joint_values = list(joints.position)
            # joint_names = list(joints.name)
            # joint_states = dict(zip(joint_names, joint_values))
            # print(f"joint_states:\n{joint_states}")
            self.move_group.set_joint_value_target(joints)
        elif isinstance(joints, list):
            self.move_group.set_joint_value_target(joints)
            joint_values = joints
        
        # self.move_group.go(wait=True)
        success, plan, planning_time, error_code = self.move_group.plan()
        if not success:
            return False

        # # A `DisplayTrajectory`_ msg has two primary fields, trajectory_start and trajectory.
        # # We populate the trajectory_start with our current robot state to copy over
        # # any AttachedCollisionObjects and add our plan to the trajectory.
        # display_trajectory = DisplayTrajectory()
        # display_trajectory.trajectory_start = self.home_state if startpoint == 'home' else self.commander.get_current_state()
        # display_trajectory.trajectory.append(plan)
        # # Publish
        # self.display_trajectory_publisher.publish(display_trajectory)

        # Now, we call the planner to compute the plan and execute it.
        ret = self.move_group.execute(plan, wait=True)

        # Calling `stop()` ensures that there is no residual movement
        self.move_group.stop()

        # print(f"\ngoal_joint_values: {joint_values}")
        current_joints = self.move_group.get_current_joint_values()
        # print(f"current_joint_values: {current_joints}")
        return all_close(joint_values, current_joints, joint_tolerance=joint_tolerance)


    @requires_controller(JOINT_TRAJ_CONTROLLER)
    def move_to_pose(self,
                          pose_stamped,
                          allowed_planning_time=10.0,
                          execution_timeout=15.0,
                          num_planning_attempts=20,
                          orientation_constraint=None,
                          replan=True,
                          replan_attempts=5,
                          position_tolerance=0.001,
                          orientation_tolerance=0.01,
                          startpoint = 'home'):
        """ Moves the end-effector to a pose, using motion planning.
        
        Args:
            pose: geometry_msgs/PoseStamped. 
                  The goal pose for the gripper.
            allowed_planning_time: float. 
                                   The maximum duration to wait for a planning result.
            execution_timeout: float. 
                               The maximum duration to wait for an arm motion to execute 
                               (or for planning to fail completely), in seconds.
            num_planning_attempts: int. 
                                   The number of times to compute the same plan. 
                                   The shortest path is ultimately used. 
                                   For random planners, this can help get shorter, less weird paths.
            orientation_constraint: moveit_msgs/OrientationConstraint. 
                                    An orientation constraint for the entire path.
            replan: bool. 
                    If True, then if an execution fails (while the arm is moving), 
                    then come up with a new plan and execute it.
            replan_attempts: int. 
                             How many times to replan if the execution fails.
            position_tolerance:    float. 
                                   The goal position tolerance, in meters.
            orientation_tolerance: float. 
                                   The goal orientation tolerance, in radians. 1 degree = 0.017 rad
            startpoint: string. 
                       The startpoint for motion planning. If any other than 'home', current robot state will be startpoint
        
        Returns:
            string describing the error if an error occurred, else None. """

        pose_stamped = from_msg_msg(pose_stamped)
        # print(f"\npose_stamped:\n{pose_stamped.pose}")
        goal_in_planning_frame = self.tf2_buffer.transform(pose_stamped, self.planning_frame,
                                       rospy.Duration(1))
        # print(f"\ngoal_in_planning_frame:\n{goal_in_planning_frame.pose}")

        self.move_group.set_end_effector_link("epick_end_effector")
        self.move_group.set_pose_target(pose_stamped)
        self.move_group.set_planning_time(allowed_planning_time)
        self.move_group.set_num_planning_attempts(num_planning_attempts)
        self.move_group.allow_replanning(replan)
        self.move_group.set_goal_position_tolerance(position_tolerance)
        self.move_group.set_goal_orientation_tolerance(orientation_tolerance)
        if startpoint == 'home':
            self.move_group.set_start_state(self.home_state)
        else:
            self.move_group.set_start_state_to_current_state()

        success, plan, planning_time, error_code = self.move_group.plan()
        if not success:
            return False

        # # A `DisplayTrajectory`_ msg has two primary fields, trajectory_start and trajectory.
        # # We populate the trajectory_start with our current robot state to copy over
        # # any AttachedCollisionObjects and add our plan to the trajectory.
        # display_trajectory = DisplayTrajectory()
        # display_trajectory.trajectory_start = self.home_state if startpoint == 'home' else self.commander.get_current_state()
        # display_trajectory.trajectory.append(plan)
        # # Publish
        # self.display_trajectory_publisher.publish(display_trajectory)

        # Now, we call the planner to compute the plan and execute it.
        ret = self.move_group.execute(plan, wait=True)
        # Calling `stop()` ensures that there is no residual movement
        self.move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        self.move_group.clear_pose_targets()

        current_pose = self.move_group.get_current_pose()
        # print(f"\ncurrent_pose:\n{current_pose.pose}")
        current_joint_values = self.move_group.get_current_joint_values()
        # print(f"\ncurrent_joint_values: {current_joint_values}")
        # print(f"pose_dist: {pose_dist(goal_in_planning_frame, current_pose)}")
        return all_close(goal_in_planning_frame, current_pose, 
                         position_tolerance=position_tolerance, 
                         orientation_tolerance=orientation_tolerance)


    @requires_controller(JOINT_TRAJ_CONTROLLER)
    def straight_move_to_pose(self,
                              pose_stamped,
                              tolerance=0.01,
                              ee_step=0.001,
                              jump_threshold=6.0,
                              avoid_collisions=True,
                              use_force=False,
                              use_gripper=False):
        """
        Moves the end-effector to a pose in a straight line.
        
        Args:
          pose_stamped: geometry_msgs/PoseStamped. The goal pose for the
            gripper.
          ee_step: float. The distance in meters to interpolate the path.
          jump_threshold: float. The maximum allowable distance in the arm's
            configuration space allowed between two poses in the path. Used to
            prevent "jumps" in the IK solution.
          avoid_collisions: bool. Whether to check for obstacles or not.
        
        Returns:
            string describing the error if an error occurred, else None.
        """
 
        self.move_group.set_end_effector_link("epick_end_effector")
        goal_in_planning_frame = self.tf2_buffer.transform(pose_stamped, self.planning_frame, rospy.Duration(1))

        self.grasp_pose_pub.publish(pose_stamped)

        waypoints = [goal_in_planning_frame.pose]
        (plan, fraction) = self.move_group.compute_cartesian_path(
            waypoints, ee_step, jump_threshold, avoid_collisions
        )
        plan = self.move_group.retime_trajectory(self.move_group.get_current_state(), plan, velocity_scaling_factor=.05, acceleration_scaling_factor=.05)
        if fraction < .9:
            rospy.logwarn(f"Not moving in cartesian path. Only {fraction} waypoints reached")
            # return False

        wait = not (use_force or use_gripper)
        rospy.loginfo("Waiting?: " + str(wait))
        old_goal_stamp = self.goal_stamp

        ret = self.move_group.execute(plan, wait=wait)
        if use_gripper:
            self.close_gripper(return_before_done=True)

        if not wait:        # # A `DisplayTrajectory`_ msg has two primary fields, trajectory_start and trajectory.
        # # We populate the trajectory_start with our current robot state to copy over
        # # any AttachedCollisionObjects and add our plan to the trajectory.
        # display_trajectory = DisplayTrajectory()
        # display_trajectory.trajectory_start = self.home_state if startpoint == 'home' else self.commander.get_current_state()
        # display_trajectory.trajectory.append(plan)
        # # Publish
        # self.display_trajectory_publisher.publish(display_trajectory)
            timeout = 1
            steps = 0
            while steps < timeout and (self.goal_stamp == old_goal_stamp):
                rospy.loginfo("Waiting for moveit goal to update:" + str(old_goal_stamp) + " " + str(self.goal_stamp))
                rospy.sleep(.01)
                steps = steps + .01


            force_limit = 50
            early_stop = False
            rospy.loginfo("Waiting?: " + str(wait) + " " + str(self.goal_finished))
            timeout = 5
            steps = 0
            while steps < timeout and not self.goal_finished:
                # rospy.loginfo("Waiting for feedback or goal finishing")
                if use_force and self.force_mag > force_limit:
                    self.move_group.stop()
                    rospy.loginfo("Stopping movement due to force feedback")
                    early_stop = True
                    break
                elif use_gripper and self.object_detected:
                    self.move_group.stop()
                    rospy.loginfo("Stopping movement due to object detection")
                    early_stop = True
                    break
                rospy.sleep(.01)
                steps = steps + .01        # # A `DisplayTrajectory`_ msg has two primary fields, trajectory_start and trajectory.
        # # We populate the trajectory_start with our current robot state to copy over
        # # any AttachedCollisionObjects and add our plan to the trajectory.
        # display_trajectory = DisplayTrajectory()
        # display_trajectory.trajectory_start = self.home_state if startpoint == 'home' else self.commander.get_current_state()
        # display_trajectory.trajectory.append(plan)
        # # Publish
        # self.display_trajectory_publisher.publish(display_trajectory)

 
        current_pose = self.move_group.get_current_pose()
        rospy.loginfo(f"Pose dist: {pose_dist(goal_in_planning_frame, current_pose)}")
        return early_stop or all_close(goal_in_planning_frame, current_pose, tolerance)


    @requires_controller(JOINT_GROUP_CONTROLLER)
    def servo_to_pose(self,
                      pose_stamped,
                      timeout=10,
                      pos_tolerance=0.01,
                      angular_tolerance=0.2,
                      avoid_collisions=True):
        if not self.servo_to_pose_client.wait_for_server(rospy.Duration(1)):
            rospy.logerr("Servo action server couldn't be reached")
            return False
        goal = ServoToPoseGoal()
        goal.pose = pose_stamped
        goal.positional_tolerance = [pos_tolerance, pos_tolerance, pos_tolerance]
        goal.angular_tolerance = angular_tolerance
        goal.timeout = rospy.Duration(timeout)
        self.servo_to_pose_client.send_goal(goal)
        # 0 timeout to give the server an infinite time to come back.
        finished = self.servo_to_pose_client.wait_for_result(rospy.Duration(0))
        if not finished:
            return False
        result = self.servo_to_pose_client.get_result()
        if not result:
            return False
        return result.error_code == 0


    def check_pose(self,
                   pose_stamped,
                   allowed_planning_time=10.0,
                   group_name=ARM_GROUP_NAME,
                   tolerance=0.01):
        return self.move_to_pose(
            pose_stamped,
            allowed_planning_time=allowed_planning_time,
            group_name=group_name,
            tolerance=tolerance,
            plan_only=True)


    def compute_ik(self, pose_stamped, timeout=rospy.Duration(5)):
        """ Computes inverse kinematics (joint values) for the given pose.
        
        Note: if you are interested in returning the IK solutions, we have
            shown how to access them.
        
        Args:
            pose_stamped: geometry_msgs/PoseStamped.
            timeout: rospy.Duration. How long to wait before giving up on the IK solution.
        
        Returns: A list of (name, value) for the arm joints if the IK solution
            was found, False otherwise. """
        # compute the ik solution for the robot to move to the pre-grasp pose
        request = GetPositionIKRequest()
        request.ik_request.group_name = ARM_GROUP_NAME
        # set robot state to home state each time method is run
        request.ik_request.robot_state = self.home_state
        request.ik_request.avoid_collisions = True
        request.ik_request.ik_link_name = "epick_end_effector"
        request.ik_request.pose_stamped = pose_stamped
        request.ik_request.timeout = timeout

        response = self._compute_ik(request)
        error_str = moveit_error_string(response.error_code.val)
        success = error_str == 'SUCCESS'
        if not success:
            return None
        
        # response.solution = moveit_msgs/RobotState
        # moveit_msgs/RobotState:
        #   sensor_msgs/JointState joint_state:
        #       string[] name
        #       float64[] position
        #       float64[] velocity
        #       float64[] effort
        #   sensor_msgs/MultiDOFJointState multi_dof_joint_state [eg. planar joint has 3DOF = (x,y,yaw)] NA because urdf file defines joints as 1DOF joints
        #   ...
        joint_state = response.solution.joint_state
        # print(f"\njoint values before:\t{joint_state.position}")

        # wrap angles into range [-2pi, 2pi) for all joints except elbow_joint which should be within [-pi, pi)
        # joints = dict()
        # for joint_name, joint_value in zip(joint_state.name, joint_state.position):
        #     joints[joint_name] = self.wrap_angle(joint_value, math.pi) if joint_name == 'arm_elbow_joint' else self.wrap_angle(joint_value, 2*math.pi)
        joint_state.position = tuple([self.wrap_angle(joint_value, math.pi) if joint_name == 'arm_elbow_joint' \
                                      else self.wrap_angle(joint_value, 2*math.pi) \
                                      for joint_name, joint_value in zip(joint_state.name, joint_state.position)])
        # print(f"joint values after: \t{joint_state.position}")

        # joints = []
        # for name, position in zip(joint_state.name, joint_state.position):
        #     # There is no public function get_active_joint_names() in self.commander, still it's callable
        #     # On the other hand, there is self.move_group.get_active_joints() which gives the same result
        #     if name in self.commander.get_active_joint_names():
        #         joints.append((name, position))
        return joint_state


    def compute_path(self,goal_config,
                          allowed_planning_time=10.0,
                          num_planning_attempts=20,
                          orientation_constraint=None,
                          replan=True,
                          replan_attempts=1,
                          joint_tolerance=0.01,
                          position_tolerance=0.001,
                          orientation_tolerance=0.01,
                          startpoint = 'home',
                          goal_type='pose',
                          collision_free=False):
        """ Computes a path for the end-effector to a pose from startpoint, using motion planning.
        
        Args:
            goal_config: geometry_msgs/PoseStamped, list of joint values, sensor_msgs/JointState
                          The goal configuration for the end-effector.
            allowed_planning_time: float. 
                                   The maximum duration to wait for a planning result.
            num_planning_attempts: int. 
                                   The number of times to compute the same plan. 
                                   The shortest path is ultimately used. 
                                   For random planners, this can help get shorter, less weird paths.
            orientation_constraint: moveit_msgs/OrientationConstraint. 
                                    An orientation constraint for the entire path.
            replan: bool. 
                    If True, then if an execution fails (while the arm is moving), 
                    then come up with a new plan and execute it.
            replan_attempts: int. 
                             How many times to replan if the execution fails.
            joint_tolerance:       float. 
                                   The goal joint tolerance, in radians.
            position_tolerance:    float. 
                                   The goal position tolerance, in meters.
            orientation_tolerance: float. 
                                   The goal orientation tolerance, in radians. 1 degree = 0.017 rad
            startpoint: string. 
                       The startpoint for motion planning. If any other than 'home', current robot state will be startpoint
        
        Returns:
            moveit_msgs/RobotTrajectory.msg if path exists, else None. """
        
        goal_config = from_msg_msg(goal_config)

        self.move_group.set_end_effector_link("epick_end_effector")
        self.move_group.set_planning_time(allowed_planning_time)
        self.move_group.set_num_planning_attempts(num_planning_attempts)
        self.move_group.allow_replanning(replan)
        self.move_group.set_goal_joint_tolerance(joint_tolerance)
        self.move_group.set_goal_position_tolerance(position_tolerance)
        self.move_group.set_goal_orientation_tolerance(orientation_tolerance)
        if startpoint == 'home':
            self.move_group.set_start_state(self.home_state)
        else:
            self.move_group.set_start_state_to_current_state()
        success = False
        if goal_type == 'joint':
            self.move_group.set_joint_value_target(goal_config)
            success, plan, planning_time, error_code = self.move_group.plan()            
        elif goal_type == 'pose':
            if collision_free:
                for attempt_id in range(replan_attempts):
                    goal_joint_state = self.compute_ik(goal_config, timeout=rospy.Duration(5))
                    if goal_joint_state is not None:
                        # print(f"goal_joint_values:\n{goal_joint_state.position}")
                        self.move_group.set_joint_value_target(goal_joint_state)
                        success, plan, planning_time, error_code = self.move_group.plan()
                        if success:
                            break
            else:
                self.move_group.set_pose_target(goal_config)
                success, plan, planning_time, error_code = self.move_group.plan()

            # It is always good to clear your targets after planning with poses.
            # Note: there is no equivalent function for clear_joint_value_targets()
            self.move_group.clear_pose_targets()

        if not success:
            return None
        return tuple([plan, planning_time])
    

    def execute_path(self, path, goal_pose=None, full_cycle=False):
        """ Executes the path generated by compute_path() method
        
        Args:
            path: moveit_msgs/RobotTrajectory.msg. 
                  The path
            goal_pose: geometry_msgs/PoseStamped. 
                       The goal pose for the end-effector.
            full_cycle: bool. 
                        If True, executes the full cycle, i.e. returns the robot along the reversed path """

        ret = self.move_group.execute(path, wait=True) # Call the planner to compute the reversed path and execute it.
        self.move_group.stop() # Calling 'stop()' ensures that there is no residual movement

        # print(f"\ngoal_pose_in_bin_frame:\n{goal_pose.pose}")
        if goal_pose is not None:
            # Transform pose_stamped in bin's referrence frame to self.planning_frame (= world)
            goal_in_planning_frame = self.tf2_buffer.transform(goal_pose, self.planning_frame, rospy.Duration(1))
            # print(f"\ngoal_pose_in_planning_frame:\n{goal_in_planning_frame.pose}")
            current_pose = self.move_group.get_current_pose()
            # print(f"\ncurrent_pose:\n{current_pose.pose}")
            pose_diff = pose_dist(goal_in_planning_frame, current_pose)
            print(f"\nposes_difference:\n\tdistance: {pose_diff[0]:2.4f} meters\n\t   angle: {pose_diff[1]:2.4f} degrees\n")

        if full_cycle:
            # Return the robot to original pose, i.e. reverse the path
            path.joint_trajectory.points.reverse()
            total_duration = path.joint_trajectory.points[0].time_from_start
            for point in path.joint_trajectory.points:
                point.time_from_start = total_duration - point.time_from_start # point.time_from_start is reversed, decreasing over path, while it has to increase
                point.velocities = (np.array(point.velocities) * (-1)).tolist() # point.velocities/accelerations should be inverted to get smooth reversed path
                point.accelerations = (np.array(point.accelerations) * (-1)).tolist()
            ret = self.move_group.execute(path, wait=True) # Call the planner to compute the reversed path and execute it.
            self.move_group.stop() # Calling `stop()` ensures that there is no residual movement


    def wrap_angle(self, angle:float, limit:float):
        # wrap angle into range [-limit, limit)
        return (angle - limit) % (2*limit) - limit
    

    def cancel_all_goals(self):
        self._move_group_client.cancel_all_goals()
        self._joint_traj_client.cancel_all_goals()


    def remove_pod_collision_geometry(self):
        # self.scene.remove_world_object() # if no name is provided, then all objects are removed from the scene
        self.scene.clear() # There is no such function as moveit_commander.planning_scene_interface.PlanningSceneInterface.clear() aka scene.clear(), but this function works!


    def add_pod_collision_geometry(self, pod_id=1):
        '''
        Includes collision boxes for each bin into MoveIt planning scene depending on pod's ID
        pod_id = 1: pod of 13x4 small bins
        pod_id = 2: pod of 8x3 large bins
        '''
        POD_BRACE_WIDTH = self.pod_sizes[f"pod{pod_id}_brace_width"]
        POD_BIN_WIDTH = self.pod_sizes[f"pod{pod_id}_bin_width"]
        POD_BINS_HEIGHT = sum(self.pod_sizes[f"pod{pod_id}_bin_heights"])
        POD_BRACE_FRAME_WIDTH = self.pod_sizes[f"pod{pod_id}_brace_frame_width"]
        POD_BRACE_FRAME_THICKNESS = self.pod_sizes[f"pod{pod_id}_brace_frame_thickness"]
        POD_BIN_DEPTH = self.pod_sizes[f"pod{pod_id}_bin_depth"]
        POD_BIN_WALL_THICKNESS = self.pod_sizes[f"pod{pod_id}_bin_wall_thickness"]
        POD_BIN_BOTTOM_THICKNESS = self.pod_sizes[f"pod{pod_id}_bin_bottom_thickness"]
        POD_BIN_FLAP_HEIGHT = self.pod_sizes[f"pod{pod_id}_bin_flap_height"]
        NUM_COLUMNS = math.ceil(POD_BRACE_WIDTH / POD_BIN_WIDTH)

        X_AXIS_QUAT = Quaternion(x=0, y=0, z=0, w=1)


        # Clear the scene from collision boxes
        self.remove_pod_collision_geometry()

        # SETUP POD BIN'S COLLISION BOXES
        # COLUMNS
        for i in range(NUM_COLUMNS + 1):
            self.scene.add_box(f"col_{i+1:02d}", PoseStamped(header=std_msgs.msg.Header(frame_id="pod_fabric_base"),
                                                 pose=Pose(position=Point(x=i*POD_BIN_WIDTH, 
                                                                          y=1/2*POD_BIN_DEPTH, 
                                                                          z=1/2*POD_BINS_HEIGHT), orientation=X_AXIS_QUAT)), 
                                                 (POD_BIN_WALL_THICKNESS, POD_BIN_DEPTH, POD_BINS_HEIGHT))
        # ROWS
        POD_BIN_Z_OFFSET_LIST = [0.0] + self.pod_sizes[f"pod{pod_id}_bin_heights"]
        POD_BIN_NUM = len(POD_BIN_Z_OFFSET_LIST)
        POD_BIN_Z_OFFSET = 0
        for POD_BIN_ID, POD_BIN_HEIGHT in enumerate(POD_BIN_Z_OFFSET_LIST, start=1):
            POD_BIN_Z_OFFSET += POD_BIN_HEIGHT
            if POD_BIN_ID != POD_BIN_NUM:
                self.scene.add_box(f"row_{POD_BIN_ID:02d}", PoseStamped(header=std_msgs.msg.Header(frame_id="pod_fabric_base"),
                                                            pose=Pose(position=Point(x=1/2*POD_BRACE_WIDTH, 
                                                                                    y=1/2*POD_BIN_DEPTH, 
                                                                                    z=POD_BIN_Z_OFFSET), orientation=X_AXIS_QUAT)), 
                                                            (POD_BRACE_WIDTH, POD_BIN_DEPTH, POD_BIN_BOTTOM_THICKNESS))
                # FLAPS
                self.scene.add_box(f"flap_{POD_BIN_ID:02d}", PoseStamped(header=std_msgs.msg.Header(frame_id="pod_fabric_base"),
                                                            pose=Pose(position=Point(x=1/2*POD_BRACE_WIDTH, 
                                                                                     y=0, 
                                                                                     z=POD_BIN_Z_OFFSET+1/2*POD_BIN_FLAP_HEIGHT), orientation=X_AXIS_QUAT)), 
                                                            (POD_BRACE_WIDTH, POD_BIN_WALL_THICKNESS, POD_BIN_FLAP_HEIGHT))
            else:
                self.scene.add_box(f"row_{POD_BIN_ID:02d}", PoseStamped(header=std_msgs.msg.Header(frame_id="pod_fabric_base"),
                                                            pose=Pose(position=Point(x=1/2*POD_BRACE_WIDTH, 
                                                                                    y=1/2*POD_BIN_DEPTH, 
                                                                                    z=POD_BIN_Z_OFFSET), orientation=X_AXIS_QUAT)), 
                                                            (POD_BRACE_WIDTH, POD_BIN_DEPTH, POD_BIN_WALL_THICKNESS))                

        self.scene.add_box(f"back_wall", PoseStamped(header=std_msgs.msg.Header(frame_id="pod_fabric_base"),
                                                     pose=Pose(position=Point(x=1/2*POD_BRACE_WIDTH, 
                                                                              y=POD_BIN_DEPTH, 
                                                                              z=1/2*POD_BINS_HEIGHT), orientation=X_AXIS_QUAT)), 
                                                     (POD_BRACE_WIDTH, POD_BIN_WALL_THICKNESS, POD_BINS_HEIGHT))
                                
        self.scene.add_box(f"pod_brace_leftside_frame", PoseStamped(header=std_msgs.msg.Header(frame_id="pod_fabric_base"),
                                                        pose=Pose(position=Point(x=1/2*POD_BRACE_FRAME_WIDTH, 
                                                                                 y=1/2*POD_BRACE_FRAME_THICKNESS, 
                                                                                 z=1/2*POD_BINS_HEIGHT), orientation=X_AXIS_QUAT)), 
                                                        (POD_BRACE_FRAME_WIDTH, POD_BRACE_FRAME_THICKNESS, POD_BINS_HEIGHT))      
        self.scene.add_box(f"pod_brace_rightside_frame", PoseStamped(header=std_msgs.msg.Header(frame_id="pod_fabric_base"),
                                                         pose=Pose(position=Point(x=POD_BRACE_WIDTH-1/2*POD_BRACE_FRAME_WIDTH, 
                                                                                  y=1/2*POD_BRACE_FRAME_THICKNESS, 
                                                                                  z=1/2*POD_BINS_HEIGHT), orientation=X_AXIS_QUAT)), 
                                                         (POD_BRACE_FRAME_WIDTH, POD_BRACE_FRAME_THICKNESS, POD_BINS_HEIGHT)) 
