import copy
import math
import sys
from functools import wraps
from turtle import pos

from actionlib import SimpleActionClient
from actionlib_msgs.msg import GoalStatusArray, GoalStatus
from control_msgs.msg import FollowJointTrajectoryAction, GripperCommandAction, GripperCommandGoal
from control_msgs.msg import FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
import tf2_ros

#import tf2_geometry_msgs

import actionlib
import control_msgs.msg
import trajectory_msgs.msg
import rospy

from tf2_geometry_msgs import from_msg_msg
from geometry_msgs.msg import PoseStamped, WrenchStamped
from robotiq_2f_gripper_control.msg import vacuum_gripper_input as VacuumGripperStatus
from aurmr_tasks.util import all_close, pose_dist
from moveit_msgs.msg import MoveItErrorCodes, MoveGroupAction, DisplayTrajectory
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest

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
                    return False
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
        self.move_group.set_max_velocity_scaling_factor(.15)
        self.display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            DisplayTrajectory,
            queue_size=1,
            latch=True
        )
        self._controller_lister = rospy.ServiceProxy("/controller_manager/list_controllers", ListControllers)
        self._controller_switcher = rospy.ServiceProxy("/controller_manager/switch_controller", SwitchController)
        self.servo_to_pose_client = SimpleActionClient("/servo_server/servo_to_pose", ServoToPoseAction)

        self.grasp_pose_pub = rospy.Publisher("~grasp_pose", PoseStamped)

        planning_frame = self.move_group.get_planning_frame()
        eef_link = self.move_group.get_end_effector_link()
        group_names = self.commander.get_group_names()

        self.wrench_listener = rospy.Subscriber("/wrench", WrenchStamped, self.wrench_cb)
        self.gripper_status_listener = rospy.Subscriber("/gripper_control/status", VacuumGripperStatus, self.gripper_status_cb)
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

    def wrench_cb(self, msg: WrenchStamped):
        self.force_mag = math.sqrt(msg.wrench.force.x**2 + msg.wrench.force.y**2+ msg.wrench.force.z**2)
        self.torque_mag = math.sqrt(msg.wrench.torque.x**2 + msg.wrench.torque.y**2+ msg.wrench.torque.z**2)

    def gripper_status_cb(self, msg: VacuumGripperStatus):
        self.object_detected = (msg.gOBJ == 1 or msg.gOBJ == 2)

    def goal_status_cb(self, msg: GoalStatusArray):
        latest_time = 0
        latest_status = GoalStatus.SUCCEEDED
        for g in msg.status_list:
            if g.goal_id.stamp.secs > latest_time:
                latest_time = g.goal_id.stamp.secs + g.goal_id.stamp.secs*10**(-9)
                latest_status = g.status
        self.goal_finished = latest_status != GoalStatus.PENDING and latest_status != GoalStatus.ACTIVE
        self.goal_stamp = latest_time
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
                           execution_timeout=15.0,
                           num_planning_attempts=8,
                           plan_only=False,
                           replan=False,
                           replan_attempts=5,
                           tolerance=0.01):
        """Moves the end-effector to a pose, using motion planning.

        Args:
            joints: A list of (name, value) for the arm joints. Alternatively a string name of a stored configuration
            allowed_planning_time: float. The maximum duration to wait for a
                planning result.
            execution_timeout: float. The maximum duration to wait for an arm
                motion to execute (or for planning to fail completely), in
                seconds.
            group_name: string.
            num_planning_attempts: int. The number of times to compute the same
                plan. The shortest path is ultimately used. For random
                planners, this can help get shorter, less weird paths.
            plan_only: bool. If True, then this method does not execute the
                plan on the robot. Useful for determining whether this is
                likely to succeed.
            replan: bool. If True, then if an execution fails (while the arm is
                moving), then come up with a new plan and execute it.
            replan_attempts: int. How many times to replan if the execution
                fails.
            tolerance: float. The goal tolerance, in meters.

        Returns:
            string describing the error if an error occurred, else None.
        """
        self.move_group.stop()
        
        self.move_group.set_num_planning_attempts(num_planning_attempts)
        self.move_group.allow_replanning(replan)
        self.move_group.set_goal_joint_tolerance(tolerance)
        self.move_group.set_planning_time(allowed_planning_time)

        joint_values = joints
        if isinstance(joints, str):
            joint_values = self.get_joint_values_for_name(joints)
            self.move_group.set_named_target(joints)
        else:
            self.move_group.set_joint_value_target(joints)
        self.move_group.go(wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        self.move_group.stop()

        current_joints = self.move_group.get_current_joint_values()
        return all_close(joint_values, current_joints, tolerance)

    @requires_controller(JOINT_TRAJ_CONTROLLER)
    def move_to_pose(self,
                          pose_stamped,
                          allowed_planning_time=10.0,
                          execution_timeout=15.0,
                          num_planning_attempts=20,
                          orientation_constraint=None,
                          replan=True,
                          replan_attempts=5,
                          tolerance=0.01):
        """Moves the end-effector to a pose, using motion planning.

        Args:
            pose: geometry_msgs/PoseStamped. The goal pose for the gripper.
            allowed_planning_time: float. The maximum duration to wait for a
                planning result.
            execution_timeout: float. The maximum duration to wait for an arm
                motion to execute (or for planning to fail completely), in
                seconds.
            num_planning_attempts: int. The number of times to compute the same
                plan. The shortest path is ultimately used. For random
                planners, this can help get shorter, less weird paths.
            orientation_constraint: moveit_msgs/OrientationConstraint. An
                orientation constraint for the entire path.
            replan: bool. If True, then if an execution fails (while the arm is
                moving), then come up with a new plan and execute it.
            replan_attempts: int. How many times to replan if the execution
                fails.
            tolerance: float. The goal tolerance, in meters.

        Returns:
            string describing the error if an error occurred, else None.
        """

        pose_stamped = from_msg_msg(pose_stamped )

        goal_in_planning_frame = self.tf2_buffer.transform(pose_stamped, self.move_group.get_planning_frame(),
                                       rospy.Duration(1))

        self.move_group.set_end_effector_link("arm_tool0")
        self.move_group.set_pose_target(pose_stamped)
        self.move_group.set_planning_time(allowed_planning_time)
        self.move_group.set_num_planning_attempts(num_planning_attempts)
        self.move_group.allow_replanning(replan)
        self.move_group.set_goal_position_tolerance(tolerance)
        success, plan, planning_time, error_code = self.move_group.plan()
        if not success:
            return False

        # A `DisplayTrajectory`_ msg has two primary fields, trajectory_start and trajectory.
        # We populate the trajectory_start with our current robot state to copy over
        # any AttachedCollisionObjects and add our plan to the trajectory.
        display_trajectory = DisplayTrajectory()
        display_trajectory.trajectory_start = self.commander.get_current_state()
        display_trajectory.trajectory.append(plan)
        # Publish
        self.display_trajectory_publisher.publish(display_trajectory)

        # Now, we call the planner to compute the plan and execute it.
        self.move_group.execute(plan, wait=True)
        # Calling `stop()` ensures that there is no residual movement
        self.move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        self.move_group.clear_pose_targets()

        current_pose = self.move_group.get_current_pose()
        rospy.loginfo(f"Pose dist: {pose_dist(goal_in_planning_frame, current_pose)}")
        return all_close(goal_in_planning_frame, current_pose, tolerance)

    @requires_controller(JOINT_TRAJ_CONTROLLER)
    def straight_move_to_pose(self,
                              pose_stamped,
                              tolerance=0.01,
                              ee_step=0.01,
                              jump_threshold=0.5,
                              avoid_collisions=True,
                              use_force=False,
                              use_gripper=False):
        """Moves the end-effector to a pose in a straight line.

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
 
        self.move_group.set_end_effector_link("arm_tool0")
        goal_in_planning_frame = self.tf2_buffer.transform(pose_stamped, self.planning_frame, rospy.Duration(1))

        self.grasp_pose_pub.publish(pose_stamped)

        waypoints = [goal_in_planning_frame.pose]

        (plan, fraction) = self.move_group.compute_cartesian_path(
            waypoints, ee_step, jump_threshold, avoid_collisions
        )
        if fraction < .9:
            rospy.logwarn(f"Not moving in cartesian path. Only {fraction} waypoints reached")
            # return False

        wait = not (use_force or use_gripper)
        rospy.loginfo("Waiting?: " + str(wait))
        old_goal_stamp = self.goal_stamp
        self.move_group.execute(plan, wait=wait)

        if use_gripper:
            self.close_gripper(return_before_done=True)

        if not wait:
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
                steps = steps + .01
        
 
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
                   group_name='manipulator',
                   tolerance=0.01):
        return self.move_to_pose(
            pose_stamped,
            allowed_planning_time=allowed_planning_time,
            group_name=group_name,
            tolerance=tolerance,
            plan_only=True)

    def compute_ik(self, pose_stamped, timeout=rospy.Duration(5)):
        """Computes inverse kinematics for the given pose.

        Note: if you are interested in returning the IK solutions, we have
            shown how to access them.

        Args:
            pose_stamped: geometry_msgs/PoseStamped.
            timeout: rospy.Duration. How long to wait before giving up on the
                IK solution.

        Returns: A list of (name, value) for the arm joints if the IK solution
            was found, False otherwise.
        """
        request = GetPositionIKRequest()
        request.ik_request.pose_stamped = pose_stamped
        request.ik_request.group_name = 'manipulator'
        request.ik_request.timeout = timeout
        response = self._compute_ik(request)
        error_str = moveit_error_string(response.error_code.val)
        success = error_str == 'SUCCESS'
        if not success:
            return False
        joint_state = response.solution.joint_state
        joints = []
        for name, position in zip(joint_state.name, joint_state.position):
            if name in self.commander.get_active_joint_names():
                joints.append((name, position))
        return joints

    def cancel_all_goals(self):
        self._move_group_client.cancel_all_goals()
        self._joint_traj_client.cancel_all_goals()
