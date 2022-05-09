import copy
import sys

from control_msgs.msg import FollowJointTrajectoryAction
from control_msgs.msg import FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
import tf2_ros
import tf2_geometry_msgs

import actionlib
import control_msgs.msg
import trajectory_msgs.msg
import rospy
import tf
from aurmr_tasks.util import all_close, pose_dist
from moveit_msgs.msg import MoveItErrorCodes, MoveGroupAction, DisplayTrajectory
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from tf.listener import TransformListener
import moveit_commander

ARM_GROUP_NAME = 'manipulator'
JOINT_ACTION_SERVER = '/pos_joint_traj_controller/follow_joint_trajectory'
MOVE_GROUP_ACTION_SERVER = 'move_group'
TIME_FROM_START = 5

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


def moveit_error_string(val):
    """Returns a string associated with a MoveItErrorCode.

    Args:
        val: The val field from moveit_msgs/MoveItErrorCodes.msg

    Returns: The string associated with the error value, 'UNKNOWN_ERROR_CODE'
        if the value is invalid.
    """
    return MOVE_IT_ERROR_TO_STRING.get(val, 'UNKNOWN_ERROR_CODE')


class Tahoma:
    def __init__(self):
        self.joint_state = None
        self.point_cloud = None

        self.tf2_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer)

        self.wrist_position = None
        self.lift_position = None

        # self.joint_states_subscriber = rospy.Subscriber('/joint_states', JointState, self.joint_states_callback)
        self._joint_traj_client = actionlib.SimpleActionClient(
            JOINT_ACTION_SERVER, control_msgs.msg.FollowJointTrajectoryAction)
        server_reached = self._joint_traj_client.wait_for_server(rospy.Duration(10))
        if not server_reached:
            rospy.signal_shutdown('Unable to connect to arm action server. Timeout exceeded.')
            sys.exit()
        self._move_group_client = actionlib.SimpleActionClient(
            MOVE_GROUP_ACTION_SERVER, MoveGroupAction)
        self._move_group_client.wait_for_server(rospy.Duration(10))
        self._compute_ik = rospy.ServiceProxy('compute_ik', GetPositionIK)
        self._tf_listener = TransformListener()
        self.tuck_pose = [1.32, 1.40, -0.2, 1.72, 0.0, 1.66, 0.0]
        moveit_commander.roscpp_initialize(sys.argv)
        self.commander = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface(synchronous=True)
        self.move_group = moveit_commander.MoveGroupCommander(ARM_GROUP_NAME)
        self.move_group.set_max_velocity_scaling_factor(.15)
        self.display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            DisplayTrajectory,
            queue_size=20,
        )
        planning_frame = self.move_group.get_planning_frame()
        eef_link = self.move_group.get_end_effector_link()
        group_names = self.commander.get_group_names()

        # Misc variables
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names

    def move_to_pose(self, pose, return_before_done=False):
        joint_names = [key for key in pose]
        point = JointTrajectoryPoint()
        point.time_from_start = rospy.Duration(0, 1)

        trajectory_goal = FollowJointTrajectoryGoal()
        trajectory_goal.goal_time_tolerance = rospy.Time(1.0)
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

    def tuck(self):
        """
        Uses motion-planning to tuck the arm within the footprint of the base.
        :return: a string describing the error, or None if there was no error
        """

        return self.move_to_joint_goal(zip(ArmJoints.names(), self.tuck_pose))

    def tuck_unsafe(self):
        """
        TUCKS BUT DOES NOT PREVENT SELF-COLLISIONS, WHICH ARE HIGHLY LIKELY.

        Don't use this unless you have prior knowledge that the arm can safely return
        to tucked from its current configuration. Most likely, you should only use this
        method in simulation, where the arm can clip through the base without issue.
        :return:
        """
        return self.move_to_joints(ArmJoints.from_list(self.tuck_pose))

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

    def move_to_joint_angles(self,
                           joints,
                           allowed_planning_time=10.0,
                           execution_timeout=15.0,
                           group_name=ARM_GROUP_NAME,
                           num_planning_attempts=1,
                           plan_only=False,
                           replan=False,
                           replan_attempts=5,
                           tolerance=0.01):
        """Moves the end-effector to a pose, using motion planning.

        Args:
            joints: A list of (name, value) for the arm joints.
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

        self.move_group.set_planning_time(allowed_planning_time)
        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        self.move_group.go(joints, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        self.move_group.stop()

        current_joints = self.move_group.get_current_joint_values()
        return all_close(joints, current_joints, 0.01)

    def move_to_pose_goal(self,
                          pose_stamped,
                          allowed_planning_time=10.0,
                          execution_timeout=15.0,
                          group_name='manipulator',
                          num_planning_attempts=1,
                          orientation_constraint=None,
                          plan_only=False,
                          replan=False,
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
            group_name: string.
            num_planning_attempts: int. The number of times to compute the same
                plan. The shortest path is ultimately used. For random
                planners, this can help get shorter, less weird paths.
            orientation_constraint: moveit_msgs/OrientationConstraint. An
                orientation constraint for the entire path.
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

        # FIXME: Hardcoded to the global frame that MoveIt will use for reporting current pose
        transform = self.tf2_buffer.lookup_transform('base_link',
                                       # source frame:
                                       pose_stamped.header.frame_id,
                                       # get the tf at the time the pose was valid
                                       pose_stamped.header.stamp,
                                       # wait for at most 1 second for transform, otherwise throw
                                       rospy.Duration(1.0))

        goal_in_base_link = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)
        self.move_group.set_pose_target(pose_stamped)
        plan = self.move_group.plan()[1]


        ## A `DisplayTrajectory`_ msg has two primary fields, trajectory_start and trajectory.
        ## We populate the trajectory_start with our current robot state to copy over
        ## any AttachedCollisionObjects and add our plan to the trajectory.
        display_trajectory = DisplayTrajectory()
        display_trajectory.trajectory_start = self.commander.get_current_state()
        display_trajectory.trajectory.append(plan)
        # Publish
        self.display_trajectory_publisher.publish(display_trajectory)

        ## Now, we call the planner to compute the plan and execute it.
        plan = self.move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        self.move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        self.move_group.clear_pose_targets()

        current_pose = self.move_group.get_current_pose()
        rospy.loginfo("Pose dist:", pose_dist(pose_stamped, current_pose), pose_stamped.header.frame_id, current_pose.header.frame_id)
        return all_close(goal_in_base_link, current_pose, 0.01)

    def straight_move_to_pose(self,
                              pose_stamped,
                              ee_step=0.025,
                              jump_threshold=2.0,
                              avoid_collisions=True):
        """Moves the end-effector to a pose in a straight line.

        Args:
          group: moveit_commander.MoveGroupCommander. The planning group for
            the arm.
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
        waypoints = [pose_stamped]

        # We want the Cartesian path to be interpolated at a resolution of 1 cm
        # which is why we will specify 0.01 as the eef_step in Cartesian
        # translation.  We will disable the jump threshold by setting it to 0.0,
        # ignoring the check for infeasible jumps in joint space, which is sufficient
        # for this tutorial.
        (plan, fraction) = self.move_group.compute_cartesian_path(
            waypoints, 0.01, jump_threshold, avoid_collisions  # waypoints to follow  # eef_step
        )
        if fraction < .9:
            rospy.logwarn(f"Not moving in cartesian path. Only {fraction} waypoints reached")
            return False
        return self.move_group.execute(plan, wait=True)

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
