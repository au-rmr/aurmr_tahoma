from copy import deepcopy
from typing import Union

import geometry_msgs.msg
from math import pi, tau, dist, fabs, cos

import rospy
from geometry_msgs.msg import PoseStamped, Pose
from moveit_commander.conversions import pose_to_list
from smach import State, StateMachine
from controller_manager_msgs.srv import SwitchController
import tf.transformations as tf
from tf2_ros import Buffer
import numpy as np

from aurmr_perception.util import qv_mult, quat_msg_to_vec


def apply_offset_to_pose(pose, offset, offset_frame=None, tf_buffer=None):
    if not isinstance(pose, PoseStamped):
        raise RuntimeError("Can't offset without knowing pose frame")
    offset_pose = deepcopy(pose)
    if offset_frame is None:
        offset_frame = offset_pose.header.frame_id

    offset_pose = tf_buffer.transform(offset_pose, offset_frame, rospy.Duration(1))
    offset_pose.pose.position.x += offset[0]
    offset_pose.pose.position.y += offset[1]
    offset_pose.pose.position.z += offset[2]
    # It would be better to grab the transform once and apply it to the offset vector
    offset_pose = tf_buffer.transform(offset_pose, pose.header.frame_id, rospy.Duration(1))
    return offset_pose

def add_offset(offset, grasp_pose):
    v = qv_mult(
        quat_msg_to_vec(grasp_pose.pose.orientation), (0, 0, offset))
    offset_pose = deepcopy(grasp_pose)
    offset_pose.pose.position.x += v[0]
    offset_pose.pose.position.y += v[1]
    offset_pose.pose.position.z += v[2]
    return offset_pose

def apply_transform_to_pose(transformation: PoseStamped, tf_buffer: Buffer) -> PoseStamped:
    """Given some local transformation, return a new pose

    Say you have the pose of the end effector in the world coordinates and you want
    to offset in both translation and orientation. This returns a new pose in the original
    pose coordinates.

    All the offset motions in tahoma use the arm_tool0 frame (hardcoded for some reason),
    so we need output the resulting pose for arm_tool0.

    :param current_pose: The current pose in some coordinate frame.
    :param transformation: The offsets in translation and orientation in the local frame
    """
    # flow is as follows
    # get base to gripper
    # get arm_tool0 to gripper transform
    # T_base_gripper @ transform @ T_arm_tool0_gripper^-1
    transform_frame = transformation.header.frame_id
    T_base_gripper = tf_buffer.lookup_transform(
        "base_link",
        transform_frame,
        rospy.Time(0),
        rospy.Duration(1.0)
    )
    T_arm_tool0_gripper = tf_buffer.lookup_transform(
        "arm_tool0",
        transform_frame,
        rospy.Time(0),
        rospy.Duration(1.0)
    )

    def pose_to_mat(p):
        pos = p.pose.position
        rot = p.pose.orientation
        return tf.concatenate_matrices(
            tf.translation_matrix([pos.x, pos.y, pos.z]),
            tf.quaternion_matrix([rot.x, rot.y, rot.z, rot.w,])
        )

    def transform_to_mat(T):
        pos = T.transform.translation
        rot = T.transform.rotation
        return tf.concatenate_matrices(
            tf.translation_matrix([pos.x, pos.y, pos.z]),
            tf.quaternion_matrix([rot.x, rot.y, rot.z, rot.w,])
        )
    def mat_to_pose(T):
        p = Pose()
        p.position.x, p.position.y, p.position.z = tf.translation_from_matrix(T)
        q = tf.quaternion_from_matrix(T)
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q
        return p

    mat_base_tip = transform_to_mat(T_base_gripper)
    mat_tool0_tip = transform_to_mat(T_arm_tool0_gripper)
    mat_tip_delta = pose_to_mat(transformation)

    mat_base_new_tool0 = mat_base_tip @ mat_tip_delta @ np.linalg.inv(mat_tool0_tip)

    target_tool0_pose = mat_to_pose(mat_base_new_tool0)
    target_tool0_posestamped = PoseStamped()
    target_tool0_posestamped.header.frame_id = "base_link"
    target_tool0_posestamped.pose = target_tool0_pose
    target_tool0_posestamped.header.stamp = rospy.Time.now()
    return target_tool0_posestamped


class Formulator(State):
    def __init__(self, template, input_keys, output_key):
        State.__init__(self, outcomes=["succeeded", "aborted"], input_keys=input_keys,
                       output_keys=[output_key])
        self.template = template

    def execute(self, ud):
        try:
            args = [ud[input_key] for input_key in self._input_keys]
            ud[list(self._output_keys)[0]] = self.template.format(*args)
            return "succeeded"
        except Exception as e:
            # Catch any fumbled templates
            rospy.logerr("Bad formulate_ud_str: {}, {}, {}".format(self.template, self._input_keys, self._output_keys))
            return "aborted"


def formulate_ud_str_auto(name, template, input_keys, output_key, transitions=None):
    if transitions is None:
        transitions = {}
    StateMachine.add_auto(name, Formulator(template, input_keys, output_key), ["succeeded"],transitions=transitions)


def all_close(goal: Union[geometry_msgs.msg.Pose, geometry_msgs.msg.PoseStamped, list], actual: Union[geometry_msgs.msg.Pose, geometry_msgs.msg.PoseStamped, list], tolerance: float):
    """
    Convenience method for testing if the values in two lists are within a tolerance of each other.
    For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
    between the identical orientations q and -q is calculated correctly).
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        assert (goal.header.frame_id == actual.header.frame_id)
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))
        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        # return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)
        return d <= tolerance

    return True


def pose_dist(goal, actual):
    if type(goal) is geometry_msgs.msg.PoseStamped:
        return pose_dist(goal.pose, actual.pose)

    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))
        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return (d, cos_phi_half)
    else:
        return None


class SwitchControllers(State):
    def __init__(self, start_controllers, stop_controllers):
        State.__init__(self, outcomes=['succeeded', 'aborted'])
        self.start_controllers = start_controllers
        self.stop_controllers = stop_controllers

    def execute(self, userdata):
        rospy.wait_for_service('/controller_manager/switch_controller')
        try:
            # Create a service proxy
            switch_controller_service = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)

            # Call the service with the provided arguments
            response = switch_controller_service(self.start_controllers, self.stop_controllers, 1, False, 0.0)
            print(f"Service call response: {response}")

            return 'succeeded'
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
            return 'aborted'
