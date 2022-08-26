from copy import copy, deepcopy

import rospy
from visualization_msgs.msg import Marker
import tf2_geometry_msgs
import tf2_ros
from tf_conversions import transformations

from aurmr_perception.util import quat_msg_to_vec, qv_mult, vec_to_quat_msg


def create_gripper_pose_markers(poses, color, ns="gripper_poses", tf_buffer=None):
    if not tf_buffer:
        # These are expensive. You really should pass one in.
        tf_buffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tf_buffer)
    # We're drawing the bulky part of the gripper, which is notably different in orientation and offset
    # than say, arm_tool0. Figure out how to get from the given frame to the gripper_base_link frame
    transform = tf_buffer.lookup_transform("gripper_base_link", "gripper_equilibrium_grasp", rospy.Time(0),
                                                   rospy.Duration(1)).transform

    markers = []
    colors = []
    if hasattr(color, '__iter__') and not hasattr(color[0], '__iter__'):
        # User passed a single color (like (0,0,1,1)
        colors = [color for i in range(len(poses))]
    elif callable(color):
        colors = [color(pose) for pose in poses]

    for i, pose in enumerate(poses):
        marker = Marker()
        marker.header.frame_id = pose.header.frame_id
        marker.header.stamp = rospy.Time(0)
        marker.ns = ns
        marker.id = i
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        marker.scale.x = 1
        marker.scale.y = 1
        marker.scale.z = 1
        marker.color.r = color[i][0]
        marker.color.g = color[i][1]
        marker.color.b = color[i][2]
        marker.color.a = color[i][3]
        marker.mesh_resource = "package://robotiq_2f_85_gripper_visualization/meshes/visual/full_opened.stl"
        #full_opened.stl"

        transformed_pose = deepcopy(pose.pose)
        rotated_quat = transformations.quaternion_multiply(quat_msg_to_vec(pose.pose.orientation), quat_msg_to_vec(transform.rotation))
        # This grasp is the pose between the fingers. Push the marker back along the z axis to have the fingers over the grasp point
        offset = qv_mult(quat_msg_to_vec(pose.pose.orientation), (-transform.translation.x, -transform.translation.y, -transform.translation.z))
        transformed_pose.position.x += offset[0]
        transformed_pose.position.y += offset[1]
        transformed_pose.position.z += offset[2]
        transformed_pose.orientation = vec_to_quat_msg(rotated_quat)
        marker.pose = transformed_pose

        markers.append(marker)
    return markers


def create_pose_arrow_markers(poses_stamped, ns="pose_arrows"):
    markers = []
    for i, pose in enumerate(poses_stamped):
        marker = Marker()
        marker.ns = ns
        marker.id = i
        marker.header = pose.header
        marker.type = Marker.ARROW
        marker.scale.x = 0.1
        marker.scale.y = 0.01
        marker.scale.z = 0.01
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.pose = pose.pose
        markers.append(marker)
    return markers