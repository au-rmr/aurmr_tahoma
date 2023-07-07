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
    
    # (Old comment for default gripper 2F85) We're drawing the bulky part of the gripper, which is notably different in orientation and offset
    # than say, arm_tool0. Figure out how to get from the given frame to the gripper_base_link frame

    # (New comment for gripper robotiq_epick) "gripper_equilibrium_grasp" is aligned with "arm_tool0" orientation and placed at the base of suction cup
    # Notice that "epick_end_effector" and "gripper_equilibrium_grasp" are displaced from each other at distance of the suction cup height (10 mm), with the same orientations
    # "gripper_equilibrium_grasp" is rotated by 90deg around z-axis of "gripper_base_link"
    # Model in robotiq_epick_full.stl has origin reference frame the same as "gripper_base_link"
    # As I (Sanjar Normuradov) understood, "tf_buffer.lookup_transform().transform" tries to transform from source ("gripper_base_link") representation to target ("epick_end_effector"),
    # i.e. see in RViz how object pose(position, orientation) would look like when the frame "gripper_base_link" is aligned with a given frame
    # and when the frame "epick_end_effector" is aligned the given frame
    transform = tf_buffer.lookup_transform("epick_end_effector", "gripper_base_link", rospy.Time(0),
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
        # marker.scale.x = 1
        # marker.scale.y = 1
        # marker.scale.z = 1
        marker.color.r = color[i][0] 
        marker.color.g = color[i][1]
        marker.color.b = color[i][2]
        marker.color.a = color[i][3]
        # marker.mesh_resource = "package://robotiq_2f_85_gripper_visualization/meshes/visual/full_opened.stl"
        
        # NOTICE: robotiq_epick_full and robotiq_epick_tcp files has different origin reference frames!!! 
        #   robotiq_epick_full has at its center of contact surface with "coupling" (Look at aurmr_tahoma/tahoma_description/urdf/tahoma.xacro)
        #   robotiq_epick_tcp has at its TCP (Tool Contact Point) or EOT (End of Tool), i.e. where suction cup
        marker.mesh_resource = "package://robotiq_epick_visualization/meshes/visual/robotiq_epick_full.stl"
        marker.scale.x = 0.001
        marker.scale.y = 0.001
        marker.scale.z = 0.001
        #"package://robotiq_epick_visualization/meshes/visual/robotiq_epick.dae"
        #full_opened.stl"

        transformed_pose = deepcopy(pose.pose)
        rotated_quat = transformations.quaternion_multiply(quat_msg_to_vec(pose.pose.orientation), quat_msg_to_vec(transform.rotation))
        # This grasp is the pose between the fingers. Push the marker back along the z axis to have the fingers over the grasp point
        offset = qv_mult(quat_msg_to_vec(pose.pose.orientation), (transform.translation.x, transform.translation.y, transform.translation.z))
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