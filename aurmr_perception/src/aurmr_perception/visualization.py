import rospy
from visualization_msgs.msg import Marker


def create_gripper_pose_markers(poses, color, ns="gripper_poses"):
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
        marker.pose = pose.pose
        marker.scale.x = 1
        marker.scale.y = 1
        marker.scale.z = 1
        marker.color.r = colors[i][0]
        marker.color.g = colors[i][1]
        marker.color.b = colors[i][2]
        marker.color.a = colors[i][3]
        marker.mesh_resource = "package://robotiq_2f_85_gripper_visualization/meshes/visual/base_link.dae"
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