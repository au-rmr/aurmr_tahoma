import rospy
from visualization_msgs.msg import Marker, MarkerArray
import rospy

def create_gripper_pose_markers(poses, colors):
    markers = []

    for i, pose in enumerate(poses):
        marker = Marker()
        marker.header.frame_id = pose.header.frame_id
        marker.header.stamp = rospy.Time(0)
        marker.ns = "tahoma_gripper_viz"
        marker.id = i
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        marker.pose = pose.pose
        marker.scale.x = 1
        marker.scale.y = 1
        marker.scale.z = 1
        marker.color.a = 1.0
        marker.color.r = colors[i][0]
        marker.color.g = colors[i][1]
        marker.color.b = colors[i][2]
        marker.mesh_resource = "package://robotiq_2f_85_gripper_visualization/meshes/visual/robotiq_arg2f_base_link.stl"
        markers.append(marker)
    return markers