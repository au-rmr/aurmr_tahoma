import rospy
from visualization_msgs.msg import Marker


class EEFVisualization():

    def __init__(self, topic="/visualization_eef",  color=(1.0, 0.0, 0.0)):
        self.pub = rospy.Publisher(topic, Marker, queue_size=1)
        self.color = color

    def visualize_eef(self, pose):

        marker = Marker()
        marker.header.frame_id = pose.header.frame_id
        marker.pose = pose.pose
        marker.type = Marker.SPHERE
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.r = self.color[0]
        marker.color.g = self.color[1]
        marker.color.b = self.color[2]
        marker.color.a = 1.0
        self.pub.publish(marker)