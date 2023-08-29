#!/usr/bin/env python
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2

rospy.init_node('point_cloud_publisher')

def depth_to_point_cloud(depth_image):
    cx = 680.839599609375
    cy = 355.6268310546875
    fx = 903.14697265625
    fy = 903.5240478515625
    height, width = depth_image.shape
    x = np.linspace(0, width-1, width)
    y = np.linspace(0, height-1, height)
    x, y = np.meshgrid(x, y)
    normalized_x = (x - cx) / fx
    normalized_y = (y - cy) / fy
    z = depth_image
    x = normalized_x * z
    y = normalized_y * z
    point_cloud = np.dstack((x, y, z))
    return point_cloud

def callback_depth(data):
    try:
        depth_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
    except CvBridgeError as e:
        print(e)
    depth_array = np.array(depth_image, dtype=np.float32)/1000.0
    point_cloud = depth_to_point_cloud(depth_array)
    point_cloud = point_cloud.reshape(-1, 3)
    header = rospy.Header()
    pc_msg = point_cloud2.create_cloud_xyz32(header, point_cloud)
    pub.publish(pc_msg)

bridge = CvBridge()
depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, callback_depth)
pub = rospy.Publisher('/camera/depth/color/points2', PointCloud2, queue_size=10)
if __name__ == '__main__':
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass