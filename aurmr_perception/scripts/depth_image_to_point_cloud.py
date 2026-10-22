#!/usr/bin/env python
import cv2
import time
import numpy as np

import rospy

from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point32
import std_msgs.msg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import sensor_msgs.point_cloud2 as pcl2

bridge = CvBridge()
global depth_img, rgb_img
depth_img = []
rgb_img = []

class CameraInfo():
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale

def depth_callback(data):
    global depth_img
    try:
        depth_img = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
    except CvBridgeError as e:
        print(e)

def rgb_callback(data):
    global rgb_img
    try:
        rgb_img = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)

def convert_rgb_depth_to_point_cloud(depth, camera_info):
    xmap = np.arange(camera_info.width)
    ymap = np.arange(camera_info.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth
    points_x = (xmap - camera_info.cx) * points_z / camera_info.fx
    points_y = (ymap - camera_info.cy) * points_z / camera_info.fy
    return np.array([points_x.flatten(), points_y.flatten(), points_z.flatten()]).T/1000.0

if __name__ == '__main__':
    rospy.init_node('Point_Cloud_Publisher_From_Depth_Cloud')
    pointcloud_publisher = rospy.Publisher("/camera/depth/color/points", PointCloud2, queue_size=10)
    camera_depth_subscriber = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, depth_callback)
    camera_rgb_subscriber = rospy.Subscriber('/camera/color/image_raw', Image, rgb_callback)
    fx, fy = 633.3889770507812, 631.7814331054688
    cx, cy = 634.9332885742188, 374.93145751953125
    width = 1280
    height = 720
    s = 1000.0
    rospy.sleep(1)
    camera_info = CameraInfo(width, height, fx, fy, cx, cy, s)
    # pointcloud_pub = PointCloud2()
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'camera_color_optical_frame'
    # pointcloud_pub.header = header
    while not rospy.is_shutdown():
        point_cloud_convert = convert_rgb_depth_to_point_cloud(depth_img, camera_info)
        # point_cloud_convert = point_cloud_convert.reshape((point_cloud_convert.shape[1]*point_cloud_convert.shape[2]), point_cloud_convert.shape[0])
        # print(point_cloud_convert.shape)
        # pointcloud_pub.points = point_cloud_convert
        pointcloud_pub = pcl2.create_cloud_xyz32(header, point_cloud_convert)
        # pointcloud_pub.height = 720
        # pointcloud_pub.width = 1280

        # print(pointcloud_pub)
        pointcloud_publisher.publish(pointcloud_pub)
        rospy.sleep(0.033)