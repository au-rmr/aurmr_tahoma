#!/usr/bin/env python

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

"""collect images from Azure Kinect v2"""

import rospy
import message_filters
import cv2
import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np
import yaml
import json
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo

class ImageListener:

    def __init__(self):

        self.bridge = CvBridge()
        self.count = 0

        # initialize a node
        rospy.init_node("image_listener")
        self.camera_depth_subscriber = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        self.camera_rgb_subscriber = rospy.Subscriber('/camera/color/image_raw', Image, self.rgb_callback)
        # msg = rospy.wait_for_message('/camera_lower_right/depth_to_rgb/camera_info', CameraInfo)

    def depth_callback(self, data):
        try:
            self.depth_img = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
            # self.depth_img = np.array(self.depth_img, dtype=np.float32)
            self.depth_img = self.depth_img.astype(np.float32)
            # print(self.depth_img.shape)
        except CvBridgeError as e:
            print(e)
    
    def rgb_callback(self, data):
        try:
            self.rgb_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
            rgb = self.rgb_img.copy()
            # print("rgb image shape", self.rgb_img.shape)
            # rgb = cv2.resize(rgb, (640, 360), interpolation = cv2.INTER_AREA)
            cv2.imshow("rgb frame", rgb)
            cv2.waitKey(0)
        except CvBridgeError as e:
            print(e)


if __name__ == '__main__':

    # image listener
    listener = ImageListener()
    try:  
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
