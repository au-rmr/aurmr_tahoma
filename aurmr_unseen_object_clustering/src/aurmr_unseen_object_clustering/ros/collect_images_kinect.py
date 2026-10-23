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
        self.camera_depth_subscriber = rospy.Subscriber('/camera_lower_right/depth_to_rgb/image_raw', Image, self.depth_callback)
        self.camera_rgb_subscriber = rospy.Subscriber('/camera_lower_right/rgb/image_raw', Image, self.rgb_callback)
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
            # bin_bounds = {
            #     '1H':[296*4, 353*4, 317*4, 407*4],
            #     '2H':[297*4, 353*4, 408*4, 512*4],
            #     '3H':[298*4, 353*4, 514*4, 619*4],
            #     '4H':[301*4, 353*4, 619*4, 711*4],
            #     '1G':[364*4, 408*4, 317*4, 406*4],
            #     '2G':[366*4, 407*4, 407*4, 512*4],
            #     '3G':[368*4, 408*4, 514*4, 618*4],
            #     '4G':[369*4, 406*4, 620*4, 711*4],
            #     '1F':[419*4, 512*4, 315*4, 406*4],
            #     '2F':[424*4, 509*4, 407*4, 513*4],
            #     '3F':[425*4, 511*4, 515*4, 620*4],    
            #     '4F':[424*4, 512*4, 620*4, 713*4],
            #     '1E':[527*4, 570*4, 313*4, 406*4],
            #     '2E':[531*4, 566*4, 407*4, 513*4],
            #     '3E':[531*4, 568*4, 515*4, 620*4],
            #     '4E':[531*4, 568*4, 622*4, 714*4],
            #         }
            self.rgb_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
            rgb = self.rgb_img.copy()
            # rgb = cv2.resize(rgb, (1024, 768), interpolation = cv2.INTER_AREA)
            cv2.imwrite("/home/aurmr/Downloads/test.png", rgb)
            # for bin in bin_bounds:
            #     rgb_cropped = rgb[bin_bounds[bin][0]:bin_bounds[bin][1], bin_bounds[bin][2]:bin_bounds[bin][3], 0:3]
            #     cv2.imwrite("/home/aurmr/workspaces/uois_soofiyan_ws/src/segnetv2_mask2_former/Mask_Results/Data_Collection/rgb_update_"+str(bin)+".png", rgb_cropped.astype(np.uint8))
            print("rgb image shape", self.rgb_img.shape)
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
