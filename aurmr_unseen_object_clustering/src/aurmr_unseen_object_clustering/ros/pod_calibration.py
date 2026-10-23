#!/usr/bin/env python

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

"""collect images from Intel RealSense D435"""

import rospy
import message_filters
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
import numpy as np

class ImageListener:

    def __init__(self):

        self.cv_bridge = CvBridge()
        # initialize a node
        rospy.init_node("pod_calibration")
        self.camera_depth_subscriber = rospy.Subscriber('/camera_lower_right/depth_to_rgb/image_raw', Image, self.depth_callback)
        self.camera_rgb_subscriber = rospy.Subscriber('/camera_lower_right/rgb/image_raw', Image, self.rgb_callback)
        msg = rospy.wait_for_message('/camera_lower_right/rgb/camera_info', CameraInfo)

        # save camera intrinsics
        intrinsic_matrix = np.array(msg.K).reshape(3, 3)
        print(intrinsic_matrix)

        # json format
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[0, 0]
        px = intrinsic_matrix[0, 2]
        py = intrinsic_matrix[1, 2]
        camera_params = {'fx': fx, 'fy': fy, 'x_offset': px, 'y_offset': py}

    def depth_callback(self, depth):
        depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)

    def rgb_callback(self, rgb):
        # write images
        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')
        im = cv2.resize(im, (1024, 768), interpolation = cv2.INTER_AREA)
        cv2.imshow("Rgb Image", im)
        cv2.waitKey(0)


if __name__ == '__main__':
    # image listener
    listener = ImageListener()
    try:  
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
