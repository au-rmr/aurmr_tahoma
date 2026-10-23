#!/usr/bin/env python

from __future__ import print_function
import rospy

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt

def image_callback(data):
    try:
      cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="mono8")
      cv2.imshow("image", cv_image*50)
      cv2.waitKey(25)
    except CvBridgeError as e:
      print(e)

if __name__ == "__main__":

    rospy.init_node('mask_visulaizer', anonymous=True)
    image_sub = rospy.Subscriber("/mask_segment", Image, image_callback)
    bridge = CvBridge()
    rospy.spin()
    

    
