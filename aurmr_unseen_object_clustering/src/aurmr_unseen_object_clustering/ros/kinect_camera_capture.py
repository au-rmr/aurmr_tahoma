#!/usr/bin/env python
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import matplotlib.pyplot as plt

class image_converter:

  def __init__(self):
    self.bridge = CvBridge()
    self.rgb_sub = rospy.Subscriber("/camera_lower_right/rgb/image_raw", Image, self.callback_rgb)
    self.depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.callback_depth)

  def callback_rgb(self, data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    cv_image = cv2.resize(cv_image, (1024, 768), interpolation= cv2.INTER_LINEAR)
    cv2.imshow("Image window", cv_image)
    cv2.waitKey(0)

  def callback_depth(self, data):
    try:
        depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
    except CvBridgeError as e:
      print(e)
    depth_array = np.array(depth_image, dtype=np.float32)
    depth_image = cv2.resize(depth_image, (1024, 768), interpolation= cv2.INTER_LINEAR)
    # cv2.imshow("Depth window", depth_image)
    # cv2.waitKey(0)

def main():
  ic = image_converter()
  rospy.init_node('kinect_image_saver', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main()