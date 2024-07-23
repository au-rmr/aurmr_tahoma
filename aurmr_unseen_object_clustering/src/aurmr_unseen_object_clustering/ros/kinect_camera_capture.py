#!/usr/bin/env python
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import matplotlib.pyplot as plt

global output_image

class image_converter:

  def __init__(self):
    self.bridge = CvBridge()
    self.rgb_sub = rospy.Subscriber("/camera_lower_right/rgb/image_raw", Image, self.callback_rgb)
    # self.depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.callback_depth)

  def callback_rgb(self, data):
    global output_image
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    output_image = cv_image

  # def callback_depth(self, data):
  #   try:
  #       depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
  #   except CvBridgeError as e:
  #     print(e)
  #   depth_array = np.array(depth_image, dtype=np.float32)
  #   depth_image = cv2.resize(depth_image, (1024, 768), interpolation= cv2.INTER_LINEAR)
  #   # cv2.imshow("Depth window", depth_image)
  #   # cv2.waitKey(0)

def main():
  ic = image_converter()
  rospy.init_node('kinect_image_saver', anonymous=True)
  image_counter = 0
  while not rospy.is_shutdown():
    try:
      image = output_image
      i = input()
      if(i == 'v'):
        image = cv2.resize(image, (1024, 768), interpolation= cv2.INTER_LINEAR)
        cv2.imshow("RGB window", image)
        cv2.waitKey(1000)
        cv2.destroyWindow("RGB window")
      elif(i == 's'):
        cv2.imwrite(f"/home/aurmr/Documents/stack_objects_dataset/image_{image_counter}.png", image)
        image_counter += 1
    except:
      pass
  
  print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main()