#!/usr/bin/env python
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import matplotlib.pyplot as plt

global rgb_image, depth_image

class image_converter:

  def __init__(self):
    self.bridge = CvBridge()
    self.rgb_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback_rgb)
    self.depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.callback_depth)

  def callback_rgb(self, data):
    global rgb_image
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    rgb_image = cv_image

  def callback_depth(self, data):
    global depth_image
    try:
        depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
    except CvBridgeError as e:
      print(e)
    depth_image = np.array(depth_image, dtype=np.float32)

def main():
  ic = image_converter()
  rospy.init_node('realsense_image_saver', anonymous=True)
  image_counter = 0
  bin_id = '1B_'
  while not rospy.is_shutdown():
    try:
        i = input()
        if(i == 'v'):
            cv2.imshow("RGB window", rgb_image)
            cv2.waitKey(1000)
            cv2.destroyWindow("RGB window")
            cv2.imshow("DEPTH window", depth_image*1000)
            cv2.waitKey(1000)
            cv2.destroyWindow("DEPTH window")
        elif(i == 's'):
            cv2.imwrite(f"/home/aurmr/workspaces/uois_dynamo_grasp_rgb/Near_Range_Data/rgb_image_{image_counter}.png", rgb_image)
            np.save(f"/home/aurmr/workspaces/uois_dynamo_grasp_rgb/Near_Range_Data/depth_image_{image_counter}", depth_image)
            image_counter += 1
        elif(i == 'e'):
            bin_id = input("Enter the bin id: ")
    except:
      pass
  
  print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main()