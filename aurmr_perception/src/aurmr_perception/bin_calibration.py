#!/usr/bin/env python
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
import cv2
import PIL
global rgb_img
rgb_img = None

class Calibration:
    def __init__(self):
        rospy.init_node("bin_calibration_node")
        self.camera_rgb_subscriber = rospy.Subscriber('/camera_lower_right/rgb/image_raw', Image, self.rgb_callback)
        self.rgb_img = None

    def rgb_callback(self, data):
        global rgb_img
        bridge = CvBridge()
        self.rgb_img = bridge.imgmsg_to_cv2(data, "bgr8")

    def extract_pose_from_box(self, box):
        if(box[0][1] == box[3][1]):
            if(box[2][1] != box[1][1]):
                return False
            if((box[0][0]-box[2][0]) != (box[3][0]-box[1][0])):
                return False
            if((box[0][1]-box[1][1]) != (box[3][1]-box[2][1])):
                return False
            if(((box[0][0]-box[2][0]) < (box[1][1]-box[0][1])-1) and ((box[0][0]-box[2][0]) > (box[1][1]-box[0][1])+1)):
                print((box[0][0]-box[2][0]), (box[1][1]-box[0][1]))
                return False
        elif(box[0][1] == box[1][1]):
            if(box[2][1] != box[3][1]):
                return False
            if((box[0][0]-box[1][0]) != (box[3][0]-box[2][0])):
                return False
            if((box[0][1]-box[2][1]) != (box[1][1]-box[3][1])):
                return False
            if(((box[0][0]-box[2][0]) < (box[0][1]-box[2][1])-1) or ((box[0][0]-box[2][0]) > (box[0][1]-box[2][1])+1)):
                print((box[0][0]-box[2][0]), (box[0][1]-box[2][1]))
                return False
        
        return True

    # camera_depth_subscriber = rospy.Subscriber('/camera_lower_right/depth_to_rgb/image_raw', Image, depth_callback)
    def qr_code_detector(self):
        # temp_img = self.rgb_img[424*4:552*4, 620*4:753*4, ...]
        temp_img = self.rgb_img[297*4:373*4, 408*4:542*4, ...]
        # temp_img = self.rgb_img
        well = cv2.cvtColor(np.array(temp_img), cv2.COLOR_BGRA2GRAY)
        harris = cv2.cornerHarris(well,4, 1,0.00)
        x, thr = cv2.threshold(harris, 0.01 * harris.max(), 255, cv2.THRESH_BINARY)
        thr = thr.astype('uint8')
        contours, hierarchy = cv2.findContours(thr.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        areas = map(lambda x: cv2.contourArea(cv2.convexHull(x)), contours)  
        areas = list(areas)  
        max_i = areas.index(max(areas))
        d = cv2.drawContours(np.zeros_like(thr), contours, max_i, 255, 1)
        rect = cv2.minAreaRect(contours[max_i])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        pose = self.extract_pose_from_box(box)
        print(pose, box)
        # pose = True
        if(pose):
            e = cv2.drawContours(well,[box],0,5,5)
            plt.title('E')
            plt.imshow(e)
            plt.show()

        # img = cv2.polylines(self.rgb_img, points.astype(int), True, (0, 255, 0), 3)
        # for s, p in zip(decoded_info, points):
        #     img = cv2.putText(img, s, p[0].astype(int), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        

calib_object = Calibration()
while not rospy.is_shutdown():
    try:
        calib_object.qr_code_detector()
    except:
        pass