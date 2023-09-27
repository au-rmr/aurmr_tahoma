#!/usr/bin/env python  
import rospy

import math
import tf2_ros
import geometry_msgs.msg
import turtlesim.srv
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import Image, CameraInfo
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import pickle

POD_FACE_C = ['pod_bin_1h', 'pod_bin_2h', 'pod_bin_3h', 'pod_bin_4h',
              'pod_bin_1g', 'pod_bin_2g', 'pod_bin_3g', 'pod_bin_4g',
              'pod_bin_1f', 'pod_bin_2f', 'pod_bin_3f', 'pod_bin_4f',
              'pod_bin_1e', 'pod_bin_2e', 'pod_bin_3e', 'pod_bin_4e',]

POD_FACE_C_FROM_MARKER_X = [0, 0]
POD_FACE_C_FROM_MARKER_Y = {'h': 0.195, 'g': 0.085, 'f': 0.11, 'e': 0.165}

bin_DM_coords_base_link = {}
bin_DM_coords = {}
bin_DM_pixel_coords = {}

bridge = CvBridge()


def fetch_rgb_img(camera='/camera_lower_right/rgb'):
    msg = rospy.wait_for_message(camera + "/image_raw", Image, timeout=None)
    img_rgb = bridge.imgmsg_to_cv2(msg, "bgr8")
    return img_rgb

def fetch_camera_info(camera='/camera_lower_right/rgb'):
    camera_info = rospy.wait_for_message(camera + "/camera_info", CameraInfo, timeout=None)
    fx, cx, fy, cy = camera_info.K[0], camera_info.K[2], camera_info.K[4], camera_info.K[5]
    dist_coeffs = camera_info.D
    return fx, cx, fy, cy, dist_coeffs

def depth_to_point_cloud(depth_image, fx, cx, fy, cy):
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

def fetch_depth_img(camera='/camera_lower_right/depth_to_rgb'):
    msg = rospy.wait_for_message(camera + "/image_raw", Image, timeout=None)
    img_depth = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    return img_depth

def convert_xyz_point_to_uv_point(xyz_point, fx, cx, fy, cy):
    X = xyz_point[0]
    Y = xyz_point[1]
    Z = xyz_point[2]
    U = X*fx/Z + cx
    V = Y*fy/Z + cy

    return U, V

def distorted_point(pt, mtx, dist):
    # Normalize point (remove the effects of the intrinsic camera matrix)
    x_u = (pt[0] - mtx[0, 2]) / mtx[0, 0]
    y_u = (pt[1] - mtx[1, 2]) / mtx[1, 1]
    r2 = x_u**2 + y_u**2
    r4 = r2**2
    r6 = r2*r4
    k1, k2, p1, p2, k3 = dist.flatten()
    # Radial distortion
    x_d = x_u * (1 + k1*r2 + k2*r4 + k3*r6) 
    y_d = y_u * (1 + k1*r2 + k2*r4 + k3*r6) 
    # Tangential distortion
    x_d = x_d + (2*p1*x_u*y_u + p2*(r2 + 2*x_u**2))
    y_d = y_d + (p1*(r2 + 2*y_u**2) + 2*p2*x_u*y_u)
    # Convert back to pixel coordinates
    x_d = x_d * mtx[0, 0] + mtx[0, 2]
    y_d = y_d * mtx[1, 1] + mtx[1, 2]
    return x_d, y_d

if __name__ == '__main__':
    rospy.init_node('DM_tf_listener')

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    while not rospy.is_shutdown():
        for bin in POD_FACE_C:
            try:
                trans_top_left_base_link = tfBuffer.lookup_transform('base_link', bin, rospy.Time())
                trans_right_bottom_base_link = tfBuffer.lookup_transform('base_link', bin, rospy.Time())
                trans_top_left_base_link = np.array([trans_top_left_base_link.transform.translation.x, trans_top_left_base_link.transform.translation.y, trans_top_left_base_link.transform.translation.z]) 
                trans_right_bottom_base_link = np.array([trans_right_bottom_base_link.transform.translation.x, trans_right_bottom_base_link.transform.translation.y, trans_right_bottom_base_link.transform.translation.z])
                bin_DM_coords_base_link[bin] = trans_top_left_base_link

                trans_top_left = tfBuffer.lookup_transform('rgb_camera_link', bin, rospy.Time())
                trans_right_bottom = tfBuffer.lookup_transform('rgb_camera_link', bin, rospy.Time())
                trans_top_left = np.array([trans_top_left.transform.translation.x, trans_top_left.transform.translation.y, trans_top_left.transform.translation.z]) 
                trans_right_bottom = np.array([trans_right_bottom.transform.translation.x, trans_right_bottom.transform.translation.y, trans_right_bottom.transform.translation.z])
                bin_DM_coords[bin] = trans_top_left
                # break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                continue
        if(len(bin_DM_coords) == len(POD_FACE_C)):
            break
    print(bin_DM_coords)

    with open('/tmp/calibration_xyz_coords_pod_base_link.pkl', 'wb') as f:
        pickle.dump(bin_DM_coords_base_link, f)

    with open('/tmp/calibration_xyz_coords_pod.pkl', 'wb') as f:
        pickle.dump(bin_DM_coords, f)

    rgb_img = fetch_rgb_img()
    rgb_img_undistort = rgb_img
    # rgb_img_undistort = undistort_img(rgb_img)

    depth_img = fetch_depth_img()
    depth_img_undistort = depth_img
    # print(depth_img_undistort.shape)

    trans_kinect_base_ink = tfBuffer.lookup_transform('base_link', 'depth_camera_link', rospy.Time())
    # trans_kinect_base_ink = np.array([trans_kinect_base_ink.transform.translation.x, trans_kinect_base_ink.transform.translation.y, trans_kinect_base_ink.transform.translation.z])
    
    rgb_img = cv2.resize(rgb_img_undistort, (int(rgb_img_undistort.shape[1]/4), int(rgb_img_undistort.shape[0]/4)), interpolation = cv2.INTER_AREA)

    for bin in POD_FACE_C:
        bin_id = bin[-1]
        kinect_3F = bin_DM_coords[bin]

        fx, cx, fy, cy, dist_c = fetch_camera_info()
        u,v = convert_xyz_point_to_uv_point(kinect_3F, fx, cx, fy, cy)

        if(bin[8] == '1'):
            kinect_3F[0] += 0.025
        
        kinect_3F[1] -= 0.025

        u1,v1 = convert_xyz_point_to_uv_point(kinect_3F, fx, cx, fy, cy)
        # u1,v1 = distorted_point((u1,v1), mtx, dist)
        # rgb_img = cv2.circle(rgb_img, (int(u1/4), int(v1/4)), 2, (255,255,0), 3)

        if(bin[8] == '1'):
            kinect_3F[1] -= POD_FACE_C_FROM_MARKER_Y[bin_id]
            kinect_3F[0] += 0.23
            kinect_3F[0] -= 0.025
        else:
            kinect_3F[1] -= POD_FACE_C_FROM_MARKER_Y[bin_id]
            kinect_3F[0] += 0.23
        
        if(bin[8] == '4'):
            kinect_3F[0] -= 0.025

        u2,v2 = convert_xyz_point_to_uv_point(kinect_3F, fx, cx, fy, cy)
        # u2,v2 = distorted_point((u2,v2), mtx, dist)
        # rgb_img = cv2.circle(rgb_img, (int(u2/4), int(v2/4)), 2, (255,0,0), 3)
        cv2.rectangle(rgb_img, (int(u2/4), int(v2/4)), (int(u1/4), int(v1/4)), (128, 0, 0), 2)

        bin_DM_pixel_coords[bin[3:]] = np.array([int(v2), int(v1), int(u2), int(u1)])

    print(bin_DM_pixel_coords)

    with open('/tmp/calibration_pixel_coords_pod.pkl', 'wb') as f:
        pickle.dump(bin_DM_pixel_coords, f)
    
    cv2.imshow("rgb", rgb_img)
    cv2.waitKey(0)

    # pub = rospy.Publisher('/kinect/points2', PointCloud2, queue_size=10)
    # while not rospy.is_shutdown():
    #     depth_array = np.array(depth_img_undistort, dtype=np.float32)/1000.0
    #     point_cloud = depth_to_point_cloud(depth_array, fx, cx, fy, cy)
    #     point_cloud = point_cloud.reshape(-1, 3)
    #     header = rospy.Header()
    #     header.stamp = rospy.Time.now()
    #     header.frame_id = 'rgb_camera_link'
    #     pc_msg = point_cloud2.create_cloud_xyz32(header, point_cloud)
    #     pub.publish(pc_msg)
    #     rospy.sleep(1)