#!/usr/bin/env python
import rospy

import tf2_ros
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import Image, CameraInfo

import pickle
import yaml

columns = [1, 2, 3, 4]
rows = ['h', 'g', 'f', 'e']


bin_DM_coords_base_link = {}
bin_DM_coords = {}
bin_DM_pixel_coords = {}

bridge = CvBridge()


def transform_to_array(trans):
    return np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])

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
    rospy.init_node('bin_bound_maker_pod_1')

    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)

    while not rospy.is_shutdown():
        for row in rows:
            for col in columns:
                try:

                    trans_bottom_left = tf_buffer.lookup_transform('rgb_camera_link', f"pod_bin_{col}{row}_left", rospy.Time())
                    if col < 4:
                        trans_bottom_right = tf_buffer.lookup_transform('rgb_camera_link', f"pod_bin_{col + 1}{row}_left", rospy.Time())
                    else:
                        trans_bottom_right = tf_buffer.lookup_transform('rgb_camera_link', f"pod_bin_{col}{row}_right", rospy.Time())

                    trans_top_left = tf_buffer.lookup_transform('rgb_camera_link', f"pod_bin_{col}{row}_top", rospy.Time())
                    bin_DM_coords[(row, col)] = (transform_to_array(trans_bottom_left), transform_to_array(trans_top_left), transform_to_array(trans_bottom_right))

                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    continue
        if(len(bin_DM_coords) == len(rows) * len(columns)):
            break
    print(bin_DM_coords)

    rgb_img = fetch_rgb_img()
    rgb_img_undistort = rgb_img
    # rgb_img_undistort = undistort_img(rgb_img)

    depth_img = fetch_depth_img()
    depth_img_undistort = depth_img
    # print(depth_img_undistort.shape)

    trans_kinect_base_ink = tf_buffer.lookup_transform('base_link', 'depth_camera_link', rospy.Time())
    # trans_kinect_base_ink = np.array([trans_kinect_base_ink.transform.translation.x, trans_kinect_base_ink.transform.translation.y, trans_kinect_base_ink.transform.translation.z])

    rgb_img = cv2.resize(rgb_img_undistort, (int(rgb_img_undistort.shape[1]/4), int(rgb_img_undistort.shape[0]/4)), interpolation = cv2.INTER_AREA)
    fx, cx, fy, cy, dist_c = fetch_camera_info()
    for row in rows:
        for col in columns:
            bin_id = f"{str(row).upper()}{col}"
            bl, tl, br = bin_DM_coords[(row, col)]

            # Crop out steel frame
            if col == 1:
                bl[0] += 0.03
                tl[0] += 0.03
            elif col == 4:
                br[0] -= 0.03

            # Crop out flap
            bl[1] -= 0.02
            br[1] -= 0.02

            u1,v1 = convert_xyz_point_to_uv_point(br, fx, cx, fy, cy)
            # u1,v1 = distorted_point((u1,v1), mtx, dist)
            # rgb_img = cv2.circle(rgb_img, (int(u1/4), int(v1/4)), 2, (255,255,0), 3)

            u2,v2 = convert_xyz_point_to_uv_point(tl, fx, cx, fy, cy)
            # u2,v2 = distorted_point((u2,v2), mtx, dist)
            # rgb_img = cv2.circle(rgb_img, (int(u2/4), int(v2/4)), 2, (255,0,0), 3)
            cv2.rectangle(rgb_img, (int(u2/4), int(v2/4)), (int(u1/4), int(v1/4)), (128, 0, 0), 2)

            '''
            Need to add other diagonal points which are upper left and bottom right points but the TF which is provided in the urdf is of the other diagonal
            The coordinate system in the urdf of the pod is flipped with respect to the perception peipline used in segmetnation_net.py file
            '''
            bin_DM_pixel_coords[bin_id] = np.array([int(v2), int(v1), int(u1), int(u2)])

    print(bin_DM_pixel_coords)


    with open('/tmp/calibration_pixel_coords_pod.pkl', 'wb') as f:
        pickle.dump(bin_DM_pixel_coords, f)

    with open('/tmp/bin_bounds.yaml', 'w') as f:
        yaml.dump({"bin_bounds": {k: v.tolist() for k, v in bin_DM_pixel_coords.items()}}, f)

    cv2.imshow("rgb", rgb_img)
    cv2.waitKey(0)