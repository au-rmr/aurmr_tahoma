#!/usr/bin/env python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
import glob

from gqcnn.gqcnn_examples.policy_for_training import dexnet3
from autolab_core import (YamlConfig, Logger, BinaryImage,
                          CameraIntrinsics, ColorImage, DepthImage, RgbdImage)

import matplotlib.pyplot as plt
import cv2

import ros_numpy
import rospy

def add_padding(image, channels, value, bin_crop_dim=256):
    # Create padding
    padding = int(bin_crop_dim/2)
    # Define padding for height, width and channels
    if(channels == 3):
        padding_width = ((padding, padding), (padding, padding), (0, 0))
    elif(channels == 1):
        padding_width = ((padding, padding), (padding, padding))
    # Apply padding
    padded_img = np.pad(image, padding_width, mode='constant', constant_values=value)
    return padded_img

rospy.init_node('dexnet_inference', anonymous=True)

data_dir = "/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/PostProcessed_Data/PostProcessed_Data/"
data_list_depth = glob.glob(data_dir + "*_depth_image.png", recursive=True)

for i in range(len(data_list_depth)):
    numbers = re.findall(r'\d+', data_list_depth[i])
    # number = int(numbers[1]) if numbers else None
    number = numbers[0]
    depth_img = cv2.imread(data_list_depth[i])



    gray_values = np.arange(256, dtype=np.uint8)
    color_values = map(tuple, cv2.applyColorMap(gray_values, cv2.COLORMAP_JET).reshape(256, 3))
    color_to_gray_map = dict(zip(color_values, gray_values))
    depth_image = np.apply_along_axis(lambda bgr: color_to_gray_map[tuple(bgr)], 2, depth_img)
    depth_image = (depth_image/255).astype(np.float32)
    
    depth_image = add_padding(depth_image, 1, 0.667, bin_crop_dim=400)
    data_list_mask = glob.glob(data_dir + str(number) + "_*_mask.png", recursive=True)
    
    for j in range(len(data_list_mask)):
        rgb_image = cv2.imread(data_dir + str(number) + "_rgb_image" + ".png")

        mask_all = cv2.imread(data_list_mask[j], 0)
        numbers = re.findall(r'\d+', data_list_mask[j])
        mask_number = int(numbers[1])
        mask_all[mask_all == 255] = 1
        mask_all = add_padding(mask_all, 1, 0, bin_crop_dim=400)

        segmask_image = mask_all
        
        cx = depth_image.shape[1]/2
        cy = depth_image.shape[0]/2
        fx = 903.14697265625
        fy = 903.5240478515625
        camera_intrinsics_back_cam = CameraIntrinsics(frame="realsense", fx=fx, fy=fy,
                                                        cx=cx, cy=cy, skew=0.0,
                                                        height=depth_image.shape[1], width=depth_image.shape[0])

        dexnet_object = dexnet3(camera_intrinsics_back_cam)
        dexnet_object.load_dexnet_model()

        segmask_numpy_temp = np.zeros_like(segmask_image).astype(np.uint8)
        segmask_numpy_temp[segmask_image == 1] = 1
        depth_img_temp = depth_image*segmask_numpy_temp
        depth_img_temp[depth_img_temp == 0] = 0.76
        depth_img_dexnet = DepthImage(depth_img_temp, frame=camera_intrinsics_back_cam.frame)

        segmask_numpy_temp[segmask_image == 1] = 255
        segmask_dexnet = BinaryImage(
            segmask_numpy_temp, frame=camera_intrinsics_back_cam.frame)

        action, grasps_and_predictions, unsorted_grasps_and_predictions = dexnet_object.inference(depth_img_dexnet, segmask_dexnet, None)

        grasp_point = np.array([grasps_and_predictions[0][0].center.x, grasps_and_predictions[0][0].center.y])
        grasp_point = grasp_point - np.array([200, 200])
        rgb_image = cv2.circle(rgb_image,(grasp_point[0], grasp_point[1]), 5, (0,255,0), -1)

        cv2.imwrite(f"/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/PostProcessed_Data/DexNet/{int(number):03d}_{mask_number:01d}_grasp_image.png", rgb_image)
        f = open(f"/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/PostProcessed_Data/DexNet/{int(number):03d}_{mask_number:01d}_grasp_point.txt", "w")
        f.write(f"{grasp_point[0]} {grasp_point[1]}")
        f.close()