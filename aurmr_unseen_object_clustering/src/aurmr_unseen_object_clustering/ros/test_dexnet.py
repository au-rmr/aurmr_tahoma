#!/usr/bin/env python
import numpy as np
# # import sys
# # sys.path.append("/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/aurmr_tahoma/gqcnn/")
# from gqcnn.gqcnn_examples.policy_for_training import dexnet3
# from autolab_core import (YamlConfig, Logger, BinaryImage,
#                           CameraIntrinsics, ColorImage, DepthImage, RgbdImage)
# import cv2


# depth_image = np.load("/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/aurmr_tahoma/aurmr_perception/src/grasp_training/vit_model/depth_full_size.npy")
# segmask_image = np.load("/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/aurmr_tahoma/aurmr_perception/src/grasp_training/vit_model/segmask.npy")

# segmask_image = segmask_image.astype(np.uint8)
# depth_image = cv2.resize(depth_image, (640, 360),  interpolation = cv2.INTER_NEAREST)
# segmask_image = cv2.resize(segmask_image, (640, 360),  interpolation = cv2.INTER_NEAREST)
# depth_image /= 1000.0


# segmask_numpy = np.zeros_like(segmask_image)
# segmask_numpy[segmask_image == 1] = 255

# cx = 320 + 0.839599609375/2
# cy = 180 - (10-5.6268310546875)/2
# fx = 903.14697265625/2
# fy = 903.5240478515625/2
# camera_intrinsics_back_cam = CameraIntrinsics(frame="realsense", fx=fx, fy=fy,
#                                                 cx=cx, cy=cy, skew=0.0,
#                                                 height=360, width=640)

# segmask_dexnet = BinaryImage(
#     segmask_image, frame=camera_intrinsics_back_cam.frame)

# depth_numpy = depth_image
# depth_numpy_temp = depth_numpy*segmask_image
# depth_numpy_temp[depth_numpy_temp == 0] = 0.69
# depth_img_dexnet = DepthImage(
#             depth_image, frame=camera_intrinsics_back_cam.frame)

# dexnet_object = dexnet3(camera_intrinsics_back_cam)
# dexnet_object.load_dexnet_model()
# action, grasps_and_predictions, unsorted_grasps_and_predictions = dexnet_object.inference(
#             depth_img_dexnet, segmask_dexnet, None)


# print(action, grasps_and_predictions)


from gqcnn.gqcnn_examples.policy_for_training import dexnet3
from autolab_core import (YamlConfig, Logger, BinaryImage,
                          CameraIntrinsics, ColorImage, DepthImage, RgbdImage)
# import assets.urdf_models.models_data as md

# from homogeneous_trasnformation_and_conversion.rotation_conversions import *

import matplotlib.pyplot as plt
import cv2

cx = 640/2 + 0.839599609375/2
cy = 360/2 - (10-5.6268310546875)/2
fx = 903.14697265625/2
fy = 903.5240478515625/2
camera_intrinsics_back_cam = CameraIntrinsics(frame="realsense", fx=fx, fy=fy,
                                                cx=cx, cy=cy, skew=0.0,
                                                height=360, width=640)
depth_img = np.load("/home/aurmr/workspaces/uois_dynamo_grasp_rgb/depth_full_size.npy")
segmask = np.load("/home/aurmr/workspaces/uois_dynamo_grasp_rgb/segmask.npy")
depth_img = cv2.resize(depth_img, (640, 360),  interpolation = cv2.INTER_NEAREST)
segmask = cv2.resize(segmask, (640, 360),  interpolation = cv2.INTER_NEAREST)
depth_img /= 1000.0

cx = 640/2 + 0.839599609375/2
cy = 360/2 - (10-5.6268310546875)/2
fx = 903.14697265625/2
fy = 903.5240478515625/2
camera_intrinsics_back_cam = CameraIntrinsics(frame="realsense", fx=fx, fy=fy,
                                                cx=cx, cy=cy, skew=0.0,
                                                height=360, width=640)


dexnet_object = dexnet3(camera_intrinsics_back_cam)
dexnet_object.load_dexnet_model()

segmask_numpy_temp = np.zeros_like(segmask).astype(np.uint8)
segmask_numpy_temp[segmask == 1] = 1
depth_img_temp = depth_img*segmask_numpy_temp
plt.imshow(depth_img)
plt.show()
depth_img_temp[depth_img_temp == 0] = 0.69
depth_img_dexnet = DepthImage(depth_img_temp, frame=camera_intrinsics_back_cam.frame)
# plt.imshow(depth_img_temp)
# plt.show()

segmask_numpy_temp[segmask == 1] = 255
segmask_dexnet = BinaryImage(
    segmask_numpy_temp, frame=camera_intrinsics_back_cam.frame)

# plt.imshow(segmask_numpy_temp)
# plt.show()

action, grasps_and_predictions, unsorted_grasps_and_predictions = dexnet_object.inference(depth_img_dexnet, segmask_dexnet, None)