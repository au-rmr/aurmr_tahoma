import cv2
from gqcnn_hardware_comms import TrainDexNetModel, SimpleGQCNN
import matplotlib.pyplot as plt

import numpy as np
import torch

image = cv2.imread("/home/aurmr/workspaces/aurmr_demo_perception/src/segnetv2_mask2_former/Mask_Results/match_rgb1.png")
mask = cv2.imread("/home/aurmr/workspaces/aurmr_demo_perception/src/segnetv2_mask2_former/Mask_Results/after_refine_mask.png", 0)
depth = cv2.imread("/home/aurmr/workspaces/aurmr_demo_perception/src/segnetv2_mask2_former/Mask_Results/depth.png", cv2.IMREAD_UNCHANGED)

depth = depth[1073: 1408, 2146: 2541]/1000.0

object_hardware_comm = TrainDexNetModel(image, depth, mask)
grasp_point = object_hardware_comm.run_dexnet_hardware_comm_inference()

fig, axs = plt.subplots(1, 1, figsize=(12, 3))
grasp_point[0] = grasp_point[0]*image.shape[1]/224
grasp_point[1] = grasp_point[1]*image.shape[0]/224

axs.imshow(image)
axs.add_patch(
    plt.Circle((grasp_point[0], grasp_point[1]), radius=2, color="red", fill=True)
)
axs.axis("off")
plt.tight_layout()
plt.show()
print(grasp_point)