import numpy as np
import matplotlib.pyplot as plt

# depth_image = np.load("/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/segnetv2_mask2_former/Mask_Results/VLM_model/1699084292_rgb_image.npy")
segmask = np.load("/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/segnetv2_mask2_former/Mask_Results/VLM_model/1699141648_segmask.npy")

plt.imshow(segmask)
plt.show()