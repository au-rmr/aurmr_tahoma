import cv2
from hardware_inference_comms import TrainRGBModel
import matplotlib.pyplot as plt

import numpy as np

image = cv2.imread("src/Paper evaluation/Dataset/rgb_img_2_16_54_45.png")
mask = cv2.imread("src/Paper evaluation/Dataset/segmask_img_2_16_54_45.png", 0)

plt.imshow(mask)
plt.show()

object_hardware_comm = TrainRGBModel(image, mask)
grasp_point, grasp_angle = object_hardware_comm.train_model()

fig, axs = plt.subplots(1, 1, figsize=(12, 3))
print(image.shape)
grasp_point[0] = grasp_point[0]*image.shape[1]/224
grasp_point[1] = grasp_point[1]*image.shape[0]/224
print(grasp_point, grasp_angle*180/np.pi)

axs.imshow(image)
axs.add_patch(
    plt.Circle((grasp_point[0], grasp_point[1]), radius=2, color="red", fill=True)
)
axs.axis("off")
plt.tight_layout()
plt.show()