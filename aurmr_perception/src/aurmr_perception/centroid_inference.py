import cv2
import numpy as np
import sys
sys.path.append("/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/segnetv2_mask2_former/UIE_main/mask2former_frame/")
from normal_std.inference_grasp_main import run_normal_std
sys.path.append("/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/aurmr_tahoma/aurmr_perception/src/")
from grasp_training.vit_model.inference_script import inference
import matplotlib.pyplot as plt
import re
import glob

# depth_img = cv2.imread("/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/PostProcessed_Data/PostProcessed_Data/000_depth_image.png")
# rgb_img = cv2.imread("/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/PostProcessed_Data/PostProcessed_Data/000_rgb_image.png")
# mask_all = cv2.imread("/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/PostProcessed_Data/PostProcessed_Data/000_0_mask.png", 0)



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
    
    data_list_mask = glob.glob(data_dir + str(number) + "_*_mask.png", recursive=True)
    
    for j in range(len(data_list_mask)):
        rgb_image = cv2.imread(data_dir + str(number) + "_rgb_image" + ".png")

        mask_all = cv2.imread(data_list_mask[j], 0)
        numbers = re.findall(r'\d+', data_list_mask[j])
        mask_number = int(numbers[1])

        mask_all[mask_all == 255] = 1
        segmask_image = mask_all

        y,x = np.where(segmask_image == 1)

        grasp_point = int(np.mean(x)), int(np.mean(y))
        rgb_image = cv2.circle(rgb_image,(grasp_point[0], grasp_point[1]), 5, (0,255,0), -1)


        cv2.imwrite(f"/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/PostProcessed_Data/Centroid/{int(number):03d}_{mask_number:01d}_grasp_image.png", rgb_image)
        f = open(f"/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/PostProcessed_Data/Centroid/{int(number):03d}_{mask_number:01d}_grasp_point.txt", "w")
        f.write(f"{grasp_point[0]} {grasp_point[1]}")
        f.close()
# grasp_3d_point, grasp_euler_angles, center = inference_object.inference(rgb_img, depth_image, segmask_image, grasp_point)