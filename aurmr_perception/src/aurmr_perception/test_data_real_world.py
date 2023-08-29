import numpy as np
import json
import cv2

segmask_single = np.load("/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/real_world_data/single_object_segmask_8.npy")
segmask = np.load("/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/real_world_data/segmask_8.npy")
rgb= np.load("/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/real_world_data/rgb_image_8.npy")
depth= np.load("/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/real_world_data/depth_image_8.npy")


cv2.imshow("depth", depth*100)
cv2.waitKey(0)

cv2.imshow("segmask_single", segmask_single*50)
cv2.waitKey(0)

cv2.imshow("segmask", segmask*50)
cv2.waitKey(0)


# json_path = "/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/real_world_data/perception_json_data_4_2023-07-27T133119.json"

# with open(json_path) as json_file:
#     data = json.load(json_file)


# cv2.circle(rgb, (data['grasp point'][0], data['grasp point'][1]), 3, (255,255,0), 2)
cv2.imshow("rgb", rgb)
cv2.waitKey(0)
print(data)