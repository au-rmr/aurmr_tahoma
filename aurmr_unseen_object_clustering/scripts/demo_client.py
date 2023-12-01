#!/usr/bin/env python
import cv2
import numpy as np

from uois_service_multi_demo.srv import GetSegmentationUOIS
import rospy
import cv2
import matplotlib.pyplot as plt

# img = np.load("/home/aurmr/workspaces/scenario_based_grasp_point_selection/rgb_img.npy")
# depth_img = np.load("/home/aurmr/workspaces/scenario_based_grasp_point_selection/depth_img.npy")
# plt.imshow(depth_img)
# plt.show()

# np.save("rgb.npy", img)
# np.save("depth_full_size.npy", depth_img)
# depth_img = cv2.resize(depth_img, (640, 360),  interpolation = cv2.INTER_NEAREST)
# plt.imshow(depth_img)
# plt.show()
# np.save("depth_resize.npy", depth_img)
# depth_temp = np.load("/home/aurmr/workspaces/scenario_based_grasp_point_selection/depth_img.npy")
# depth_temp = depth_temp[70:430, :]
# plt.imshow(depth_temp)
# plt.show()
# img_crop = img[137:477, 431:836]
# cv2.imshow("img", img_crop)
# cv2.waitKey(0)
# print(img_crop.shape)
# reshaped_img = img_crop.reshape(-1)
# print(len(reshaped_img))

def uois_segmentation(bin_id):
    rospy.wait_for_service('segmentation_with_embeddings')
    try:
        uois_client = rospy.ServiceProxy('segmentation_with_embeddings', GetSegmentationUOIS)
        req = uois_client(bin_id)
        return req.out_label, req.embeddings
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

if __name__ == "__main__":
    # mask, embed = uois_segmentation(reshaped_img, img_crop.shape[0], img_crop.shape[1])
    mask, embed = uois_segmentation('3F')
    # image = np.load('/home/aurmr/workspaces/aurmr_demo_perception/src/uois_service_multi_demo/dataset/3F/color_0000.npy')
    image = cv2.imread("/home/aurmr/Documents/CVPR_Nov2023/Testing_Data/2E/rgb_image_131.png")
    print(embed)
    print(len(mask), len(embed))
    mask = np.asarray(mask).astype(np.uint8)
    print(mask.shape)
    mask = np.asarray(mask).astype(np.uint8).reshape(image.shape[0], image.shape[1])
    embed = np.asarray(embed).astype(np.float64).reshape(np.max(mask), 256)
    print(mask.shape)

    plt.imshow(mask)
    plt.show()
    # print(embed)
    # segmask = np.zeros((img.shape[0], img.shape[1]))
    # segmask[131:477, 431:809] = mask
    # print(segmask.shape)
    # print(np.unique(segmask))
    # segmask = cv2.resize(segmask, (1280, 720),  interpolation = cv2.INTER_NEAREST)
    # np.save("segmask.npy", segmask)
    # plt.imshow(segmask)
    # plt.show()