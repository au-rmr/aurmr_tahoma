#!/usr/bin/env python
import cv2
import numpy as np

from uois_service_multi_demo.srv import GetSegmentationUOIS
import rospy
import cv2
import matplotlib.pyplot as plt


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
    mask, embed = uois_segmentation('1D')
    image = np.load('/home/aurmr/workspaces/aurmr_demo_perception/src/uois_service_multi_demo/dataset/1D/color_0000.npy')
    
    plt.imshow(image)
    plt.show()
    mask = np.asarray(mask).astype(np.uint8).reshape(image.shape[0], image.shape[1])
    embed = np.asarray(embed).astype(np.float64).reshape(np.max(mask), 256)
    print(mask.shape)