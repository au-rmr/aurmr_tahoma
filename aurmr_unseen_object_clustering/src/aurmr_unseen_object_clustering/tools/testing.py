import cv2       
import numpy as np

out_label = cv2.imread('/home/aurmr/workspaces/test_ros_bare_bone/src/aurmr_perception/VITA_my/vis_real/frame_R50_v26_c17_csf/scene_03_full/result.png', 0)
out_label[out_label <= 1] = 0
cv2.imshow('sample image',out_label)
 
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() #