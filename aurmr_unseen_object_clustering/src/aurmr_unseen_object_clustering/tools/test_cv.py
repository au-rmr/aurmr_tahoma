import numpy as np
import cv2 as cv
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from demo.segnetv2_demo import SegnetV2
from copy import copy

def img_save_cb(data):
    global frame

    bridge = CvBridge()

    try:
        frame = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)
    frame = frame[100:250, 35:500, :]
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        exit()

colors = [(0, 0, 255),   # Red
          (0, 255, 0),   # Green
          (255, 0, 0)]   # Blue
global frame
frame = None
rospy.init_node('gripper_cam_preview')
cam = rospy.Subscriber('/gripper_camera/image_raw', Image, img_save_cb)
# net = SegnetV2()
while True:
    # if frame is read correctly ret is True
    if frame is None:
        continue
    # Our operations on the frame come here
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    # cv.imshow('frame', frame)
    # cv.imwrite('frame.png', frame)
    # masks, _ = net.mask_generator(frame)
    # overlay = frame.copy()
    # # Iterate through each mask (assuming masks is a list or array of masks)
    # for idx, mask in enumerate(masks):
    #     # Convert single-channel binary mask to 3-channel RGB
    #     colored_mask = cv.cvtColor(mask.astype(np.uint8), cv.COLOR_GRAY2BGR)


    #     # Apply the random color to the mask
    #     colored_mask[np.where(mask != 0)] = colors[idx % 3]
    #     # Overlay mask on frame
    #     overlay = cv.addWeighted(overlay, 1, colored_mask, 0.5, 0)  # Adjust transparency as needed

    # Display or process the overlayed frame
    # cv.imshow('Overlay', overlay)

    # if cv.waitKey(1) == ord('q'):
    #     break
