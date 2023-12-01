#!/usr/bin/env python
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../../VITA_uiebaseline/'))
# fmt: on

import numpy as np

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config

from mask2former import add_maskformer2_config
from mask2former_frame import add_maskformer2_frame_config
sys.path.insert(1, os.path.join(sys.path[0], '../../VITA_uiebaseline/demo_vita/'))
from predictor import VisualizationDemo

from mask2former_group import (
    add_mask2former_group_config,
)
import cv2
from uois_service_multi_demo.srv import GetSegmentationUOIS
import rospy

WINDOW_NAME = "mask2former demo"
class uois_multi_frame():
    def __init__(self):
        mp.set_start_method("spawn", force=True)

        #workspace_name = os.environ.get('WORKSPACE_NAME', 'aurmr_demo_perception')
        workspace_name = os.environ.get('WORKSPACE_NAME', None)
        if not workspace_name:
            print("ERROR. Unable to determine workspace.")
            sys.exit(1)
        workspace_path = os.path.expanduser(f'~/workspaces/{workspace_name}')

        self.args = {'config_file': f'{workspace_path}/src/VITA_uiebaseline/configs/syn_baselines/vita_R50_bin.yaml',
                'sequence': "/home/aurmr/workspaces/aurmr_demo_perception/src/uois_service_multi_demo/dataset/",
                'test_type': 'ytvis',
                'confidence_threshold': 0.5,
                'opts': ['MODEL.WEIGHTS', f'{workspace_path}/src/VITA_uiebaseline/vita_bin_final.pth', 'TEST.DETECTIONS_PER_IMAGE', '100',
                'DATALOADER.NUM_WORKERS', '0', 'INPUT.SAMPLING_FRAME_NUM', '10', 'MODEL.REID.TEST_MATCH_THRESHOLD', '0.2', 'MODEL.MASK_FORMER.TEST.INFERENCE_THRESHOLD', '0.6']}
        self.cfg = self.setup_cfg(self.args)
        self.demo = VisualizationDemo(self.cfg, test_type=self.args["test_type"])

    def setup_cfg(self, args):
        # load config from file and command-line arguments
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        add_maskformer2_frame_config(cfg)
        if "seq_frame_v2" in args["config_file"]:
            add_mask2former_group_config(cfg)
        cfg.merge_from_file(args["config_file"])
        cfg.merge_from_list(args["opts"])
        cfg.freeze()
        return cfg

    def inference(self, bin_id):
        # path = "/home/aurmr/workspaces/test_ros_bare_bone/src/aurmr_perception/VITA_my/data/"
        path = "/home/aurmr/workspaces/aurmr_demo_perception/src/uois_service_multi_demo/dataset/"
        sequence = path+str(bin_id)+"/"
        file_list = os.listdir(sequence)
        image_list = [f for f in file_list if f.endswith('.npy')]
        image_list.sort()
        frame_names = [os.path.join(sequence, f) for f in image_list]

        sequence_imgs = []
        for frame_idx, frame_path in enumerate(frame_names):
            img = np.load(frame_path)
            # cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            # cv2.imshow(WINDOW_NAME, img)
            # cv2.waitKey(0)
            # img = read_image(frame_path, format="RGB")
            sequence_imgs.append(img)
        
        embedded_array = []
        if(len(sequence_imgs) == 1):
            pred_masks, embedded_array = self.demo.run_on_sequence_single(sequence_imgs)
            embedded_array = embedded_array.reshape(1, 256)
            full_mask = pred_masks.detach().cpu().numpy().astype(np.uint8)
            # full_mask = np.zeros((400,400), dtype = np.uint8)
        else:
            pred_masks, embed = self.demo.run_on_sequence(sequence_imgs)
            # embeddings_array = []
            full_mask = np.zeros((400,400), dtype = np.uint8)
            if(len(pred_masks) > 0):
                seg_mask = pred_masks[0].detach().cpu().numpy().astype(np.uint8)
                full_mask = np.zeros(seg_mask.shape, dtype = seg_mask.dtype)
                count_mask_idx = 1
                for i in range(len(pred_masks)):
                    seg_mask = pred_masks[i].detach().cpu().numpy().astype(np.uint8)
                    if(np.count_nonzero(seg_mask) != 0):
                        full_mask += count_mask_idx*seg_mask
                        full_mask[full_mask > count_mask_idx] = count_mask_idx
                        count_mask_idx += 1
            for object_idx in range(len(embed)):
                if(embed[object_idx][len(sequence_imgs)-1] != None):
                    embedded_array.append(embed[object_idx][len(sequence_imgs)-1])
            try:
                embedded_array = np.stack(embedded_array)
            except:
                pass
            if(len(embedded_array) == 0):
                full_mask = np.zeros((400,400), dtype = np.uint8)
        return full_mask, embedded_array

def return_mask_and_embeddings(req):
    full_mask, embeddings =  object.inference(req.bin_id)
    full_mask = full_mask.reshape(-1)
    try:
        embeddings = embeddings.reshape(-1)
    except:
        embeddings = np.array([])
        full_mask = np.zeros((400,400), dtype = np.uint8)
        full_mask = full_mask.reshape(-1)
    return full_mask, embeddings


if __name__ == "__main__":
    rospy.init_node('service_for_seg_mask_and_embeddings')
    object = uois_multi_frame()
    uois_multi_frame_service = rospy.Service('segmentation_with_embeddings', GetSegmentationUOIS, return_mask_and_embeddings)
    rospy.spin()

# object = uois_multi_frame()
# full_mask, embeddings_array = object.inference("3F")
# print(full_mask)
# print(embeddings_array)
# cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
# cv2.imshow(WINDOW_NAME, full_mask*50)
# cv2.waitKey(0)
