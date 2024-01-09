#!/usr/bin/env python
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../../UOIS_multi_frame/'))
# fmt: on

import numpy as np

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config

from mask2former import add_maskformer2_config
from mask2former_frame import add_maskformer2_frame_config
sys.path.insert(1, os.path.join(sys.path[0], '../../UOIS_multi_frame/mask2former_frame/demo/'))
from predictor import VisualizationDemo

from mask2former_group import (
    add_mask2former_group_config,
)
import cv2
import rospy
import glob
import re

WINDOW_NAME = "mask2former demo"
class uois_multi_frame():
    def __init__(self):
        mp.set_start_method("spawn", force=True)
        self.single_image_multi_masks = True

        workspace_path = "/home/aurmr/workspaces/aurmr_demo_perception"

        self.args = {'config_file': f'{workspace_path}/src/UOIS_multi_frame/configs/amazon/256x256_synbin_v0p3_stackXYZ011010_optRPY110100_randXYZ110_2s12345_v6_v76_contra_1p0_softmax_0p1_21k24k.yaml',
                     'sequence': '/home/aurmr/workspaces/test_ros_bare_bone/src/aurmr_perception/VITA_my/data/bin_3F/',
                     'test_type': 'ytvis',
                     'confidence_threshold': 0.5,
                     'opts': ['MODEL.WEIGHTS', f'{workspace_path}/model_final.pth', 'TEST.DETECTIONS_PER_IMAGE', '100',
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

    def inference(self, img):
        sequence_imgs = []
        sequence_imgs.append(img)
        
        embedded_array = []

        if(len(sequence_imgs) == 1 and self.single_image_multi_masks):
            pred_masks, embed = self.demo.run_on_sequence_ytvis_single_image_multi_masks(sequence_imgs)

            full_mask = np.zeros((400,400), dtype = np.uint8)
            if(len(pred_masks) > 0):
                seg_mask = pred_masks[0].squeeze(0).detach().cpu().numpy().astype(np.uint8)
                full_mask = np.zeros(seg_mask.shape, dtype = seg_mask.dtype)
                count_mask_idx = 1
                for i in range(len(pred_masks)):
                    seg_mask = pred_masks[i].squeeze(0).detach().cpu().numpy().astype(np.uint8)
                    if(np.count_nonzero(seg_mask) != 0):
                        full_mask += count_mask_idx*seg_mask
                        full_mask[full_mask > count_mask_idx] = count_mask_idx
                        count_mask_idx += 1
            for object_idx in range(len(embed)):
                if(embed[object_idx] != None):
                    embedded_array.append(embed[object_idx].cpu().numpy())
            embedded_array = np.stack(embedded_array)
            if(len(embedded_array) == 0):
                full_mask = np.zeros((400,400), dtype = np.uint8)

        elif(len(sequence_imgs) == 1):
            pred_masks, embedded_array = self.demo.run_on_sequence_single(sequence_imgs)
            embedded_array = embedded_array.reshape(1, 256)
            full_mask = pred_masks.detach().cpu().numpy().astype(np.uint8)
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
    full_mask, embeddings = object.inference(req.bin_id)
    full_mask = full_mask.reshape(-1)
    embeddings = np.array(embeddings).reshape(-1)
    return full_mask, embeddings

data_dir = "/home/aurmr/Documents/CVPR_Nov2023/Testing_Data/Processed_Data/"
data_list_rgb = glob.glob(data_dir + "/*_rgb_image.png", recursive=True)

for i in range(len(data_list_rgb)):
    numbers = re.findall(r'\d+', data_list_rgb[i])
    number = int(numbers[1]) if numbers else None
    object = uois_multi_frame()
    image = cv2.imread( data_list_rgb[i])
    full_mask, embeddings_array = object.inference(image)
    segmask = full_mask
    mask_list = np.unique(segmask)
    mask_count = 0
    for mask_id in mask_list:
        if mask_id == 0:
            continue
        temp_mask = np.zeros_like(segmask)
        temp_mask[segmask == mask_id] = 255
        temp_mask = temp_mask.astype(np.uint8)
        cv2.imwrite(f"/home/aurmr/Documents/CVPR_Nov2023/Testing_Data/Processed_Data/{number:03d}_{mask_count:01d}_mask.png", temp_mask)
        mask_count += 1