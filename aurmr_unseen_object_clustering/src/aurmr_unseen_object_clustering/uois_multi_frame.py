#!/usr/bin/env python
import multiprocessing as mp
import os
from typing import List

from aurmr_setup.utils.workspace_utils import get_active_workspace_path
workspace_path = get_active_workspace_path()
# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], f'{workspace_path}/src/aurmr_perception/'))
# fmt: on

import numpy as np

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config

from mask2former import add_maskformer2_config
from mask2former_frame import add_maskformer2_frame_config
sys.path.insert(1, os.path.join(sys.path[0], f'{workspace_path}/src/aurmr_perception/mask2former_frame/demo/'))
from predictor import VisualizationDemo

from mask2former_group import (
    add_mask2former_group_config,
)


class UOISMultiFrame():
    def __init__(self, config_path, weights_path):
        mp.set_start_method("spawn", force=True)
        self.single_image_multi_masks = True

        self.args = {'config_file': config_path,
                     'sequence': '/home/aurmr/workspaces/test_ros_bare_bone/src/aurmr_perception/VITA_my/data/bin_3F/',
                     'test_type': 'ytvis',
                     'confidence_threshold': 0.5,
                     'opts': ['MODEL.WEIGHTS', weights_path, 'TEST.DETECTIONS_PER_IMAGE', '100',
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

    def inference(self, sequence_imgs: List[np.ndarray]):
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
