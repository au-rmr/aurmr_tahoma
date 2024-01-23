from fileinput import filename
from typing import Dict, List, Set, Tuple
from unittest import result

from aurmr_perception.srv import CaptureObject, RemoveObject, GetObjectPoints, ResetBin, LoadDataset


from skimage.color import label2rgb

import numpy as np

import matplotlib.pyplot as plt
from collections import Counter, defaultdict
# from aurmr_unseen_object_clustering.tools.run_network import clustering_network
# from aurmr_unseen_object_clustering.tools.match_masks import match_masks
from aurmr_unseen_object_clustering.tools.segmentation_net import SegNet, NO_OBJ_STORED, UNDERSEGMENTATION, OBJ_NOT_FOUND, MATCH_FAILED, IN_BAD_BINS
from aurmr_dataset.io import DatasetReader, DatasetWriter
from aurmr_dataset.dataset import Dataset, Item, Entry
import pickle

import scipy.ndimage as spy

from aurmr_perception.util import mask_pointcloud


class BinModel:
    def __init__(self, dataset: Dataset, bin_id: str, bounds, min_bg_change=0.01):
        self.bin = bin_id
        self.bounds = bounds
        self.dataset = dataset
        self.min_bg_change = min_bg_change

        init_depth = dataset.entries[0].depth_image
        self.init_depth = init_depth[bounds[0]:bounds[1], bounds[2]:bounds[3]]
        self.H, self.W = init_depth.shape
        self.bg_mask = np.ones(self.init_depth.shape, dtype=np.uint8)
        self.points_table = {}
        self.masks_table = {}
        self.bad = False

    def crop(self, img_array):
        # RGB might come in with a 4th dim
        return img_array[self.bounds[0]:self.bounds[1], self.bounds[2]:self.bounds[3], 0:3]

    def uncrop(self, img_array):
        full = np.zeros((self.H, self.W), dtype=np.uint8)
        r1, r2, c1, c2 = self.bounds
        full[r1:r2, c1:c2] = img_array
        return full

    def add_new(self, object_id, rgb=None, xyz=None, mask=None, embeddings=None):
        new_entry = {"object_id": object_id, "rgb": None, "depth": None, "mask": None, "embeddings": None, "object_id": object_id, "action": "stow", "n": self.n + 1}
        if rgb is not None:
            new_entry['rgb'] = self.crop(rgb)
        if xyz is not None:
            new_entry['depth'] = self.crop(xyz)
            self.bg_mask = np.abs(new_entry['depth'][...,2] - self.init_depth[...,2]) < self.min_bg_change
        new_entry['embeddings'] = embeddings

        # plt.imshow(self.bg_mask)
        # plt.title("Background Mask")
        # plt.show()
        new_entry['mask'] = mask
        self.history.append(new_entry)

    def update_current(self, rgb=None, depth=None, mask=None, embeddings=None):
        if rgb is not None:
            self.history[-1]['rgb'] = self.crop(rgb)
        if depth is not None:
            self.history[-1]['depth'] = self.crop(depth)
            self.bg_mask = np.abs(self.history[-1]['depth'][...,2] - self.init_depth[...,2]) < self.min_bg_change

        if embeddings is not None:
            self.history[-1]['embeddings'] = embeddings

        if mask is not None:
            self.history[-1]['mask'] = mask

        # plt.imshow(self.bg_mask)
        # plt.title("Background Mask")
        # plt.show()


    def remove(self, object_id: str, rgb, depth):
        del self.points_table[object_id]
        del self.masks_table[object_id]
        self.history.append({"rgb": None, "depth": None, "mask": None, "embeddings": None, "action": "pick", "object_id": object_id, "n": None})

    def reset(self):
        self.masks_table.clear()
        self.points_table.clear()
        self.bg_mask = np.ones(self.init_depth.shape, dtype=np.uint8)
        self.history = [{'rgb':np.zeros(shape=(self.bounds[1] - self.bounds[0], self.bounds[3] - self.bounds[2], 3), dtype=np.uint8), 'depth':None, 'mask':None, 'embeddings':np.array([]), "action": None, "object_id": None}]
        self.bad = False

    def new_object_detected(self, xyz) -> bool:
        # Crop the current depth down to bin size
        r1,r2,c1,c2 = self.bounds
        depth_now = xyz[r1:r2, c1:c2, 2]

        if self.history[-1]['depth'] is None:
            depth_bin = self.init_depth[...,2]
        else:
            depth_bin = self.history[-1]['depth'][...,2]


        mask = (np.abs(depth_bin - depth_now) > self.config['min_pixel_threshhold'])
        kernel = np.ones(shape=(9,9), dtype=np.uint8)
        mask_now = spy.binary_erosion(mask, structure=kernel, iterations=2)

        if np.sum(mask_now) > 0:
            return True
        return False

    @property
    def current(self):
        if len(self.history) == 0:
            return None
        return self.history[-1]

    @property
    def prev(self):
        if len(self.history) < 2:
            return None
        return self.history[-2]

    @property
    def n(self):
        count = 0
        for snapshot in self.history:
            if snapshot["action"] == "stow":
                count += 1
            elif snapshot["action"] == "pick":
                count -= 1
        return count

    @property
    def object_ids_counts(self) -> "Dict[str]":
        contained = defaultdict(int)
        for snapshot in self.history:
            if snapshot["action"] == "stow":
                contained[snapshot["object_id"]] += 1
            elif snapshot["action"] == "pick":
                contained[snapshot["object_id"]] -= 1
        return contained

    def get_history_images(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        return [(snap["rgb"], snap["depth"]) for snap in self.history]

    def get_obj_mask(self, obj_id):
        for entry in reversed(self.history):
            if entry["object_id"] == obj_id:
                mask_crop = entry["mask"]
                obj_id = entry["n"]
                break
        else:
            return None
        return self.uncrop(np.array((mask_crop == id)).astype(np.uint8))


    def refresh_points_table(self, points_msg):
        for o_id in self.object_ids_counts.keys():
            mask = self.get_obj_mask(o_id)
            self.masks_table[o_id] = mask
            self.points_table[o_id] = mask_pointcloud(points_msg, mask)
