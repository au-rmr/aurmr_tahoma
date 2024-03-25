import os
import subprocess
import encodings
from fileinput import filename
from typing import Any, Dict, List, Set, Tuple
from unittest import result
from aurmr_perception.bin_model import BinModel
from aurmr_perception.pod_model import PodModel
import cv2
import matplotlib
from aurmr_perception.srv import CaptureObject, RemoveObject, GetObjectPoints, ResetBin, LoadDataset


from skimage.color import label2rgb

import numpy as np
import ros_numpy
import rospy
import message_filters
import tf2_ros
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from sensor_msgs.msg import Image, CameraInfo, PointField, PointCloud2
from std_msgs.msg import Header
from std_srvs.srv import Trigger
# from aurmr_unseen_object_clustering.tools.run_network import clustering_network
# from aurmr_unseen_object_clustering.tools.match_masks import match_masks
from aurmr_unseen_object_clustering.tools.segmentation_net import SegNet, NO_OBJ_STORED, UNDERSEGMENTATION, OBJ_NOT_FOUND, MATCH_FAILED, IN_BAD_BINS
from aurmr_dataset.io import DatasetReader, DatasetWriter
from aurmr_dataset.dataset import Dataset
import pickle

import scipy.ndimage as spy

from aurmr_perception.util import compute_xyz, mask_pointcloud

class SegNetPodModel(PodModel):

    def __init__(self, dataset: Dataset, camera_name: str) -> None:
        super().__init__(dataset, camera_name)
        self.net = None
        self.occluded_table = {}

    def initialize_with_data(self, dataset: Dataset):
        super().initialize_with_data(dataset)
        depth_img = dataset.entries[-1].depth_image
        camera_intrinsics = dataset.camera_info.K.reshape(3,3)
        self.net = SegNet(init_depth=depth_img, init_info=camera_intrinsics)


    def get_masks(self):
        # Retrieves the mask with bin product id obj_id from the full camera reference frame
        mask = np.zeros((self.H, self.W))
        offset = 0
        for bin in self.bins.values():
            if bin.current['mask'] is not None:
                mask_bin = bin.current['mask'].copy()
                mask_bin[mask_bin > 0] += offset
                r1,r2,c1,c2 = bin.bounds
                mask[r1:r2, c1:c2] = mask_bin

                offset += np.max(mask_bin)

        return mask

    def capture_object(self, bin_id, object_id, rgb_image, depth_image, camera_intrinsics):
        if not bin_id or not object_id:
            return False

        current_rgb = rgb_image

        points = rospy.wait_for_message(f'/camera_lower_right/points2', PointCloud2)
        points = mask_pointcloud(points, mask)

        if bin_id not in self.points_table:
            self.points_table[bin_id] = {}
            self.masks_table[bin_id] = {}

        self.points_table[bin_id][object_id] = points
        self.masks_table[bin_id][object_id] = mask


        # Matches masks with previous mask labels
        # The (i-1)-th entry of recs is the mask id of the best match for mask i in the second image
        if not self.latest_masks:
            self.latest_masks[bin_id] = np.zeros(mask.shape, dtype=np.uint8)

        self.latest_captures[bin_id] = current_rgb
        self.latest_masks[bin_id] = mask

        return True, f"Object {object_id} in bin {bin_id} has been captured with {points.shape[0]} points.", ros_numpy.msgify(Image, mask.astype(np.uint8), encoding="mono8")

    def add_object(self, bin_id, object_id, rgb_image, depth_image, camera_intrinsics, points_msg):
        if not bin_id or not object_id:
            return False, "bin_id and object_id are required"
        intrinsics_3x3 = np.reshape(camera_intrinsics.K, (3,3))
        bin = self.bins[bin_id]
        xyz = compute_xyz(depth_image, intrinsics_3x3)
        # check whether the stow is valid (there is a meaningful pixel difference)
        if not bin.new_object_detected(xyz) and False: #FIXME
            print(f"No new object detected: Did you store an object in bin {bin_id}?\nPlease try again")
            if bin_id not in self.occluded_table:
                self.occluded_table[bin_id] = {}
            self.occluded_table[bin_id][object_id] = True
            msg = f"Object {object_id} in bin {bin_id} was not detected, it is either occluded or was never placed."
            mask_ret = None
            success = False
            return success, msg, mask_ret

        bin.add_new(object_id, rgb_image, xyz)

        # After object with product id obj_id is stowed in bin with id bin_id,
        #           updates perception using the current camera information (rgb_raw, depth_raw, intrinsic)
        # FAILURES:
        #       1. No object is stored. Covered using depth difference in bin.new_object_detected CODE NO_OBJ_STORED
        #       2. Fewer than n objects are segmented (undersegmentation). Covered in refine_masks CODE UNDERSEGMENTATION

        # Check if in bad bins
        if self.bins[bin_id].bad:
            return False, "Bin is already bad. Not updating masks", None


        # Current mask recommendations on the bin
        mask_crop, embeddings = self.net.segment(bin.get_history_images())

        if mask_crop is None:
            bin.bad = True
            return False, "Segmentation failed", None

        # plt.imshow(mask_crop)
        # plt.title("Predicted object mask before refinement")
        # plt.show()

        # Make sure that the bin only has segmentations for n objects
        mask_crop, embeddings = self.net.refine_masks(mask_crop, bin.n, embeddings)

        # mask2vis = self.vis_masks(bin.current['rgb'], mask_crop)
        # plt.imshow(mask2vis)
        # plt.title(f"Masks in the scene. There should be {bin.n}")
        # plt.show()

        if mask_crop is None:
            print(f"Bin {bin_id} added to bad bins. CAUSE Undersegmentation")
            bin.bad = True
            return False, f"Bin {bin_id} added to bad bins. CAUSE Undersegmentation", None

        #mask2vis = self.vis_masks(bin.current['rgb'], mask_crop)
        # plt.imshow(mask2vis)
        # plt.title(f"Masks in the scene (stow). There should be {bin.n} but there are {np.unique(mask_crop)}")
        # plt.show()


        # Find the recommended matches between the two frames
        if bin.prev and bin.prev['mask'] is not None:
            prev_rgb = bin.prev["rgb"]
            cur_rgb = bin.current["rgb"]
            prev_mask = bin.prev["mask"]
            prev_emb = bin.prev["embeddings"]
            recs, match_failed, row_recs = self.net.match_masks_using_embeddings(prev_rgb,cur_rgb, prev_mask, mask_crop, prev_emb, embeddings)

            if(match_failed):
                print(f"WARNING: Embeddings Matching Failure on bin {bin_id}. But not Appending to bad bins yet.")
                recs, sift_failed, row_recs = self.net.match_masks_using_sift(prev_rgb,cur_rgb, prev_mask, mask_crop)
                match_failed = False

                if sift_failed:
                    print(f"WARNING: SIFT Matching Failure on bin {bin_id}. Appending to bad bins.")
                    bin.bad = True

            # Find the index of the new object (not matched)
            for i in range(1, bin.n + 1):
                if i not in recs:
                    recs = np.append(recs, i)
                    break

            # Update the new frame's masks
            mask, embed = self.net.update_masks(mask_crop, recs, embeddings)
        else:
            # This should only happen at the first insertion
            assert(bin.n == 1)
            # No matching is necessary because there is only one object in the scene
            mask = mask_crop.copy()
            embed = embeddings

        bin.update_current(mask=mask, embeddings=embed)

        #mask = self.net.get_masks()

        if False:
            rospy.loginfo("No stow occurred. Returning.")
            return True, f"Object {object_id} in bin {bin_id} hasn't been detected.", ros_numpy.msgify(Image, mask.astype(np.uint8), encoding="mono8")
        # plt.imshow(mask)
        # plt.title(f"get_object all masks for {object_id} in {bin_id}")
        # plt.show()
        rospy.loginfo(np.unique(mask, return_counts=True))
        # mask = (mask != 0).astype(np.uint8)
        cv2.imwrite("/tmp/mask.bmp", mask)
        cv2.imwrite("/tmp/image.png", rgb_image)
        # plt.imshow(mask)
        # plt.show()

        if not bin.bad:
            bin.refresh_points_table(points_msg)
            mask = bin.get_obj_mask(object_id)
            num_points = bin.points_table[object_id].shape[0]
            mask_ret = ros_numpy.msgify(Image, mask.astype(np.uint8), encoding="mono8")
            msg = f"Object {object_id} in bin {bin_id} has been stowed with {num_points} points."
            success = True
        else:
            msg = f"Object {object_id} in bin {bin_id} could not be SIFT matched. Bin state is lost."
            mask_ret = None
            success = False
        # plt.imshow(mask)
        # plt.title(f"add_object {object_id} in {bin_id}")
        # plt.show()

        return success, msg, mask_ret

    def get_object(self, bin_id, object_id, points_msg, im_shape):

        if not object_id:
            return False, "object_id is required"

        if bin_id in self.occluded_table and object_id in self.occluded_table[bin_id]:
            return False, f"Object {object_id} in bin {bin_id} was occluded or never stowed and could not be found"

        if bin_id not in self.bin_names:
            return False, f"Bin {bin_id} was not found"
        bin = self.bins[bin_id]

        if object_id not in bin.object_ids_counts.keys():
            return False, f"Object {object_id} was not found in bin {bin_id}"

        if bin.bad:
            msg = f"Object {object_id} in bin {bin_id} was could not be segmented, but was selected by user"
            return False, msg
        else:
            mask = bin.masks_table[object_id]
            msg = f"Object {object_id} in bin {bin_id} was found"

        points = bin.points_table[object_id]
        # plt.imshow(mask)
        # plt.title(f"get_object {object_id} in {bin_id}")
        # plt.show()
        return (bin_id, points, mask), msg

    def remove_object(self, bin_id, object_id, rgb_image, depth_image, camera_intrinsics, points_msg):
        if bin_id not in self.bin_names:
            return False, f"Bin {bin_id} was not found"

        bin = self.bins[bin_id]
        if object_id not in bin.points_table.keys():
            return False, f"Object {object_id} was not found in bin {bin_id}"


        intrinsics_3x3 = np.reshape(camera_intrinsics.K, (3,3))
        bin.remove(object_id, rgb_image, compute_xyz(depth_image,intrinsics_3x3))

        self.latest_captures[bin_id] = rgb_image

         # Current mask recommendations on the bin
        mask_crop, embeddings = self.segment(bin_id)


        # mask2vis = self.vis_masks(bin.current['rgb'], mask_crop)
        # plt.imshow(mask2vis)
        # plt.title(f"Masks in the scene (pick). There should be {bin.n}")
        # plt.show()

        # Make sure that the bin only has segmentations for n objects
        mask_crop, embeddings = self.refine_masks(mask_crop, bin.n, embeddings)

        if mask_crop is None:
            print(f"Adding {bin_id} to bad bins. Reason: Undersegmentation")
            bin.bad = True
            return UNDERSEGMENTATION

        # Optional visualization
        # if self.config['print_preds']:
        #     plt.imshow(mask_crop)
        #     plt.title("Cropped mask prediction after pick")
        #     plt.show()
        try:
            # FIXME: THIS NEEDS NEW BIN API FOR GETTING N OF THE REMOVED OBJECT
            old_mask = bin.prev['mask'].copy()
            old_embeddings = bin.prev['embeddings'].copy()
            # Remove the object that is no longer in the scene
            old_mask[old_mask == obj_n] = 0
            old_mask[old_mask > obj_n] -= 1
            old_embeddings = np.delete(old_embeddings, obj_n-1, 0)
        except Exception as e:
            print(e)
            old_embeddings = np.ones((1, 256), dtype = np.float64)
            old_mask = np.zeros((256, 256), dtype = np.uint8)

        # Find object correspondence between scenes
        recs, match_failed, row_recs = self.match_masks_using_embeddings(bin.prev['rgb'],bin.current['rgb'], old_mask, mask_crop, old_embeddings, embeddings)
        # print("matching method embeddings and sift", recs, recs_)

        if(match_failed):
            print(f"WARNING: Embeddings Matching Failure on bin {bin_id}. But not Appending to bad bins yet.")
            recs, sift_failed, row_recs = self.net.match_masks_using_sift(bin.prev['rgb'],bin.current['rgb'], old_mask, mask_crop)
            match_failed = False

            if sift_failed:
                print(f"Adding {bin_id} to bad bins. CAUSE: Unconfident matching")
                bin.bad = True
                return MATCH_FAILED

        # Checks that SIFT could accurately mask objects
        # if recs is None:
        #     print(f"SIFT can not confidently determine matches for bin {bin_id}. Reset bin to continue.")
        #     self.bad_bins.append(bin_id)
        #     return 1

        mask, emb = self.net.update_masks(mask_crop, recs, embeddings)
        bin.update_current(mask=mask, embeddings=emb)


        if not bin.bad:
            bin.refresh_points_table(points_msg)

        return True, f"Object {object_id} in bin {bin_id} has been removed"

    def update(self, bin_id, rgb_image, depth_image, camera_intrinsics, points_msg):
        if bin_id not in self.bin_names:
            rospy.logwarn("no bin id")
            return False, f"Bin {bin_id} was not found"

        self.latest_captures[bin_id] = rgb_image
        bin = self.bins[bin_id]

        intrinsics_3x3 = np.reshape(camera_intrinsics.K, (3,3))

        # Update the current state of the bin and scene with the new images
        rgb = rgb_image.astype(np.uint8)
        xyz = compute_xyz(depth_image, intrinsics_3x3)

        bin.update_current(rgb, xyz)

        # Current mask recommendations on the bin
        mask_crop, embeddings = self.net.segment(bin.get_history_images())

        if mask_crop is None:
            bin.bad = True
            return False, "bin state was lost"

        # Make sure that the bin only has segmentations for n objects
        mask_crop, embeddings = self.net.refine_masks(mask_crop, bin.n, embeddings)


        if mask_crop is None:
            bin.bad = True
            return UNDERSEGMENTATION, "bin state was lost"

        # if mask_crop is None:
        #     print(f"Segmentation could not find objects in bin {bin_id}")
        #     input("Need to reset bin. Take out all items and reput them in.")
        #     self.bad_bins.append(bin_id)
        #     self.reset_bin(bin_id)

        # Find the recommended matches between the two frames
        if bin.prev['mask'] is not None:
            recs, match_failed, row_recs = self.net.match_masks_using_embeddings(bin.prev['rgb'],bin.current['rgb'], bin.prev['mask'], mask_crop, bin.prev['embeddings'], embeddings)
            # print("matching method embeddings and sift", recs, recs_)

            if(match_failed):
                print(f"WARNING: Embeddings Matching Failure on bin {bin_id}. But not Appending to bad bins yet.")
                recs, sift_failed, row_recs = self.net.match_masks_using_sift(bin.prev['rgb'],bin.current['rgb'], bin.prev['mask'], mask_crop)
                match_failed = False

                if sift_failed:
                    print(f"WARNING: SIFT Matching Failure on bin {bin_id}. Appending to bad bins.")
                    bin.bad = True

            # if recs is None:
            #     print(f"SIFT could not confidently match bin {bin_id}")
            #     figure, axis = plt.subplots(2,)
            #     axis[0].imshow(bin.last['rgb'])
            #     axis[1].imshow(bin.current['rgb'])
            #     plt.title("Frames that sift failed on")
            #     plt.show()
            #     self.bad_bins.append(bin_id)

            # Update the new frame's masks
            mask, emb = self.net.update_masks(mask_crop, recs, embeddings)
            bin.update_current(mask=mask, embeddings=emb)
            # plt.imshow(bin.current['mask'])
            # plt.title("After update")
            # plt.show()

        else:
            print(bin.prev)
            print("ERROR: THIS SHOULD NEVER HAPPEN!!!")
            return False, "Something went very wrong"


        if not bin.bad:
            bin.refresh_points_table(points_msg)

        return True, f"{bin_id} was updated"

    def reset(self, bin_id, rgb_image):
        if not bin_id:
            return False

        self.bins[bin_id].reset()


        return True
