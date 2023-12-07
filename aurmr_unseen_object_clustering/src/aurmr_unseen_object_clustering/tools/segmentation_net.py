import os
from operator import truediv
from typing import List, Tuple
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from aurmr_unseen_object_clustering.lib.fcn.config import cfg, cfg_from_file, get_output_dir
import aurmr_unseen_object_clustering.lib.networks as networks
from aurmr_unseen_object_clustering.lib.fcn.test_dataset import clustering_features, crop_rois, match_label_crop
import matplotlib.pyplot as plt
import cv2

# Mask refinement Inputs
import scipy.ndimage as spy
from skimage.measure import label as lab
from scipy.optimize import linear_sum_assignment

from matplotlib.widgets import RectangleSelector
import sys
import calendar
import time
from aurmr_unseen_object_clustering.uois_multi_frame import UOISMultiFrame
import rospy
import imutils
import pickle

from aurmr_setup.utils.workspace_utils import get_active_workspace_path


rectangle = None


workspace_path = get_active_workspace_path()
UOC_PATH = f'{workspace_path}/src/aurmr_tahoma/aurmr_unseen_object_clustering/src/aurmr_unseen_object_clustering/'

NO_OBJ_STORED = 1
UNDERSEGMENTATION = 2
OBJ_NOT_FOUND = 3
MATCH_FAILED = 4
IN_BAD_BINS = 5

COLORS = {'red':[255,0,0],
          'green':[0,255,0],
          'blue':[0,0,255],
          'magenta':[255,0,255],
          'yellow':[255,255,0],
          'cyan':[0,255,255],
          'purple':[125,0,200]
          }

colors_list = ['red', 'green', 'blue', 'magenta', 'yellow', 'cyan', 'purple']

PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

with open('/tmp/calibration_pixel_coords_pod.pkl', 'rb') as f:
    bin_bounds = pickle.load(f)

print(bin_bounds)

def_config = {
    # Model Parameters
    'model_init':UOC_PATH + 'data/checkpoints/std_200_24_checkpoint.pth',
#    'model_ref':UOC_PATH + 'data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_186.checkpoint_crop.pth',
#    'model_ref':UOC_PATH + 'data/checkpoints/crop_net_current.pth',
#    'model_ref':UOC_PATH + 'data/checkpoints/seg_resnet34_8s_embedding_cosine_color_crop_sampling_epoch_16.checkpoint_justcolor.pth',
    'model_ref': UOC_PATH + 'data/checkpoints/std_200_color_crop.pth',
    'gpu_id':0,
    'cfg_init':UOC_PATH + 'experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml',
#    'cfg_ref':UOC_PATH + 'experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml',
    'cfg_ref':UOC_PATH + 'experiments/cfgs/seg_resnet34_8s_embedding_cosine_color_tabletop.yml',
    'randomize':False,
    'network_name':'seg_resnet34_8s_embedding',
#    'model_ref':None,

    # Bin / setup parameters
   'bounds':bin_bounds,
   'image_path':None,
   'min_pixel_threshhold':30,
   'min_match_count': 3,

   # Segmentation / 
    # Erosion/Dilation and largest connected component
   'perform_cv_ops':False,
   'perform_cv_ops_ref':True,
   # 'rm_back':True,
   'rm_back':True,
   'rm_back_old':False,
   'kernel_size':5,
   'erosion_num':3,
   'dilation_num':4,
   'rgb_is_standardized':True,
   'xyz_is_standardized':True,
   'min_bg_change':15,
   'resize':True,
   'print_preds':False,

    # UOIS
    "uois_config": f'{workspace_path}/src/aurmr_perception/configs/amazon/256x256_synbin_v0p3_stackXYZ011010_optRPY110100_randXYZ110_2s12345_v6_v76_contra_1p0_softmax_0p1_21k24k.yaml',
    "uois_weights": f'{workspace_path}/model_final.pth'
}



class SegNet:
    def __init__(self, config=def_config, init_depth=None, init_info=None, mask=None):
        # init_depth[:,:][np.isnan(init_depth)] = 0
        self.frame_count_yi = 0
        # Initialize the initial network (and refinement network if it exists)
        assert(config['cfg_init'] is not None and config['cfg_ref'] is not None)
        cfg_from_file(config['cfg_init'])
        if len(cfg.TEST.CLASSES) == 0:
            cfg.TEST.CLASSES = cfg.TRAIN.CLASSES
       
        self.visualize = False
        # Locates the device
        cfg.gpu_id = 0
        cfg.device = torch.device('cuda:{:d}'.format(cfg['gpu_id']))
        cfg.instance_id = 0
        num_classes = 2 
        cfg.MODE = 'TEST'
  
        self.config = config
        self.uois_multi_frame = UOISMultiFrame(config["uois_config"], config["uois_weights"])
        
        self.const = 0
  
        # Loads the network for the initial embedding
        # network_data = torch.load(config['model_init'])
        # self.network = networks.__dict__[config['network_name']](num_classes, cfg.TRAIN.NUM_UNITS, network_data).cuda(device=cfg.device)
        # self.network = torch.nn.DataParallel(self.network, device_ids=[cfg.gpu_id]).cuda(device=cfg.device)
        cudnn.benchmark = True
        # self.network.eval()

        # Loads the refinement network
        cfg_from_file(config['cfg_ref'])
        if len(cfg.TEST.CLASSES) == 0:
            cfg.TEST.CLASSES = cfg.TRAIN.CLASSES
        if config['model_ref'] is not None:
            # network_data_crop = torch.load(config['model_ref'])
            # self.network_crop = networks.__dict__[config['network_name']](num_classes, cfg.TRAIN.NUM_UNITS, network_data_crop).cuda(device=cfg.device)
            # self.network_crop = torch.nn.DataParallel(self.network_crop, device_ids=[cfg.gpu_id]).cuda(device=cfg.device)
            # self.network_crop.eval()
            pass
        else:
            self.network_crop = None
 
         # Turns the initial scene into a point cloud
 
        if init_depth is None:
            print("WARNING: No initial scene provided")

        self.H, self.W = None, None
  
        # Contains mappings from pruduct bin ID to [bin, intra_bin_ID]
        self.items = {}
  
        self.n = 0

        self.init_depth = init_depth
        # Stores scene state
        self.current = {'rgb':None, 'depth':None}
        self.last = {'rgb':None, 'depth':None}

        # List of all bins to ignore
        self.bad_bins = []

  
   # Computes the point cloud from a depth array and camera intrinsics
    def compute_xyz(self, depth_img, intrinsic):
        fx = intrinsic[0][0]
        fy = intrinsic[1][1]
        px = intrinsic[0][2]
        py = intrinsic[1][2]
        height, width = depth_img.shape
  
        indices = np.indices((height, width), dtype=np.float32).transpose(1,2,0)
        z_e = depth_img
        x_e = (indices[..., 1] - px) * z_e / fx
        y_e = (indices[..., 0] - py) * z_e / fy
        xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # Shape: [H x W x 3]
        return xyz_img
    
    # Segments the current bin and returns the masks both from the bin reference and from the overall reference
    # Pass in sequence of rgb, xyz pairs already cropped to bin bounds
    def segment(self, rgb_d_sequence: List[Tuple[np.ndarray, np.ndarray]]):
       
        rgb_sequence = [pair[0] for pair in rgb_d_sequence]
        rgb, xyz = rgb_d_sequence[-1]
        # rgb = rgb.astype(np.float32)

        # rgb_n = rgb.copy()
  
        # Standardize image and point cloud
        # if self.config['rgb_is_standardized']:
        #     rgb[..., 0] = (rgb[..., 0] - np.mean(rgb[...,0])) / np.std(rgb[...,0])
        #     rgb[..., 1] = (rgb[..., 1] - np.mean(rgb[...,1])) / np.std(rgb[...,1])
        #     rgb[..., 2] = (rgb[..., 2] - np.mean(rgb[...,2])) / np.std(rgb[...,2])
        if self.config['xyz_is_standardized']:
                xyz[..., 0] = (xyz[..., 0] - np.mean(xyz[...,0])) / np.std(xyz[...,0])
                xyz[..., 1] = (xyz[..., 1] - np.mean(xyz[...,1])) / np.std(xyz[...,1])
                xyz[..., 2] = (xyz[..., 2] - np.mean(xyz[...,2])) / np.std(xyz[...,2])

        if self.config['rm_back_old']:
            sd_x, sd_y = self.config['sd_loc']
            sd = xyz[sd_x, sd_y, 2]
       
        # if self.config['resize']:
        #     rgb = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
        #     xyz = cv2.resize(xyz, (256, 256), interpolation=cv2.INTER_AREA) 
        #     bg_mask_now = cv2.resize(bin.bg_mask.astype(np.uint8), (256, 256), interpolation=cv2.INTER_AREA)

        rgb_segment = rgb.astype(np.uint8)

        # xyz = xyz.astype(np.uint8)

        '''
        old method
        '''
        # masks, out_label = self.object.mask_generator(rgb_segment)

        '''
        new method
        '''
        # mask, embed = self.uois_segmentation(reshaped_img, rgb_segment.shape[0], rgb_segment.shape[1])
        try:
            mask, embed = self.uois_multi_frame.inference(rgb_sequence)
        except Exception as e:
            rospy.logerr("Unseen object segmentation call failed")
            print(e)
            return None, None

        try:
            mask = np.asarray(mask).astype(np.uint8).reshape(rgb_segment.shape[0], rgb_segment.shape[1])
        except:
            mask = np.zeros((rgb_segment.shape[0], rgb_segment.shape[1]), dtype = np.uint8)
        embed = np.asarray(embed).astype(np.float64)
        # print(embed)
        if(np.max(mask) > 0):
            embed = embed.reshape(-1, 256)
        
        out_label = mask
        out_label_new = out_label.astype(np.uint8)
        # out_label_new = out_label\
        try:
            cv2.imwrite(f"{workspace_path}/mask_result_mask.png", out_label_new.astype(np.uint8))
            cv2.imwrite(f"{workspace_path}/mask_result_rgb.png", rgb_segment.astype(np.uint8))
        except:
            pass
        if self.config['perform_cv_ops']:
            kernel = np.ones(shape=(self.config['kernel_size'], self.config['kernel_size']), dtype=np.uint8)
            for i in range(np.max(out_label_new.astype(np.uint8))):
                mask_now = (out_label_new == (i + 1))
  
                mask_now = spy.binary_erosion(mask_now, structure=kernel, iterations=self.config['erosion_num'])
                mask_now = spy.binary_dilation(mask_now, structure=kernel, iterations=self.config['dilation_num'])
  
                labels = lab(mask_now)

                print("labels in perform cv ops", labels)
                if len(np.bincount(labels.flat)[1:]) > 0:
                    mask_now = (labels == (np.argmax(np.bincount(labels.flat)[1:]) + 1))
               
                out_label_new[out_label_new == (i + 1)] =    0
                out_label_new[mask_now] = i + 1
            out_label = torch.reshape(torch.from_numpy(out_label_new).float(), shape=(1, out_label_new.shape[0], out_label_new.shape[1]))

        mask_crop = out_label.astype(np.uint8)

        # Make sure that indices count up from 1
        indices = np.unique(mask_crop)

        mask_crop_f = np.zeros(shape=mask_crop.shape, dtype=np.uint8)
        for i, index in enumerate(indices):
            mask_crop_f[mask_crop == index] = i

        mask_crop = mask_crop_f
        
        #FIXME COMMENTING BECAUSE MASK ALREADY SEEMS T OBE THE RIGHT SIZE
        #mask_crop = cv2.resize(mask_crop, (bin.bounds[3] - bin.bounds[2], bin.bounds[1] - bin.bounds[0]), interpolation=cv2.INTER_AREA)


        return mask_crop, embed
  
    # Returns a best-guess matching between masks in frame 0 and masks in frame 1
    #       Returns a list, where the j - 1th element stores the recommended frame 1 ID of the mask with frame 0 ID j
    def match_masks_using_embeddings(self, im1, im2, mask1, mask2, embed1, embed2):
        match_failed = False
        # try:
        #     print(np.sum(embed1, axis=1), np.sum(embed2, axis=1))
        # except:
        #     pass
        # Convert the images to greyscale

        cv2.imwrite("/home/aurmr/workspaces/aurmr_demo_perception/src/segnetv2_mask2_former/Mask_Results/match_rgb1.png", im1)
        cv2.imwrite("/home/aurmr/workspaces/aurmr_demo_perception/src/segnetv2_mask2_former/Mask_Results/match_rgb2.png", im2)
        cv2.imwrite("/home/aurmr/workspaces/aurmr_demo_perception/src/segnetv2_mask2_former/Mask_Results/match_mask1.png", mask1)
        cv2.imwrite("/home/aurmr/workspaces/aurmr_demo_perception/src/segnetv2_mask2_former/Mask_Results/match_mask2.png", mask2)
        mask_recs = np.zeros(shape=(np.max(mask1), np.max(mask2)))

        for i in range(np.max(mask1)):
            max_score = 0
            for j in range(np.max(mask2)):
                if((np.count_nonzero(embed2[j]) == 0) or (np.count_nonzero(embed1[i]) == 0)): 
                    score = -1
                else:
                    score = np.dot(embed2[j], embed1[i])
                try:
                    mask_recs[i, j] = score
                except:
                    print(np.max(mask1), np.max(mask2))
                    # print(embed1, embed2)
                    print(score)
                if(score > max_score):
                    max_score = score
            if(max_score < -0.2):
                match_failed = True

        print("#############################################")
        print("mask recs", mask_recs)
  
        # Find the mask in the destination image that best matches the soruce mask using Hungarian matching
        row_ind, col_ind = linear_sum_assignment(-mask_recs)
        self.const 
  
        return col_ind + 1, match_failed, row_ind + 1
    
    def match_masks_using_sift(self, im1, im2, mask1, mask2):
        sift_failed = False
        # Convert the images to greyscale
        im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
        im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)

        cv2.imwrite("/home/aurmr/workspaces/aurmr_demo_perception/src/segnetv2_mask2_former/Mask_Results/match_mask1.png", mask1)
        cv2.imwrite("/home/aurmr/workspaces/aurmr_demo_perception/src/segnetv2_mask2_former/Mask_Results/match_mask2.png", mask2)
        mask_recs = np.zeros(shape=(np.max(mask1), np.max(mask2)))
  
        # Calculate keypoints for the frame 1 image
        #sift = cv2.SIFT_create()
        sift = cv2.SIFT_create(nfeatures=150, contrastThreshold=0.04, edgeThreshold=40)
        # sift = cv2.SIFT_create(nfeatures=150)
        k2, d2 = sift.detectAndCompute(im2, None)
  
        # For each mask in frame 0
        for i in range(1, np.max(mask1) + 1):
            # Subset the image from the mask
            im1_now = im1 * (mask1 == i)
  
            # Calculate keypoints for the masked object in frame 0
            k1, d1 = sift.detectAndCompute(im1_now, None)
  
            # Match keypoints between frames 0 and 1
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(d1, d2, 2)
            good = []
            for m, n in matches:
                if m.distance < 0.6 * n.distance:
                    good.append(m)

            # Handle matching failures
            if len(good) < self.config['min_match_count']:
                print("SIFT matching failed.")
                sift_failed = True

            print(self.config['min_match_count'])
            print(f'found {len(good)} sift matches for object {i}. failed={sift_failed}')
            

            # Draws matches for visualization purposes
            matched_img = cv2.drawMatches(im1_now, k1, im2, k2, good, im2, flags=2, matchesThickness=1)
            # cv2.imshow('image', matched_img)
            current_GMT = time.gmtime()

            time_stamp = calendar.timegm(current_GMT)

            cv2.imwrite(f"/home/aurmr/workspaces/aurmr_demo_perception/src/segnetv2_mask2_former/Mask_Results/matched_mask_{i}.png", matched_img)
           
            dst_pts = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.round(dst_pts).astype(int)
  
            # For each mask in frame 1, find out the number of matched keypoints and add it to n_hits
            for j in range(1, np.max(mask2) + 1):
                if dst_pts is not None:
                    n_hits = np.sum(mask2[dst_pts[:, 0, 1], dst_pts[:, 0, 0]] == j)
                    mask_recs[i-1, j-1] = n_hits
                else:
                    mask_recs[i - 1, j - 1] = 0
        print("mask_recs sift", mask_recs)
        # Find the mask in the destination image that best matches the soruce mask sing Hungarian matching
        row_ind, col_ind = linear_sum_assignment(-mask_recs)
        self.const 
  
        return col_ind + 1, sift_failed, row_ind + 1
  
    # Takes two masks and their recommended frame 0/1 relationship and returns the matched mask 2
    def update_masks(self, mask2, recs, embeddings):
        # try:
        #     print("embedding_sum", np.sum(embeddings, axis=1))
        # except:
        #     pass
        print("embeddings recs", recs)
        mask2_new = np.zeros(shape=mask2.shape, dtype=np.uint8)
        embeddings_new = np.empty([np.max(mask2), 256], dtype=np.float64)
        # embeddings_new = np.copy(embeddings)
  
        # For each mask in recs
        for i in range(len(recs)):
            # Find the mask in the result image
            mask_now = (mask2 == recs[i])
            # Add the mask with its new ID to mask2_new
            mask2_new[mask_now] = i + 1
            embeddings_new[recs[i]-1] = embeddings[i]

        # try:
        #     print("new embedding sum", np.sum(embeddings_new, axis=1))
        # except:
        #     pass
        return mask2_new, embeddings_new
  
    # Takes a mask and a number of categories and returns a mask that is guaranteed to have that many categories
    def refine_masks(self, mask, n, embed):
        try:
            print("before refine embeddings", np.sum(embed, axis=1))
        except:
            pass

        try:
            current_GMT = time.gmtime()
            time_stamp = calendar.timegm(current_GMT)
            cv2.imwrite("/home/aurmr/workspaces/aurmr_demo_perception/src/segnetv2_mask2_former/Mask_Results/before_refine_mask.png", mask*30)
            cv2.imwrite("/home/aurmr/workspaces/aurmr_demo_perception/src/segnetv2_mask2_former/Mask_Results/Yi/"+str(time_stamp)+"before_refine_mask"+str(self.frame_count_yi)+".png", mask*30)
        except:
            pass
        embeddings = np.copy(embed)
        if(n == 0):
            return np.zeros(mask.shape, dtype=np.uint8), np.array([])
        
        # heck if do we need to eliminate any masks
        if(np.max(mask) > n):
            # for every extra masks
            for _ in range(np.max(mask) - n):
                areas = np.array([np.sum(mask == i) for i in range(1, np.max(mask) + 1)])
                # find the smallest mask
                smallest_mask_id = np.argmin(areas) + 1
                smallest_mask = np.zeros_like(mask)
                smallest_mask[mask == smallest_mask_id] = 1

                # if the area is too small remove the mask
                if(areas[smallest_mask_id-1] < 100):
                    mask[mask == smallest_mask_id] = 0
                    mask[mask > smallest_mask_id] -= 1
                    embeddings = np.delete(embeddings, smallest_mask_id-1, 0)
                    print("small area thus mask removed")
                    continue

                # extract contours and all then extract top left and bottom right coordinate
                cnts = cv2.findContours(smallest_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)

                c = max(cnts, key=cv2.contourArea)
                extLeft = tuple(c[c[:, :, 0].argmin()][0])
                extRight = tuple(c[c[:, :, 0].argmax()][0])
                extTop = tuple(c[c[:, :, 1].argmin()][0])
                extBot = tuple(c[c[:, :, 1].argmax()][0])
                top_left_smallest = [extLeft[0], extTop[1]]
                bot_right_smallest = [extRight[0], extBot[1]]

                # track if the smallest mask is inside some other mask or is it a false segmentation
                inside_bound = 0
                for j in range(1, np.max(mask)+1):
                    if(j == smallest_mask_id):
                        continue
                    # checking the bounds of each mask and if it is under any of the bounds then we need to merge it or else delete it
                    compare_mask = np.zeros_like(mask)
                    compare_mask[mask == j] = 1
                    cnts = cv2.findContours(compare_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnts = imutils.grab_contours(cnts)
                    c = max(cnts, key=cv2.contourArea)
                    extLeft = tuple(c[c[:, :, 0].argmin()][0])
                    extRight = tuple(c[c[:, :, 0].argmax()][0])
                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    top_left = [extLeft[0], extTop[1]]
                    bot_right = [extRight[0], extBot[1]]

                    # this condition is to check if we can merge them
                    if(top_left[0] <= top_left_smallest[0]+10 and bot_right[0] >= bot_right_smallest[0]-10):
                        if(top_left[1] <= top_left_smallest[1]+10 and bot_right[1] >= bot_right_smallest[1]-10):
                            if(inside_bound == 0):
                                bounds_smallest_top_left = top_left
                                bounds_smallest_bot_right = bot_right
                                inside_bound = j
                            else:
                                # this condition is to check which mask is best to merge with the smallest area
                                if(top_left[0] >= bounds_smallest_top_left[0] and bot_right[0] <= bounds_smallest_bot_right[0]):
                                    if(top_left[1] >= bounds_smallest_top_left[1] and bot_right[1] <= bounds_smallest_bot_right[1]):
                                        bounds_smallest_top_left = top_left
                                        bounds_smallest_bot_right = bot_right
                                        inside_bound = j
                if(inside_bound > 0):
                    mask[mask == smallest_mask_id] = inside_bound
                    mask[mask > smallest_mask_id] -= 1
                    embeddings = np.delete(embeddings, smallest_mask_id-1, 0)
                    print("mask merged")
                else:
                    mask[mask == smallest_mask_id] = 0
                    mask[mask > smallest_mask_id] -= 1
                    embeddings = np.delete(embeddings, smallest_mask_id-1, 0)
                    print("mask not merged thus removed")

        
        # If there are too few masks
        #       split the largest in half along the vertical axis
        if np.max(mask) < n:
            print(f"We only see {np.max(mask)} of {n} masks")
            return None, None
           
            # # Split the largest area along idx into a new mask
            # mask[:, idx:][mask[:, idx:] == idx_max] = np.max(mask) + 1

        print(f"We predicted exactly the right number of masks! That is, {n}")
        try:
            print("after refine embeddings", np.sum(embeddings, axis=1))
        except:
            pass

        try:
            cv2.imwrite("/home/aurmr/workspaces/aurmr_demo_perception/src/segnetv2_mask2_former/Mask_Results/after_refine_mask.png", mask*30)
        except:
            pass
        return mask, embeddings


    def vis_masks(self, image, mask):
        image = image.copy()[...,0:3] * 0.3
        for i in range(1, np.max(mask) + 1):
            mask_color = np.zeros(image.shape, dtype=image.dtype)
            mask_color[mask == i] = COLORS[colors_list[i - 1]]
            image += .7 * mask_color
        
        image = image.astype(np.uint8)
        return image

    def get_obj_mask_bad_bin(self, obj_id):
        bin_id, id = self.items[obj_id]
        bin = self.bins[bin_id]
        # Mask has n ids from 1 to n
        mask, embeddings = self.segment(bin_id)
        # Implement code for using matplotlib to select the relevant mask to return
        mask2vis = self.vis_masks(bin.current['rgb'], mask)

        def onselect(eclick, erelease):
            global rectangle
            rectangle = (eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata)
            # plt.close()
    
        global rectangle
        rectangle = None

        # while rectangle is None:

            # fig, ax = plt.subplots(figsize=(8, 6))
            # __rs = RectangleSelector(ax, onselect)

            # ax.set_title('Select a mask')
            # ax.imshow(mask2vis)
            # plt.show()
        
        print(rectangle)
        print(rectangle[0])
        print(rectangle[0].dtype)
        label = mask[int(rectangle[1]), int(rectangle[0])]
        print("We think the label is ", label)

        bin_mask_crop = (mask == label)

        mask_full = np.zeros((self.H, self.W), dtype=np.uint8)
        r1, r2, c1, c2 = bin.bounds
        # plt.imshow(mask_crop)
        # plt.title("mask_crop")
        # plt.show()
        mask_full[r1:r2, c1:c2] = np.array((bin_mask_crop == id)).astype(np.uint8)

        return  mask_full

        
