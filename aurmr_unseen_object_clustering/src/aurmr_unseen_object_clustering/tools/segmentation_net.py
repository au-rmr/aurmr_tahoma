import os
from operator import truediv
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
from aurmr_unseen_object_clustering.srv import GetSegmentationUOIS
import rospy
import imutils
import pickle
import ros_numpy
from aurmr_setup.utils.workspace_utils import get_active_workspace_path
import sensor_msgs

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

bin_bounds = rospy.get_param("bin_bounds")

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
   'print_preds':False
}


class Bin:
    def __init__(self, bin_id: str, bounds, init_depth=None, config=def_config):
        self.bin = bin_id
        self.visualize = False

        # Total number of bin objects
        self.n = 0

        self.config = config
        self.current = {'rgb':np.zeros(shape=(bounds[1] - bounds[0], bounds[3] - bounds[2], 3), dtype=np.uint8), 'depth':None, 'mask':None, 'embeddings':np.array([])}
        self.last = {'rgb':None, 'depth':None, 'mask':None, 'embeddings':np.array([])}
        self.init_depth = init_depth[bounds[0]:bounds[1], bounds[2]:bounds[3]]
        self.bg_mask = np.ones(self.init_depth.shape, dtype=np.uint8)
        self.prev = {}
        self.bounds = bounds

    def update_current(self, current):
        self.prev["last"] = self.last.copy()
        self.prev["current"] = self.current.copy()
        self.prev["bg_mask"] = self.bg_mask.copy()

        self.last = self.current.copy()

        rgb = current['rgb']
        depth = current['depth']
        self.current['rgb'] = rgb[self.bounds[0]:self.bounds[1], self.bounds[2]:self.bounds[3], ...]
        self.current['depth'] = depth[self.bounds[0]:self.bounds[1], self.bounds[2]:self.bounds[3], ...]
        self.current['embeddings'] = None
        self.bg_mask = np.abs(self.current['depth'][...,2] - self.init_depth[...,2]) < self.config['min_bg_change']
        # plt.imshow(self.bg_mask)
        # plt.title("Background Mask")
        # plt.show()
        self.current['mask'] = None

    def abort_update(self):
        self.last = self.prev["last"]
        self.current = self.prev["current"]
        self.bg_mask = self.prev["bg_mask"]
    def new_object_detected(self, current):
        # Crop the current depth down to bin size
        r1,r2,c1,c2 = self.bounds
        depth_now = current['depth'][r1:r2, c1:c2, 2]

        if self.current['depth'] is None:
            depth_bin = self.init_depth[...,2]
        else:
            depth_bin = self.current['depth'][...,2]


        mask = (np.abs(depth_bin - depth_now) > self.config['min_pixel_threshhold'])
        kernel = np.ones(shape=(9,9), dtype=np.uint8)
        mask_now = spy.binary_erosion(mask, structure=kernel, iterations=2)

        if np.sum(mask_now) > 0:
            return True
        return False


class SegNet:
    def __init__(self, bounds, config=def_config, init_depth=None, init_info=None, mask=None):
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

        # Initializes the bins
        bin_names = bounds.keys()
        self.bins = {}

        init_xyz = self.compute_xyz(init_depth, init_info)
        for bin in bin_names:
            self.bins[bin] = Bin(bin, bounds=bounds[bin], init_depth=init_xyz)


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

        ######################################################################################################
        # print("init pre")
        # sys.path.append("/home/aurmr/workspaces/aurmr_demo_perception/src/segnetv2_mask2_former/UIE_main/mask2former_frame")
        # print("post sys")
        # from demo.segnetv2_demo import SegnetV2
        # print("post import")
        # self.object = SegnetV2()
        # print("post object call")


        # NOTE (henrifung): if these images aren't deleted, not only will it lead to incorrect predictions
        #                   it will also result in an exception when querying UOIS when `sequence length`
        #                   is greater than 10, as this is the sample size of the model
        #
        #                   If this is intended (you are picking from the same bin more than 10 times),
        #                   you may change the sample size in uois_service_multi_demo/scripts/demo_service.py
        workspace_path = get_active_workspace_path()
        import shutil
        rospy.loginfo("Removing directory of cropped bin images...")
        shutil.rmtree(f"{workspace_path}/src/uois_service_multi_demo/dataset/", ignore_errors=True)

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

    def uois_segmentation(self, bin_id):
        rospy.wait_for_service('segmentation_with_embeddings')
        try:
            uois_client = rospy.ServiceProxy('segmentation_with_embeddings', GetSegmentationUOIS)

            path = f"{workspace_path}/src/uois_service_multi_demo/dataset/"
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
                sequence_imgs.append(ros_numpy.msgify(sensor_msgs.msg.Image, img, 'rgb8'))

            req = uois_client(sequence_imgs)
            return req.out_label, req.embeddings
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    # Segments the current bin and returns the masks both from the bin reference and from the overall reference
    def segment(self, bin_id, scene=None):
        # Grab the current bin
        bin = self.bins[bin_id]

        # Get the image and point cloud
        if scene is None:
            rgb = self.current['rgb'].copy()
            xyz = self.current['depth'].copy()
        else:
            rgb = scene['rgb']
            xyz = scene['depth']

        if bin_id == 'syn':
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        H, W, _ = rgb.shape

        # Save the entire pod images for further reproduction
        pod_img_path = f"{workspace_path}/src/uois_service_multi_demo/dataset/pod"
        os.makedirs(pod_img_path, exist_ok=True)
        pod_img_files = os.listdir(pod_img_path)
        indices = [int(f.split('_')[-1].split('.')[0]) for f in pod_img_files if f.startswith('pod_img_') and f.endswith('.png')]
        if indices:
            next_index = max(indices) + 1
        else:
            next_index = 0
        index_str = str(next_index).zfill(4)
        cv2.imwrite(os.path.join(pod_img_path, f"pod_img_{index_str}.png"), rgb)

        # Crops the image to the current bin
        xyz = xyz[bin.bounds[0]:bin.bounds[1], bin.bounds[2]:bin.bounds[3], :]
        rgb = rgb[bin.bounds[0]:bin.bounds[1], bin.bounds[2]:bin.bounds[3], 0:3]

        # rgb = rgb.astype(np.float32)

        # rgb_n = rgb.copy()

        # Standardize image and point cloud
        # if self.config['rgb_is_standardized']:
        #     rgb[..., 0] = (rgb[..., 0] - np.mean(rgb[...,0])) / np.std(rgb[...,0])
        #     rgb[..., 1] = (rgb[..., 1] - np.mean(rgb[...,1])) / np.std(rgb[...,1])
        #     rgb[..., 2] = (rgb[..., 2] - np.mean(rgb[...,2])) / np.std(rgb[...,2])
        if self.config['xyz_is_standardized']:
            if bin_id == 'syn':
                mask = xyz[...,2] > .05
                xyz[..., 0] = (xyz[..., 0] - np.mean(xyz[..., 0], where=mask)) / np.std(xyz[..., 0], where=mask) * mask
                xyz[..., 1] = (xyz[..., 1] - np.mean(xyz[..., 1], where=mask)) / np.std(xyz[..., 1], where=mask) * mask
                xyz[..., 2] = (xyz[..., 2] - np.mean(xyz[..., 2], where=mask)) / np.std(xyz[..., 2], where=mask) * mask
            else:
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

        path = f"{workspace_path}/src/uois_service_multi_demo/dataset/{bin_id}"
        os.makedirs(path, exist_ok=True)
        #path = "/home/aurmr/workspaces/uois_multi_frame_ws/src/uois_service_multi_demo/dataset/"+bin_id+"/"
        files = os.listdir(path)
        indices = [int(f.split('_')[-1].split('.')[0]) for f in files if f.startswith('color_') and f.endswith('.npy')]
        if indices:
            next_index = max(indices) + 1
        else:
            next_index = 0
        index_str = str(next_index).zfill(4)
        file_path = os.path.join(path, f"color_{index_str}.npy")
        np.save(file_path, rgb_segment)
        # Save the image in png format to compare with predicted masks
        png_filepath = file_path.split('.')[0] + '.png'
        cv2.imwrite(png_filepath, rgb_segment)

        # xyz = xyz.astype(np.uint8)

        '''
        old method
        '''
        # masks, out_label = self.object.mask_generator(rgb_segment)

        '''
        new method
        '''
        # mask, embed = self.uois_segmentation(reshaped_img, rgb_segment.shape[0], rgb_segment.shape[1])
        mask, embed = self.uois_segmentation(bin_id)

        try:
            mask = np.asarray(mask).astype(np.uint8).reshape(rgb_segment.shape[0], rgb_segment.shape[1])
        except:
            mask = np.zeros((rgb_segment.shape[0], rgb_segment.shape[1]), dtype = np.uint8)
        embed = np.asarray(embed).astype(np.float64)
        # print(embed)
        if(np.max(mask) > 0):
            embed = embed.reshape(int(len(embed)/256), 256)

        out_label = mask
        out_label_new = out_label.astype(np.uint8)
        # out_label_new = out_label\
        try:
            cv2.imwrite("/home/aurmr/workspaces/aurmr_demo_perception/src/segnetv2_mask2_former/Mask_Results/mask.png", out_label_new.astype(np.uint8))
            cv2.imwrite("/home/aurmr/workspaces/aurmr_demo_perception/src/segnetv2_mask2_former/Mask_Results/rgb.png", rgb_segment.astype(np.uint8))
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

        mask_crop = cv2.resize(mask_crop, (bin.bounds[3] - bin.bounds[2], bin.bounds[1] - bin.bounds[0]), interpolation=cv2.INTER_AREA)

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
        self.const += 1

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
        self.const += 1

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

    # Retrieves the mask with bin product id obj_id from the full camera reference frame
    def get_obj_mask(self, obj_id):
        bin_id, id = self.items[obj_id]
        bin = self.bins[bin_id]
        print(f"get_object segnet: bin_id: {bin_id}, {id}")

        if bin_id in self.bad_bins:
            print(self.bad_bins)
            print("Redirected to manual entry")
            mask_full = self.get_obj_mask_bad_bin(obj_id)
            # plt.imshow(mask_full)
            # plt.title("This is what we are going to send to the robot")
            # plt.show()
            return mask_full

        mask_crop = bin.current['mask']
        mask_full = np.zeros((self.H, self.W), dtype=np.uint8)
        r1, r2, c1, c2 = bin.bounds
        # plt.imshow(mask_crop)
        # plt.title("mask_crop")
        # plt.show()
        mask_full[r1:r2, c1:c2] = np.array((mask_crop == id)).astype(np.uint8)
        return mask_full

    # After object with product id obj_id is stowed in bin with id bin_id,
    #           updates perception using the current camera information (rgb_raw, depth_raw, intrinsic)
    # FAILURES:
    #       1. No object is stored. Covered using depth difference in bin.new_object_detected CODE NO_OBJ_STORED
    #       2. Fewer than n objects are segmented (undersegmentation). Covered in refine_masks CODE UNDERSEGMENTATION
    def stow(self, bin_id, obj_id, rgb_raw, depth_raw=None,info=None, points_raw=None):
        # Grabs the current bin
        bin = self.bins[bin_id]

        if self.H is None:
            self.H, self.W, _ = rgb_raw.shape

        # check whether the stow is valid (there is a meaningful pixel difference)
        if not bin.new_object_detected({'rgb':rgb_raw, 'depth':self.compute_xyz(depth_raw, info)}):
            print(f"No new object detected: Did you store an object in bin {bin_id}?\nPlease try again")
            return NO_OBJ_STORED

        # points_raw[:,:][np.isnan(points_raw)] = 0

        # if not bin.new_object_detected({'rgb':rgb_raw, 'depth':points_raw}):
        #     print(f"Error detected: Did you store an object in bin {bin_id}?\nPlease try again")
        #     return 1

        # Update the current state of the bin and scene with the new images
        self.last = self.current.copy()
        self.current['rgb'] = rgb_raw.astype(np.uint8)
        # self.current['depth'] =
        self.current['depth'] = self.compute_xyz(depth_raw, info)
        self.current['mask'] = None
        bin.update_current(self.current)
        # plt.imshow(self.current['rgb'] )
        # plt.show()

        # Check if in bad bins
        if bin_id in self.bad_bins:
            bin.n += 1
            self.n += 1
            self.items[obj_id] = [bin_id, bin.n]
            return IN_BAD_BINS

        # Keep track of the total number of objects in the bin
        bin.n += 1
        self.n += 1

        # Current mask recommendations on the bin
        mask_crop, embeddings = self.segment(bin_id)

        # plt.imshow(mask_crop)
        # plt.title("Predicted object mask before refinement")
        # plt.show()

        # Make sure that the bin only has segmentations for n objects
        mask_crop, embeddings = self.refine_masks(mask_crop, bin.n, embeddings)

        current_GMT = time.gmtime()
        time_stamp = calendar.timegm(current_GMT)
        self.frame_count_yi += 1
        try:
            cv2.imwrite("/home/aurmr/workspaces/aurmr_demo_perception/src/segnetv2_mask2_former/Mask_Results/Data_Collection/"+str(time_stamp)+"rgb_update_"+str(self.frame_count_yi)+".png", self.current['rgb'][bin.bounds[0]:bin.bounds[1], bin.bounds[2]:bin.bounds[3], 0:3].astype(np.uint8))
            cv2.imwrite("/home/aurmr/workspaces/aurmr_demo_perception/src/segnetv2_mask2_former/Mask_Results/Yi/"+str(time_stamp)+"mask_stow_"+str(self.frame_count_yi)+".png", mask_crop.astype(np.uint8)*40)
            cv2.imwrite("/home/aurmr/workspaces/aurmr_demo_perception/src/segnetv2_mask2_former/Mask_Results/Yi/"+str(time_stamp)+"rgb_stow_"+str(self.frame_count_yi)+".png", self.current['rgb'][bin.bounds[0]:bin.bounds[1], bin.bounds[2]:bin.bounds[3], 0:3].astype(np.uint8))
        except:
            pass
        # mask2vis = self.vis_masks(bin.current['rgb'], mask_crop)
        # plt.imshow(mask2vis)
        # plt.title(f"Masks in the scene. There should be {bin.n}")
        # plt.show()

        if mask_crop is None:
            print(f"Bin {bin_id} added to bad bins. CAUSE Undersegmentation")
            self.items[obj_id] = [bin_id, bin.n]
            self.bad_bins.append(bin_id)
            return UNDERSEGMENTATION
        else:
            if self.visualize:
                cv2.imwrite("/home/aurmr/workspaces/aurmr_demo_perception/src/segnetv2_mask2_former/Mask_Results/refined_mask.png", mask_crop)

        mask2vis = self.vis_masks(bin.current['rgb'], mask_crop)
        # plt.imshow(mask2vis)
        # plt.title(f"Masks in the scene (stow). There should be {bin.n} but there are {np.unique(mask_crop)}")
        # plt.show()


        # Find the recommended matches between the two frames
        if bin.last['mask'] is not None:
            recs, match_failed, row_recs = self.match_masks_using_embeddings(bin.last['rgb'],bin.current['rgb'], bin.last['mask'], mask_crop, bin.last['embeddings'], embeddings)
            # recs_, sift_failed_, row_recs_ = self.match_masks_using_sift(bin.last['rgb'],bin.current['rgb'], bin.last['mask'], mask_crop)

            # print("matching method embeddings and sift", recs, recs_)

            if(match_failed):
                print(f"WARNING: Embeddings Matching Failure on bin {bin_id}. But not Appending to bad bins yet.")
                recs, sift_failed, row_recs = self.match_masks_using_sift(bin.last['rgb'],bin.current['rgb'], bin.last['mask'], mask_crop)
                match_failed = False

                if sift_failed:
                    print(f"WARNING: SIFT Matching Failure on bin {bin_id}. Appending to bad bins.")
                    self.bad_bins.append(bin_id)

            # Find the index of the new object (not matched)
            for i in range(1, bin.n + 1):
                if i not in recs:
                    recs = np.append(recs, i)
                    break

            # Update the new frame's masks
            bin.current['mask'], bin.current['embeddings'] = self.update_masks(mask_crop, recs, embeddings)
            cv2.imwrite("/home/aurmr/workspaces/aurmr_demo_perception/src/segnetv2_mask2_former/Mask_Results/Yi/"+str(time_stamp)+"mask_stow_update_"+str(self.frame_count_yi)+".png", bin.current['mask'].astype(np.uint8)*40)
        else:
            # This should only happen at the first insertion
            assert(bin.n == 1)
            # No matching is necessary because there is only one object in the scene
            bin.current['mask'] = mask_crop.copy()
            bin.current['embeddings'] = embeddings

        # Add the object-bin pair the list of stored items
        self.items[obj_id] = [bin_id, bin.n]

        return 0


    # After a successful pick of object with product id obj_id, updates perception
    #           using camera information (rgb_raw, depth_raw, intrinsic)
    def pick(self, obj_id, rgb_raw, depth_raw=None,info=None, points_raw=None):
        # Finds the bin and local ID for the object
        try:
            bin_id, obj_n = self.items[obj_id]
        except:
            print(f"Object with ID {obj_id} not found in our database. Please try another item.")
            bin.n -= 1
            self.n -= 1
            return OBJ_NOT_FOUND

        # Grabs the current bin
        bin = self.bins[bin_id]


        # points_raw[:,:][np.isnan(points_raw)] = 0


        # Update the current state of the bin and scene with the new images
        self.last = self.current.copy()
        self.current['rgb'] = rgb_raw.astype(np.uint8)
        # self.current['depth'] = points_raw
        self.current['depth'] = self.compute_xyz(depth_raw, info)
        self.current['mask'] = None
        bin.update_current(self.current)

        # Check if in bad bins
        if bin_id in self.bad_bins:
            bin.n -= 1
            self.n -= 1
            del self.items[obj_id]
            return IN_BAD_BINS

        # Current mask recommendations on the bin
        mask_crop, embeddings = self.segment(bin_id)


        # Keep track of the total number of objects in the bin
        bin.n -= 1
        self.n -= 1

        # mask2vis = self.vis_masks(bin.current['rgb'], mask_crop)
        # plt.imshow(mask2vis)
        # plt.title(f"Masks in the scene (pick). There should be {bin.n}")
        # plt.show()

        # Make sure that the bin only has segmentations for n objects
        mask_crop, embeddings = self.refine_masks(mask_crop, bin.n, embeddings)

        current_GMT = time.gmtime()
        time_stamp = calendar.timegm(current_GMT)
        self.frame_count_yi += 1
        try:
            cv2.imwrite("/home/aurmr/workspaces/aurmr_demo_perception/src/segnetv2_mask2_former/Mask_Results/Data_Collection/"+str(time_stamp)+"rgb_update_"+str(self.frame_count_yi)+".png", self.current['rgb'][bin.bounds[0]:bin.bounds[1], bin.bounds[2]:bin.bounds[3], 0:3].astype(np.uint8))
            cv2.imwrite("/home/aurmr/workspaces/aurmr_demo_perception/src/segnetv2_mask2_former/Mask_Results/Yi/"+str(time_stamp)+"mask_pick_"+str(self.frame_count_yi)+".png", mask_crop.astype(np.uint8)*40)
            cv2.imwrite("/home/aurmr/workspaces/aurmr_demo_perception/src/segnetv2_mask2_former/Mask_Results/Yi/"+str(time_stamp)+"rgb_pick_"+str(self.frame_count_yi)+".png", self.current['rgb'][bin.bounds[0]:bin.bounds[1], bin.bounds[2]:bin.bounds[3], 0:3].astype(np.uint8))
            print("successfully saved images")
        except:
            pass

        if mask_crop is None:
            print(f"Adding {bin_id} to bad bins. Reason: Undersegmentation")
            self.bad_bins.append(bin_id)
            del self.items[obj_id]
            return UNDERSEGMENTATION

        # Optional visualization
        # if self.config['print_preds']:
        #     plt.imshow(mask_crop)
        #     plt.title("Cropped mask prediction after pick")
        #     plt.show()
        try:
            old_mask = bin.last['mask'].copy()
            old_embeddings = bin.last['embeddings'].copy()
            # Remove the object that is no longer in the scene
            old_mask[old_mask == obj_n] = 0
            old_mask[old_mask > obj_n] -= 1
            old_embeddings = np.delete(old_embeddings, obj_n-1, 0)
        except:
            old_embeddings = np.ones((1, 256), dtype = np.float64)
            old_mask = np.zeros((256, 256), dtype = np.uint8)

        # Find object correspondence between scenes
        recs, match_failed, row_recs = self.match_masks_using_embeddings(bin.last['rgb'],bin.current['rgb'], old_mask, mask_crop, old_embeddings, embeddings)
        # recs_, sift_failed_, row_recs_ = self.match_masks_using_sift(bin.last['rgb'],bin.current['rgb'], bin.last['mask'], mask_crop)
        # print("matching method embeddings and sift", recs, recs_)

        if(match_failed):
            print(f"WARNING: Embeddings Matching Failure on bin {bin_id}. But not Appending to bad bins yet.")
            recs, sift_failed, row_recs = self.match_masks_using_sift(bin.last['rgb'],bin.current['rgb'], old_mask, mask_crop)
            match_failed = False

            if sift_failed:
                del self.items[obj_id]
                print(f"Adding {bin_id} to bad bins. CAUSE: Unconfident matching")
                self.bad_bins.append(bin_id)
                return MATCH_FAILED

        # Checks that SIFT could accurately mask objects
        # if recs is None:
        #     print(f"SIFT can not confidently determine matches for bin {bin_id}. Reset bin to continue.")
        #     self.bad_bins.append(bin_id)
        #     return 1

        bin.current['mask'], bin.current['embeddings'] = self.update_masks(mask_crop, recs, embeddings)

        # Remove the picked object from the list of tracked items
        for loc_obj_id, (loc_bin, loc_id) in self.items.items():
            if loc_bin == bin_id:
                if loc_id > obj_n:
                    self.items[loc_obj_id] = [loc_bin, loc_id - 1]
        del self.items[obj_id]

        return 0

    # Update the bin state if there's no new object (grasping failure)
    def update(self, bin_id, rgb_raw, depth_raw=None, info=None, points_raw=None):
        # Grabs the current bin
        bin = self.bins[bin_id]

        if self.H is None:
            self.H, self.W, _ = rgb_raw.shape

        # Update the current state of the bin and scene with the new images
        self.last = self.current.copy()
        self.current['rgb'] = rgb_raw.astype(np.uint8)
        # points_raw[:,:][np.isnan(points_raw)] = 0
        # self.current['depth'] = points_raw
        self.current['depth'] = self.compute_xyz(depth_raw, info)
        self.current['mask'] = None
        # print(f"Current: {self.current}")
        bin.update_current(self.current)

        # Current mask recommendations on the bin
        mask_crop, embeddings = self.segment(bin_id)

        # Check if in bad bins
        if bin_id in self.bad_bins:
            return IN_BAD_BINS

        # Make sure that the bin only has segmentations for n objects
        mask_crop, embeddings = self.refine_masks(mask_crop, bin.n, embeddings)

        current_GMT = time.gmtime()
        time_stamp = calendar.timegm(current_GMT)
        self.frame_count_yi += 1
        try:
            cv2.imwrite("/home/aurmr/workspaces/aurmr_demo_perception/src/segnetv2_mask2_former/Mask_Results/Data_Collection/"+str(time_stamp)+"rgb_update_"+str(self.frame_count_yi)+".png", self.current['rgb'][bin.bounds[0]:bin.bounds[1], bin.bounds[2]:bin.bounds[3], 0:3].astype(np.uint8))
            cv2.imwrite("/home/aurmr/workspaces/aurmr_demo_perception/src/segnetv2_mask2_former/Mask_Results/Yi/"+str(time_stamp)+"mask_update_"+str(self.frame_count_yi)+".png", mask_crop.astype(np.uint8)*40)
            cv2.imwrite("/home/aurmr/workspaces/aurmr_demo_perception/src/segnetv2_mask2_former/Mask_Results/Yi/"+str(time_stamp)+"rgb_update_"+str(self.frame_count_yi)+".png", self.current['rgb'][bin.bounds[0]:bin.bounds[1], bin.bounds[2]:bin.bounds[3], 0:3].astype(np.uint8))
        except:
            pass

        # if undersegmentation is detected, it should revert to the previous detected mask
        # currently, something is preventing this from happening despite current and last is reset
        if mask_crop is None:
            bin.abort_update()
            self.current = self.last
            return UNDERSEGMENTATION

        # if mask_crop is None:
        #     print(f"Segmentation could not find objects in bin {bin_id}")
        #     input("Need to reset bin. Take out all items and reput them in.")
        #     self.bad_bins.append(bin_id)
        #     self.reset_bin(bin_id)

        # Find the recommended matches between the two frames
        if bin.last['mask'] is not None:
            recs, match_failed, row_recs = self.match_masks_using_embeddings(bin.last['rgb'],bin.current['rgb'], bin.last['mask'], mask_crop, bin.last['embeddings'], embeddings)
            # recs_, sift_failed_, row_recs_ = self.match_masks_using_sift(bin.last['rgb'],bin.current['rgb'], bin.last['mask'], mask_crop)
            # print("matching method embeddings and sift", recs, recs_)

            if(match_failed):
                print(f"WARNING: Embeddings Matching Failure on bin {bin_id}. But not Appending to bad bins yet.")
                recs, sift_failed, row_recs = self.match_masks_using_sift(bin.last['rgb'],bin.current['rgb'], bin.last['mask'], mask_crop)
                match_failed = False

                if sift_failed:
                    print(f"WARNING: SIFT Matching Failure on bin {bin_id}. Appending to bad bins.")
                    self.bad_bins.append(bin_id)

            # if recs is None:
            #     print(f"SIFT could not confidently match bin {bin_id}")
            #     figure, axis = plt.subplots(2,)
            #     axis[0].imshow(bin.last['rgb'])
            #     axis[1].imshow(bin.current['rgb'])
            #     plt.title("Frames that sift failed on")
            #     plt.show()
            #     self.bad_bins.append(bin_id)

            # Update the new frame's masks
            bin.current['mask'], bin.current['embeddings'] = self.update_masks(mask_crop, recs, embeddings)

            # plt.imshow(bin.current['mask'])
            # plt.title("After update")
            # plt.show()

        else:
            print(bin.last)
            print("ERROR: THIS SHOULD NEVER HAPPEN!!!")
            return 1

        return 0

    def get_masks(self):
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




    def reset_bin(self, bin_id):
        bin = self.bins[bin_id]
        bin.n = 0
        bin.current = {'rgb':None, 'depth':None, 'mask':None, 'embeddings':None}
        bin.last = {'rgb':None, 'depth':None, 'mask':None, 'embeddings':None}

        # Remove all objects in the bin from the database
        for loc_obj_id, (loc_bin, loc_id) in self.items.copy().items():
            if loc_bin == bin_id:
                del self.items[loc_obj_id]

        # Remove bin from list of bad bins
        self.bad_bins.remove(bin_id)

        return 0
