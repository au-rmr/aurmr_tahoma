import struct
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from aurmr_unseen_object_clustering.lib.fcn.config import cfg, cfg_from_file, get_output_dir
import aurmr_unseen_object_clustering.lib.networks as networks
from aurmr_unseen_object_clustering.lib.fcn.test_dataset import clustering_features, crop_rois, match_label_crop
from aurmr_unseen_object_clustering.lib.utils.evaluation import multilabel_metrics
import time
import matplotlib.pyplot as plt

# Mask refinement Inputs
import scipy.ndimage as spy
from skimage.measure import label as lab

PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

bounds = {'3f':[780, 1010, 1070, 1320],
          '3d':[1150,1300,1070,1320],
          '2f':[780, 1010, 830, 1080]}


hyperparams = {
    # Choose
    'is_standardized':True,
    'refine_masks':True,

    'erosion_num':2,
    'dilation_num':3,

    'kernel_size':5
}

import cv2

# Change to reflect the path to the UOC directory
UOC_PATH = '/home/aurmr/workspaces/soofiyan_ws/src/aurmr_tahoma/aurmr_unseen_object_clustering/src/aurmr_unseen_object_clustering/'
class clustering_network:
    def __init__(self, bin='3f', camera='azure', dir=UOC_PATH, model='std_200_24_checkpoint.pth'):
        # This dictionary contains all of the relevant arguments for the script
        self.args = {'gpu_id':0, 'pretrained':(dir + 'data/checkpoints/' + model), 'pretrained_crop':None, 'cfg_file':None, 
        'dataset_name':'shapenet_scene_train', 'depth_name':'*depth.png', 'color_name':'*color.png', 
        'imgdir':None, 'randomize':False, 'network_name':'seg_resnet34_8s_embedding', 'image_path':None}

        # Choose which checkpoint to load
        self.args['cfg_file'] = dir + 'experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml'

        # Finds the matching config file
        if self.args['cfg_file'] is not None:
            cfg_from_file(self.args['cfg_file'])
        if len(cfg.TEST.CLASSES) == 0:
            cfg.TEST.CLASSES = cfg.TRAIN.CLASSES

        # Randomizes the seed
        if not self.args['randomize']:
            # fix the random seeds (numpy and caffe) for reproducibility
            np.random.seed(cfg.RNG_SEED)

        self.camera = camera
        self.bounds = bounds[bin]

        # Locates the device
        cfg.gpu_id = 0
        cfg.device = torch.device('cuda:{:d}'.format(cfg['gpu_id']))
        cfg.instance_id = 0
        num_classes = 2
        cfg.MODE = 'TEST'

        # Loads the network for the initial embedding
        network_data = torch.load(self.args['pretrained'])
        self.network = networks.__dict__[self.args['network_name']](num_classes, cfg.TRAIN.NUM_UNITS, network_data).cuda(device=cfg.device)
        self.network = torch.nn.DataParallel(self.network, device_ids=[cfg.gpu_id]).cuda(device=cfg.device)
        cudnn.benchmark = True
        self.network.eval()

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


    def run_net(self, rgb, depth, intrinsic):
        cat = 'frame0_'       

        if self.camera == 'realsense':
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        H, W, _ = rgb.shape

        # Loads the depth image and camera intrinsics

        # Computes the point cloud
        xyz = self.compute_xyz(depth, intrinsic).astype(np.float32)

        # Crops the image
        if self.bounds is not None:
            xyz = xyz[self.bounds[0]:self.bounds[1], self.bounds[2]:self.bounds[3], :]
            rgb = rgb[self.bounds[0]:self.bounds[1], self.bounds[2]:self.bounds[3], 0:3]
        rgb_nice = rgb.copy()
        # if self.camera == 'realsense':
        #     rgb = np.rot90(rgb, k=3).copy()
        #     xyz = np.rot90(xyz, k=3).copy()
        
        # rgb_n = rgb.copy()
        # xyz_n = xyz.copy()
        
        rgb = rgb.astype(np.float32)

        # Standardize image and point cloud
        if hyperparams['is_standardized']:
            rgb[..., 0] = (rgb[..., 0] - np.mean(rgb[...,0])) / np.std(rgb[...,0])
            rgb[..., 1] = (rgb[..., 1] - np.mean(rgb[...,1])) / np.std(rgb[...,1])
            rgb[..., 2] = (rgb[..., 2] - np.mean(rgb[...,2])) / np.std(rgb[...,2])
            xyz[..., 0] = (xyz[..., 0] - np.mean(xyz[...,0])) / np.std(xyz[...,0])
            xyz[..., 1] = (xyz[..., 1] - np.mean(xyz[...,1])) / np.std(xyz[...,1])
            xyz[..., 2] = (xyz[..., 2] - np.mean(xyz[...,2])) / np.std(xyz[...,2])

        sd = xyz[50, 50, 2]

        # Turn everything into tensors and reshape as needed
        im_tensor = torch.from_numpy(rgb) / 255.0
        # im_tensor_n = torch.from_numpy(rgb_n) / 255.0
        # pixel_mean = torch.tensor(PIXEL_MEANS / 255.0).float()
        # im_tensor_n -= pixel_mean
        image_blob = im_tensor.permute(2, 0, 1)
        # image_blob_n = im_tensor_n.permute(2, 0, 1)
        sample = {'image_color': image_blob.unsqueeze(0)}
        # sample['image_color_n'] = image_blob_n.unsqueeze(0)

        depth_blob = torch.from_numpy(xyz).permute(2, 0, 1)
        # depth_blob_n = torch.from_numpy(xyz_n).permute(2, 0, 1)
        sample['depth'] = depth_blob.unsqueeze(0)
        # sample['depth_n'] = depth_blob_n.unsqueeze(0)

        #Inputs: HxWx3 RGB numpy array, HxWx1 depth numpy array, network
        sample['sd_min'] = sd
        
        # Turns inputs into pytorch tensors
        im_tensor = torch.from_numpy(rgb.copy()) / 255.0
        # pixel_mean = torch.tensor(cfg.PIXEL_MEANS / 255.0).float()
        # im_tensor -= pixel_mean
        image_blob = im_tensor.permute(2, 0, 1)
        image = image_blob.unsqueeze(0).cuda()

        depth_blob = torch.from_numpy(xyz.copy().astype(np.float32)).permute(2, 0, 1)
        depth = depth_blob.unsqueeze(0).cuda()

        label = None

        # run network
        features = self.network(image, label, depth).detach()

        # Ignore embeddings corresponding to background pixels
        mask_bool = sample['depth'].squeeze(0).permute(1,2,0)[...,2] > (sample['sd_min'] * 0.8)
        features_now = features.squeeze(0).permute(1,2,0)
        features_base = features_now[0,0]

        features_now[mask_bool == 1] = features_base

        out_label, _ = clustering_features(features, num_seeds=100)
        
        out_label_new = out_label.clone()[0].numpy()
        # Mask refinement step
        if hyperparams['refine_masks']:
            kernel = np.ones(shape=(hyperparams['kernel_size'], hyperparams['kernel_size']), dtype=np.uint8)
            for i in range(np.max(out_label_new.astype(np.uint8))):
                mask_now = (out_label_new == (i + 1))

                mask_now = spy.binary_erosion(mask_now, structure=kernel, iterations=hyperparams['erosion_num'])
                mask_now = spy.binary_dilation(mask_now, structure=kernel, iterations=hyperparams['dilation_num'])

                labels = lab(mask_now)
                if len(np.bincount(labels.flat)[1:]) > 0:
                    print("doing CC")
                    mask_now = (labels == (np.argmax(np.bincount(labels.flat)[1:]) + 1))
                
                out_label_new[out_label_new == (i + 1)] =    0
                out_label_new[mask_now] = i + 1
            out_label = torch.reshape(torch.from_numpy(out_label_new).float(), shape=(1, out_label_new.shape[0], out_label_new.shape[1]))


        # Place masks in original context
        mask_final = np.zeros((H, W))
        if self.camera == 'azure':
            mf = out_label[0].cpu().numpy()
        else:
            mf = out_label[0].cpu().numpy()
            mf = np.rot90(mf, k=1)

        plt.imshow(cv2.cvtColor(rgb_nice, cv2.COLOR_BGR2RGB))
        plt.show()
        plt.imshow(mf)
        plt.show()
            
        mask_final[self.bounds[0]:self.bounds[1], self.bounds[2]:self.bounds[3]] = mf

        return mask_final


