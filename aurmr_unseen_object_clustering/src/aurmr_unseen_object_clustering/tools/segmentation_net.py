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

UOC_PATH = '/home/aurmr/workspaces/thomas_ws/src/aurmr_tahoma/aurmr_unseen_object_clustering/src/aurmr_unseen_object_clustering/'

NO_OBJ_STORED = 1
UNDERSEGMENTATION = 2


PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# bin bounds {'item': min_image_height, max_image_height, min_image_width, max_width}

# bin_bounds = {'3fold':[780, 1010, 1070, 1320],
#           '3d':[2300, 2600, 2150, 2630],
#           '3f':[1600, 2000, 2130, 2630],
#           '3d_small':[1150,1300,1070,1320],
#           '2f_old':[780, 1010, 830, 1080],
#           '2f':[1580, 2000, 1650, 2140],
#           '2d':[2300, 2600, 1650, 2150],
#           '2h':[1050, 1280, 1660, 2130]}

# not reachable: 1d, 
bin_bounds = {
          '1H':[1090, 1300, 1270, 1630],
          '2H':[1090, 1300, 1650, 2060],
          '3H':[1090, 1300, 2080, 2500],
          '4H':[1090, 1300, 2500, 2860],
          '1G':[1370, 1530, 1270, 1630],
          '2G':[1370, 1530, 1650, 2060],
          '3G':[1370, 1530, 2080, 2500],
          '4G':[1370, 1530, 2500, 2860],
          '1F':[1600, 1940, 1270, 1630],
          '2F':[1600, 1940, 1650, 2060],
          '3F':[1600, 1940, 2080, 2500],
          '4F':[1600, 1940, 2500, 2860],
          '1E':[2010, 2150, 1270, 1630],
          '2E':[2010, 2150, 1650, 2060],
          '3E':[2010, 2150, 2080, 2500],
          '4E':[2010, 2150, 2500, 2860],
          '1D':[2300, 2540, 1270, 1630],
          '2D':[2300, 2540, 1650, 2060],
          '3D':[2300, 2540, 2080, 2500],
          '4D':[2300, 2540, 2560, 3030],
            }

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
   'min_match_count':0,

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
    def __init__(self, bin_id, bounds=None, init_depth=None, config=def_config):
        # String ID of the bin
        self.bin = bin_id
  
        # The crop bounds of the bin
        if bounds is None:
            self.bounds = bin_bounds[self.bin]
        else:
            self.bounds = bounds
  
        # Total number of bin objects
        self.n = 0

        self.config = config
  
        # Stores bin state
        self.current = {'rgb':np.zeros(shape=(self.bounds[1] - self.bounds[0], self.bounds[3] - self.bounds[2], 3), dtype=np.uint8), 'depth':None, 'mask':None}
        self.last = {'rgb':None, 'depth':None, 'mask':None}
        self.init_depth = init_depth[bounds[0]:bounds[1], bounds[2]:bounds[3]]
        self.bg_mask = np.ones(self.init_depth.shape, dtype=np.uint8)
  
    def update_current(self, current):
        self.last = self.current.copy()
  
        rgb = current['rgb']
        depth = current['depth']
        self.current['rgb'] = rgb[self.bounds[0]:self.bounds[1], self.bounds[2]:self.bounds[3], ...]
        self.current['depth'] = depth[self.bounds[0]:self.bounds[1], self.bounds[2]:self.bounds[3], ...]
        self.bg_mask = np.abs(self.current['depth'][...,2] - self.init_depth[...,2]) < self.config['min_bg_change']
        # plt.imshow(self.bg_mask)
        # plt.title("Background Mask")
        # plt.show()
        self.current['mask'] = None

    def new_object_detected(self, current):
        # Crop the current depth down to bin size
        r1,r2,c1,c2 = self.bounds
        depth_now = current['depth'][r1:r2, c1:c2, 2]
        
        if self.current['depth'] is None:
            depth_bin = self.init_depth[...,2]
        else:
            depth_bin = self.current['depth'][...,2]
        
        # print(depth_bin.dtype)
        # print(depth_now.dtype)  
        # print("previous depth")
        # print(depth_bin)
        # print("Current depth")
        # print(depth_now)

        # print(np.mean(np.abs(depth_bin - depth_now)))
        # print(np.max(np.abs(depth_bin - depth_now)))
        # print(np.min(np.abs(depth_bin - depth_now)))
        

        # plt.imshow(depth_bin)
        # plt.title("Depth bin")
        # plt.show()

        # plt.imshow(depth_now)
        # plt.title("Depth now")
        # plt.show()

        mask = (np.abs(depth_bin - depth_now) > self.config['min_pixel_threshhold'])
        # plt.imshow(np.abs(depth_bin - depth_now))
        # plt.title("Difference")
        # plt.show()
        # plt.imshow(mask)
        # plt.title("This is the mask we see")
        # plt.show()

        kernel = np.ones(shape=(9,9), dtype=np.uint8)

        mask_now = spy.binary_erosion(mask, structure=kernel, iterations=2)

        # plt.imshow(mask_now)
        # plt.title("This is the mask we see")
        # plt.show()

        if np.sum(mask_now) > 0:
            return True
        return False


class SegNet:
    def __init__(self, config=def_config, init_depth=None, init_info=None, mask=None):
        # init_depth[:,:][np.isnan(init_depth)] = 0

        # Initialize the initial network (and refinement network if it exists)
        assert(config['cfg_init'] is not None and config['cfg_ref'] is not None)
        cfg_from_file(config['cfg_init'])
        if len(cfg.TEST.CLASSES) == 0:
            cfg.TEST.CLASSES = cfg.TRAIN.CLASSES
       
        # Locates the device
        cfg.gpu_id = 0
        cfg.device = torch.device('cuda:{:d}'.format(cfg['gpu_id']))
        cfg.instance_id = 0
        num_classes = 2
        cfg.MODE = 'TEST'
  
        self.config = config
        
        self.const = 0
  
        # Loads the network for the initial embedding
        network_data = torch.load(config['model_init'])
        self.network = networks.__dict__[config['network_name']](num_classes, cfg.TRAIN.NUM_UNITS, network_data).cuda(device=cfg.device)
        self.network = torch.nn.DataParallel(self.network, device_ids=[cfg.gpu_id]).cuda(device=cfg.device)
        cudnn.benchmark = True
        self.network.eval()

        # Loads the refinement network
        cfg_from_file(config['cfg_ref'])
        if len(cfg.TEST.CLASSES) == 0:
            cfg.TEST.CLASSES = cfg.TRAIN.CLASSES
        if config['model_ref'] is not None:
            network_data_crop = torch.load(config['model_ref'])
            self.network_crop = networks.__dict__[config['network_name']](num_classes, cfg.TRAIN.NUM_UNITS, network_data_crop).cuda(device=cfg.device)
            self.network_crop = torch.nn.DataParallel(self.network_crop, device_ids=[cfg.gpu_id]).cuda(device=cfg.device)
            self.network_crop.eval()
        else:
            self.network_crop = None
 
         # Turns the initial scene into a point cloud
 
        if init_depth is None:
            print("WARNING: No initial scene provided")

        # Initializes the bins
        bin_names = config['bounds'].keys()
        self.bins = {}

        init_xyz = self.compute_xyz(init_depth, init_info)
        for bin in bin_names:
            self.bins[bin] = Bin(bin, bounds=config['bounds'][bin], init_depth=init_xyz)
  
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

        # plt.imshow(rgb)
        # plt.title("RGB")
        # plt.show()
  
        H, W, _ = rgb.shape
  
        # Crops the image to the current bin
        xyz = xyz[bin.bounds[0]:bin.bounds[1], bin.bounds[2]:bin.bounds[3], :]
        rgb = rgb[bin.bounds[0]:bin.bounds[1], bin.bounds[2]:bin.bounds[3], 0:3]


        
        # plt.imshow(rgb)
        # plt.title("RGB Crop")
        # plt.show()
       
        rgb = rgb.astype(np.float32)

        rgb_n = rgb.copy()
  
        # Standardize image and point cloud
        if self.config['rgb_is_standardized']:
            rgb[..., 0] = (rgb[..., 0] - np.mean(rgb[...,0])) / np.std(rgb[...,0])
            rgb[..., 1] = (rgb[..., 1] - np.mean(rgb[...,1])) / np.std(rgb[...,1])
            rgb[..., 2] = (rgb[..., 2] - np.mean(rgb[...,2])) / np.std(rgb[...,2])
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
       
        if self.config['resize']:
            rgb = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
            xyz = cv2.resize(xyz, (256, 256), interpolation=cv2.INTER_AREA)
            bg_mask_now = cv2.resize(bin.bg_mask.astype(np.uint8), (256, 256), interpolation=cv2.INTER_AREA)
  
        # Turn everything into tensors and reshape as needed
        im_tensor = torch.from_numpy(rgb) / 255.0
        image_blob = im_tensor.permute(2, 0, 1)
        sample = {'image_color': image_blob.unsqueeze(0)}
  
        depth_blob = torch.from_numpy(xyz).permute(2, 0, 1)
        sample['depth'] = depth_blob.unsqueeze(0)
  
        #Inputs: HxWx3 RGB numpy array, HxWx1 depth numpy array, network
        if self.config['rm_back_old']:
            sample['sd_min'] = sd
       
        # Turns inputs into pytorch tensors
        im_tensor = torch.from_numpy(rgb.copy()) / 255.0
        image_blob = im_tensor.permute(2, 0, 1)
        image = image_blob.unsqueeze(0).cuda()

        # Normalizes unnormalized image
        im_tensor_n = torch.from_numpy(rgb_n) / 255.0
        pixel_mean = torch.tensor(PIXEL_MEANS / 255.0).float()
        im_tensor_n -= pixel_mean
        image_blob_n = im_tensor_n.permute(2, 0, 1)
        image_n = image_blob_n.unsqueeze(0).cuda()

        if self.config['resize']:
            # trans = torchvision.transforms.Compose([
            #     torchvision.transforms.ToPILImage(),
            #     torchvision.transforms.Resize((256, 256)), 
            #     torchvision.transforms.ToTensor()])
            image_n = image_n.cpu()[0].permute(1,2,0).numpy()
            image_n = cv2.resize(image_n, (256, 256), interpolation=cv2.INTER_AREA)
            image_n = torch.from_numpy(image_n).permute(2,0,1).unsqueeze(0).cuda()
            
            # print(image_n.shape)
            # image_n = trans(image_n[0].cpu()).unsqueeze(0).cuda()
  
        depth_blob = torch.from_numpy(xyz.copy().astype(np.float32)).permute(2, 0, 1)
        depth = depth_blob.unsqueeze(0).cuda()
  
        label = None
  
        # run network
        features = self.network(image, label, depth).detach()
  
        # Ignore embeddings corresponding to background pixels
        if self.config['rm_back_old']:
            mask_bool = sample['depth'].squeeze(0).permute(1,2,0)[...,2] > (sample['sd_min'] * 0.8)
            features_now = features.squeeze(0).permute(1,2,0)
            features_base = features_now[0,0]
  
            features_now[mask_bool == 1] = features_base

        if self.config['rm_back']:
            features_now = features.squeeze(0).permute(1,2,0)
            features_base = features_now[0,0]
  
            features_now[bg_mask_now == 1] = features_base
            # plt.imshow(bg_mask_now)
            # plt.title("BG Mask")
            # plt.show()
  
        out_label, _ = clustering_features(features, num_seeds=100)

        out_label_new = out_label.clone()[0].numpy()
        # print(np.unique(out_label_new))
        # plt.imshow(out_label_new.astype(np.uint8))
        # plt.title("Segmentation")
        # plt.show()
  
        # Mask refinement step
        if self.config['perform_cv_ops']:
            kernel = np.ones(shape=(self.config['kernel_size'], self.config['kernel_size']), dtype=np.uint8)
            for i in range(np.max(out_label_new.astype(np.uint8))):
                mask_now = (out_label_new == (i + 1))
  
                mask_now = spy.binary_erosion(mask_now, structure=kernel, iterations=self.config['erosion_num'])
                mask_now = spy.binary_dilation(mask_now, structure=kernel, iterations=self.config['dilation_num'])
  
                labels = lab(mask_now)
                if len(np.bincount(labels.flat)[1:]) > 0:
                    mask_now = (labels == (np.argmax(np.bincount(labels.flat)[1:]) + 1))
               
                out_label_new[out_label_new == (i + 1)] =    0
                out_label_new[mask_now] = i + 1
            out_label = torch.reshape(torch.from_numpy(out_label_new).float(), shape=(1, out_label_new.shape[0], out_label_new.shape[1]))

        # Runs refinement network
        if self.network_crop:
            # zoom in refinement
            out_label_refined = None
            if self.network_crop is not None:
                # unique_labels = torch.unique(out_label)
                # initial_masks = torch.zeros((len(unique_labels), out_label.shape[1], out_label.shape[2]), dtype=bool)
                # for unique_label in unique_labels:
                #     initial_masks[out_label == unique_label] = 1
                rgb_crop, out_label_crop, rois, depth_crop = crop_rois(image_n, out_label, depth)
                if rgb_crop.shape[0] > 0:
                    features_crop = self.network_crop(rgb_crop, out_label_crop, depth_crop)
                    labels_crop, selected_pixels_crop = clustering_features(features_crop)
                    out_label_refined, labels_crop = match_label_crop(out_label, labels_crop.cuda(), out_label_crop, rois, depth_crop)
                    # for rgb_crop_i in range(len(rgb_crop)):
                    #     image_n[0, :, int(rois[rgb_crop_i, 1].item()):int(rois[rgb_crop_i, 3].item()), int(rois[rgb_crop_i, 0].item()):int(rois[rgb_crop_i, 2].item())] = 0
                else:
                    out_label_refined = out_label
                # trans = torchvision.transforms.ToPILImage()
                # plt.imshow(trans(image_n[0].cpu()))
                # plt.show()
                # plt.close()

                if self.config['perform_cv_ops_ref']:
                    kernel = np.ones(shape=(self.config['kernel_size'], self.config['kernel_size']), dtype=np.uint8)
                    out_label_new = out_label_refined.clone()[0].numpy()
                    for i in range(np.max(out_label_new.astype(np.uint8)) ):
                        mask_now = out_label_new == (i + 1)
                        
                        mask_now = spy.binary_erosion(mask_now, structure=kernel, iterations=3)
                        mask_now = spy.binary_dilation(mask_now, structure=kernel, iterations=4)

                        labels = lab(mask_now)
                        if len(np.bincount(labels.flat)[1:]) > 0:
                            mask_now = (labels == (np.argmax(np.bincount(labels.flat)[1:]) + 1))
                        
                        out_label_new[out_label_new == (i + 1)] = 0
                        out_label_new[mask_now] = i + 1 
                    out_label_ref = torch.reshape(torch.from_numpy(out_label_new).float(), shape=(1, out_label_new.shape[0], out_label_new.shape[1]))
                else:
                    out_label_ref = out_label_refined
  
        # Converts mask back to original size and format
        if self.network_crop is not None:
            mask_crop = out_label_ref[0].cpu().numpy().astype(np.uint8)
        else:
            mask_crop = out_label[0].cpu().numpy().astype(np.uint8)

        # Make sure that indices count up from 1
        indices = np.unique(mask_crop)

        mask_crop_f = np.zeros(shape=mask_crop.shape, dtype=np.uint8)
        for i, index in enumerate(indices):
            mask_crop_f[mask_crop == index] = i

        # print(f"I find indices {indices}")

        
        mask_crop = mask_crop_f

        
        mask_crop = cv2.resize(mask_crop, (bin.bounds[3] - bin.bounds[2], bin.bounds[1] - bin.bounds[0]), interpolation=cv2.INTER_AREA)
  
        return mask_crop
  
    # Returns a best-guess matching between masks in frame 0 and masks in frame 1
    #       Returns a list, where the j - 1th element stores the recommended frame 1 ID of the mask with frame 0 ID j
    def match_masks(self, im1, im2, mask1, mask2):
        sift_failed = False
        # Convert the images to greyscale
        im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
        im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
  
        mask_recs = np.zeros(shape=(np.max(mask1), np.max(mask2)))
  
        # Calculate keypoints for the frame 1 image
        sift = cv2.SIFT_create()
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
                if m.distance < 0.7 * n.distance:
                    good.append(m)

            # Handle matching failures
            if len(good) < self.config['min_match_count']:
                sift_failed = True
            
            # Draws matches for visualization purposes
            # matched_img = cv2.drawMatches(im1_now, k1, im2, k2, good, im2, flags=2, matchesThickness=1)
            # cv2.imshow('image', matched_img)
            # cv2.imwrite(f'matched_img_{self.const}{i}.png', matched_img)
           
            dst_pts = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.round(dst_pts).astype(int)
  
            # For each mask in frame 1, find out the number of matched keypoints and add it to n_hits
            for j in range(1, np.max(mask2) + 1):
                if dst_pts is not None:
                    n_hits = np.sum(mask2[dst_pts[:, 0, 1], dst_pts[:, 0, 0]] == j)
                    mask_recs[i-1, j-1] = n_hits
                else:
                    mask_recs[i - 1, j - 1] = 0
  
        # Find the mask in the destination image that best matches the soruce mask sing Hungarian matching
        row_ind, col_ind = linear_sum_assignment(-mask_recs)
        self.const += 1
  
        return col_ind + 1, sift_failed
  
    # Takes two masks and their recommended frame 0/1 relationship and returns the matched mask 2
    def update_masks(self, mask2, recs):
        print("UPDATING_MASKS")
        mask2_new = np.zeros(shape=mask2.shape, dtype=np.uint8)
  
        # For each mask in recs
        for i in range(len(recs)):
            # Find the mask in the result image
            mask_now = (mask2 == recs[i])
            # Add the mask with its new ID to mask2_new
            mask2_new[mask_now] = i + 1
  
        return mask2_new
  
    # Takes a mask and a number of categories and returns a mask that is guaranteed to have that many categories
    def refine_masks(self, mask, n):
        while np.max(mask) > n:
            areas = np.array([np.sum(mask == i) for i in range(1, np.max(mask) + 1)])
            # Finds the ID of the largest cluster
            idx_max = np.argmax(areas) + 1

            nonzero_idx = np.where(np.sum(mask == idx_max, axis=1) > 0)
            c1 = nonzero_idx[0]
            c2 = nonzero_idx[-1]

            print(f"I think that the bounds for the largest mask are {c1} to {c2}")

            plt.imshow(mask)
            plt.title("Mask in refine_masks")
            plt.show()

            # Find the smallest area under the largest mask

            idx_min = np.argmin(areas) + 1
            nonzero_idx_under = np.where(np.sum(mask == idx_min, axis=1) > 0)
            c1_under = nonzero_idx_under[0]
            c2_under = nonzero_idx_under[-1]

            # While the area found isn't beneath the largest mask
            while c1_under < (c1 - 10) or c2_under > (c2 + 10):
                # Remove it from areas
                areas = np.delete(areas, idx_min - 1)

                # If we've gone through all areas, break
                if areas.shape[0] == 0:
                    break

                # Recalculate smallest component
                idx_min = np.argmin(areas) + 1
                nonzero_idx_under = np.where(np.sum(mask == idx_min, axis=1) > 0)
                c1_under = nonzero_idx_under[0]
                c2_under = nonzero_idx_under[-1]
            
            if areas.shape[0] == 0:
                break

            # Otherwise, a mask was found. Merge them.
            mask[mask == idx_min] = idx_max
            mask[mask > idx_min] -= 1

        # If there are STILL too many masks
        #       remove the smallest
        while np.max(mask) > n:
            print("Merge method failed. Removing excess")
            # Calculate areas for each mask
            areas = np.array([np.sum(mask == i) for i in range(1, np.max(mask) + 1)])
            idx_min = np.argmin(areas) + 1
            # Update masks
            mask[mask == idx_min] = 0
            mask[mask > idx_min] -= 1

        
        # If there are too few masks
        #       split the largest in half along the vertical axis
        while np.max(mask) < n:
            print(f"We only see {np.max(mask)} of {n} masks")
            print("We did not collect the relevant masks. Jiggle them?")
            input("Have you jiggled?")
            return None
           
            # # Split the largest area along idx into a new mask
            # mask[:, idx:][mask[:, idx:] == idx_max] = np.max(mask) + 1

        print("We predicted exactly the right number of masks!")
  
        return mask
  
    # Retrieves the mask with bin product id obj_id from the full camera reference frame
    def get_obj_mask(self, obj_id):
        bin_id, id = self.items[obj_id]
        bin = self.bins[bin_id]
        print(f"get_object segnet: bin_id: {bin_id}, {id}")
        print(bin.bounds)
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
            print(f"Error detected: Did you store an object in bin {bin_id}?\nPlease try again")
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
  
        # Current mask recommendations on the bin
        mask_crop = self.segment(bin_id)

        # plt.imshow(mask_crop)
        # plt.title("Predicted object mask before refinement")
        # plt.show()

        # Keep track of the total number of objects in the bin
        bin.n += 1
        self.n += 1
  
        # Make sure that the bin only has segmentations for n objects
        mask_crop = self.refine_masks(mask_crop, bin.n)

        if mask_crop is None:
            return UNDERSEGMENTATION

            
        assert(np.max(mask_crop) == bin.n)
  
        # Find the recommended matches between the two frames
        if bin.last['mask'] is not None:
            recs, sift_failed = self.match_masks(bin.last['rgb'],bin.current['rgb'], bin.last['mask'], mask_crop)

            if sift_failed:
                print(f"WARNING: SIFT Matching Failure on bin {bin_id}. Appending to bad bins.")
                self.bad_bins.append(bin_id)
  
            # Find the index of the new object (not matched)
            for i in range(1, bin.n + 1):
                if i not in recs:
                    recs = np.append(recs, i)
                    break
            
            # Update the new frame's masks
            bin.current['mask'] = self.update_masks(mask_crop, recs).copy()
       
        else:
            # This should only happen at the first insertion
            assert(bin.n == 1)
            # No matching is necessary because there is only one object in the scene
            bin.current['mask'] = mask_crop.copy()

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
            return 1

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
  
        # Current mask recommendations on the bin
        mask_crop = self.segment(bin_id)

        # Keep track of the total number of objects in the bin
        bin.n -= 1
        self.n -= 1
  
        # Make sure that the bin only has segmentations for n objects
        mask_crop = self.refine_masks(mask_crop, bin.n)

        if mask_crop is None:
            return GO_TO_UPDATE
  
        assert(np.max(mask_crop) == bin.n)

        # Optional visualization
        if self.config['print_preds']:
            plt.imshow(mask_crop)
            plt.title("Cropped mask prediction after pick")
            plt.show()
  
        # You should only pick if there's an object in the bin
        assert(bin.last['mask'] is not None)
  
        old_mask = bin.last['mask'].copy()
        # Remove the object that is no longer in the scene
        old_mask[old_mask == obj_n] = 0
        old_mask[old_mask > obj_n] -= 1

        # Find object correspondence between scenes
        recs = self.match_masks(bin.last['rgb'], bin.current['rgb'], old_mask, mask_crop)

        # Checks that SIFT could accurately mask objects
        if recs is None:
            print(f"SIFT can not confidently determine matches for bin {bin_id}. Reset bin to continue.")
            self.bad_bins.append(bin_id)
            return 1

        bin.current['mask'] = self.update_masks(mask_crop, recs).copy()
        
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
        print(f"Current: {self.current}")
        bin.update_current(self.current)

        # Current mask recommendations on the bin
        mask_crop = self.segment(bin_id)

        plt.imshow(bin.current['rgb'])
        plt.title("Before refinement in update")
        plt.show()

        plt.imshow(mask_crop)
        plt.title("Before refinement in update")
        plt.show()

        # Make sure that the bin only has segmentations for n objects
        mask_crop = self.refine_masks(mask_crop, bin.n)

        if mask_crop is None:
            self.current = self.last
            return GO_TO_UPDATE

        plt.imshow(mask_crop)
        plt.title("After refinement in update")
        plt.show()

        # if mask_crop is None:
        #     print(f"Segmentation could not find objects in bin {bin_id}")
        #     input("Need to reset bin. Take out all items and reput them in.")
        #     self.bad_bins.append(bin_id)
        #     self.reset_bin(bin_id)
        
        assert(np.max(mask_crop) == bin.n)

        # Find the recommended matches between the two frames
        if bin.last['mask'] is not None:
            recs = self.match_masks(bin.last['rgb'],bin.current['rgb'], bin.last['mask'], mask_crop)
            
            if recs is None:
                print(f"SIFT could not confidently match bin {bin_id}")
                figure, axis = plt.subplots(2,)
                axis[0].imshow(bin.last['rgb'])
                axis[1].imshow(bin.current['rgb'])
                plt.title("Frames that sift failed on")
                plt.show()
                self.bad_bins.append(bin_id)
            
            # Update the new frame's masks
            bin.current['mask'] = self.update_masks(mask_crop, recs).copy()

            plt.imshow(bin.current['mask'])
            plt.title("After update")
            plt.show()
        
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

    def reset_bin(self, bin_id):
        bin = self.bins[bin_id]
        bin.n = 0
        bin.current = {'rgb':None, 'depth':None, 'mask':None}
        bin.last = {'rgb':None, 'depth':None, 'mask':None}

        # Remove all objects in the bin from the database
        for loc_obj_id, (loc_bin, loc_id) in self.items.copy().items():
            if loc_bin == bin_id:
                del self.items[loc_obj_id]
        
        # Remove bin from list of bad bins
        self.bad_bins.remove(bin_id)

        return 0

        