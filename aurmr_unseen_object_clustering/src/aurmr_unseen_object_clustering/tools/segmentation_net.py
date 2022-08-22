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
MIN_MATCH_COUNT = 1


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
   'camera':'azure',
   'bounds':bin_bounds,
   'model_init':UOC_PATH + 'data/checkpoints/std_200_24_checkpoint.pth',
#    'model_ref':UOC_PATH + 'data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_186.checkpoint_crop.pth',
#    'model_ref':UOC_PATH + 'data/checkpoints/crop_net_current.pth',
#    'model_ref':UOC_PATH + 'data/checkpoints/seg_resnet34_8s_embedding_cosine_color_crop_sampling_epoch_16.checkpoint_justcolor.pth',
   'model_ref': UOC_PATH + 'data/checkpoints/std_200_color_crop.pth',
#    'model_ref':None,
   'gpu_id':0,
   'cfg_init':UOC_PATH + 'experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml',
   # 'cfg_ref':UOC_PATH + 'experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml',
   'cfg_ref':UOC_PATH + 'experiments/cfgs/seg_resnet34_8s_embedding_cosine_color_tabletop.yml',
   'dataset_name':'shapenet_scene_train',
   'depth_name':'*depth.png',
   'color_name':'*color.png',
   'imgdir':None,
   'randomize':False,
   'network_name':'seg_resnet34_8s_embedding',
   'image_path':None,
 
    # Erosion/Dilation and largest connected component
   'refine_masks':False,
   'refine_refined_masks':True,
   # 'rm_back':True,
   'rm_back':True,
   'rm_back_old':False,
   'kernel_size':5,
   'erosion_num':3,
   'dilation_num':4,
   'rgb_is_standardized':True,
   'xyz_is_standardized':True,
   'sd_loc':(50, 50),
   'min_bg_change':15,
   'resize':True,
   'print_preds':False
}

# If running the network on synthetic data
# syn_config = {
#    'camera':'azure',
#    'bounds':bin_bounds,
#    'model_init':UOC_PATH + 'data/checkpoints/std_200_24_checkpoint.pth',
# #    'model_ref':UOC_PATH + 'seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_186.checkpoint_crop.pth',
# #    'model_ref':UOC_PATH + 'data/checkpoints/crop_net_current.pth',
# #    'model_ref':UOC_PATH + 'data/checkpoints/seg_resnet34_8s_embedding_cosine_color_crop_sampling_epoch_16.checkpoint_justcolor.pth',
#    'model_ref':None,
#    'gpu_id':0,
#    # 'cfg_file':UOC_PATH + 'experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml',
#    'cfg_file':UOC_PATH + 'experiments/cfgs/seg_resnet34_8s_embedding_cosine_color_tabletop.yml',
#    'dataset_name':'shapenet_scene_train',
#    'depth_name':'*depth.png',
#    'color_name':'*color.png',
#    'imgdir':None,
#    'randomize':False,
#    'network_name':'seg_resnet34_8s_embedding',
#    'image_path':None,
#
#    'refine_masks':True,
#    'refine_refined_masks':True,
#    'rm_back':True,
#    'rm_back_old':False,
#    'kernel_size':5,
#    'erosion_num':3,
#    'dilation_num':4,
#    'rgb_is_standardized':True,
#    'xyz_is_standardized':True,
#    'sd_loc':(50, 50),
#    'min_bg_change':15,
#    'resize':True
# }
 
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
        self.current = {'rgb':None, 'depth':None, 'mask':None}
        self.last = {'rgb':None, 'depth':None, 'mask':None}
        self.init_depth = init_depth[bounds[0]:bounds[1], bounds[2]:bounds[3]]
        self.bg_mask = np.ones(self.init_depth.shape, dtype=np.uint8)
  
    def update_current(self, current):
        self.last = self.current.copy()
  
        rgb = current['rgb']
        depth = current['depth']
        self.current['rgb'] = rgb[self.bounds[0]:self.bounds[1], self.bounds[2]:self.bounds[3], ...]
        self.current['depth'] = depth[self.bounds[0]:self.bounds[1], self.bounds[2]:self.bounds[3], ...]
        self.bg_mask = np.abs(self.current['depth'][...,2] - self.init_depth) < self.config['min_bg_change']
        # plt.imshow(self.bg_mask)
        # plt.title("Background Mask")
        # plt.show()
        self.current['mask'] = None


class SegNet:
    def __init__(self, config=def_config, init_depth=None, mask=None):
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
        for bin in bin_names:
            self.bins[bin] = Bin(bin, bounds=config['bounds'][bin], init_depth=init_depth)
  
        self.H, self.W = None, None
  
        # Contains mappings from pruduct bin ID to [bin, intra_bin_ID]
        self.items = {}
  
        self.n = 0

        self.init_depth = init_depth
        # Stores scene state
        self.current = {'rgb':None, 'depth':None}
        self.last = {'rgb':None, 'depth':None}
  
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
            rgb = self.current['rgb']
            xyz = self.current['depth']
        else:
            rgb = scene['rgb']
            xyz = scene['depth']
  
        # Shift to BGR if using the realsense camera
        if self.config['camera'] == 'realsense':
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

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
        if self.config['refine_masks']:
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

                if self.config['refine_refined_masks']:
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
  
        # Place masks in original context
        # mask_full = np.zeros(shape=(H,W), dtype=np.uint8)
        if self.config['camera'] == 'azure':
            mask_crop = out_label[0].cpu().numpy().astype(np.uint8)
            if self.network_crop is not None:
                mask_crop = out_label_ref[0].cpu().numpy().astype(np.uint8)

            # plt.imshow(mask_crop)
            # plt.title("Mask Crop")
            # plt.show()

            # plt.imshow(mask_crop_ref)
            # plt.title("Mask Crop Ref")
            # plt.show()
        else:
            mask_crop = out_label[0].cpu().numpy()
            mask_crop = np.rot90(mask_crop, k=1)
  
        mask_crop = cv2.resize(mask_crop, (bin.bounds[3] - bin.bounds[2], bin.bounds[1] - bin.bounds[0]), interpolation=cv2.INTER_AREA)
        
        # mask_full[self.bounds[0]:self.bounds[1], self.bounds[2]:self.bounds[3]] = mask_crop
  
        return mask_crop
  
    # Returns a best-guess matching between masks in frame 0 and masks in frame 1
    #       Returns a list, where the j - 1th element stores the recommended frame 1 ID of the mask with frame 0 ID j
    def match_masks(self, im1, im2, mask1, mask2):
        print("MATCHING_MASKS")
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

            matched_img = cv2.drawMatches(im1_now, k1, im2, k2, good, im2, flags=2, matchesThickness=1)

            # cv2.imshow('image', matched_img)
            # cv2.imwrite(f'matched_img_{self.const}{i}.png', matched_img)
           
            # If there's any meaningful match
            if len(good) > MIN_MATCH_COUNT:
                dst_pts = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.round(dst_pts).astype(int)
            else:
                dst_pts = None
  
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
  
        return col_ind + 1
  
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
            # Calculate areas for each mask
            areas = np.array([np.sum(mask == i) for i in range(1, np.max(mask) + 1)])
            idx_min = np.argmin(areas) + 1
            # Update masks
            mask[mask == idx_min] = 0
            mask[mask > idx_min] -= 1
        while np.max(mask) < n:
            print(f"We want {n} masks but there are only {np.max(mask)}")
            # Calculate areas for each mask
            areas = np.array([np.sum(mask == i) for i in range(1, np.max(mask) + 1)])
            idx_max = np.argmax(areas) + 1
            # Cut the largest mask in half lengthwise
  
            # Finds the center of the largest object
            line = np.sum((mask == idx_max), axis=1)
            idx = 0
            sum = 0
            total = areas[idx_max - 1]
            while sum < total / 2:
                sum += line[idx]
                idx += 1
           
            # Split the largest area along idx into a new mask
            mask[idx:][mask[idx:] == idx_max] = np.max(mask) + 1
  
        return mask
  
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
        mask_full[r1:r2, c1:c2] = (mask_crop == id).astype(np.uint8)
        return mask_full
  
    def stow(self, bin_id, obj_id, rgb_raw, depth_raw, intrinsic):
        # Grabs the current bin
        bin = self.bins[bin_id]


        if self.H is None:
            self.H, self.W, _ = rgb_raw.shape
  
        # Update the current state of the bin and scene with the new images
        self.last = self.current.copy()
        self.current['rgb'] = rgb_raw.astype(np.uint8)
        self.current['depth'] = self.compute_xyz(depth_raw, intrinsic)
        self.current['mask'] = None
        bin.update_current(self.current)
        # plt.imshow(self.current['rgb'] )
        # plt.show()
  
        # Current mask recommendations on the bin
        mask_crop = self.segment(bin_id)
        # Keep track of the total number of objects in the bin
        bin.n += 1
        self.n += 1
  
        # Make sure that the bin only has segmentations for n objects
        mask_crop = self.refine_masks(mask_crop, bin.n)
        assert(np.max(mask_crop) == bin.n)
  
        # Find the recommended matches between the two frames
        if bin.last['mask'] is not None:
            recs = self.match_masks(bin.last['rgb'],bin.current['rgb'], bin.last['mask'], mask_crop)
  
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
  
            bin.current['mask'] = mask_crop.copy()
  
        self.items[obj_id] = [bin_id, bin.n]
       
    def pick(self, obj_id, rgb_raw, depth_raw, intrinsic):
        # Finds the bin and local ID for the object
        bin_id, obj_n = self.items[obj_id]
        # Grabs the current bin
        bin = self.bins[bin_id]
  
        # Update the current state of the bin and scene with the new images
        self.last = self.current.copy()
        self.current['rgb'] = rgb_raw.astype(np.uint8)
        self.current['depth'] = self.compute_xyz(depth_raw, intrinsic)
        self.current['mask'] = None
        bin.update_current(self.current)
  
        # rgb_now = bin.current['rgb']
  
        # Current mask recommendations on the bin
        mask_crop = self.segment(bin_id)

        
  
        # maskx3 = np.repeat(np.reshape(mask_crop, (mask_crop.shape[0], mask_crop.shape[1], 1)), 3, axis=2)
        # toshow = .3 * maskx3 + .7 * rgb_now

        # Keep track of the total number of objects in the bin
        bin.n -= 1
        self.n -= 1
  
        # Make sure that the bin only has segmentations for n objects
        mask_crop = self.refine_masks(mask_crop, bin.n)
  
        assert(np.max(mask_crop) == bin.n)

        if self.config['print_preds']:
            plt.imshow(mask_crop)
            plt.title("Cropped mask prediction (No refinement)")
            plt.show()
  
        # You should only pick if there's an object in the bin
        assert(bin.last['mask'] is not None)
  
        old_mask = bin.last['mask'].copy()
        # Remove the object that is no longer in the scene
        old_mask[old_mask == obj_n] = 0
        old_mask[old_mask > obj_n] -= 1
        recs = self.match_masks(bin.last['rgb'], bin.current['rgb'], old_mask, mask_crop)
        bin.current['mask'] = self.update_masks(mask_crop, recs).copy()
  
  
        for loc_obj_id, (loc_bin, loc_id) in self.items.items():
            if loc_bin == bin_id:
                if loc_id > obj_n:
                    self.items[loc_obj_id] = [loc_bin, loc_id - 1]
  
        del self.items[obj_id]
  
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