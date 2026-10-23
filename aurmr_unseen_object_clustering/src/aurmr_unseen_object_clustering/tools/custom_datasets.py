import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import cv2
import json

PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

data_loading_params = {

# Camera/Frustum parameters
'img_width' : 256, 
'img_height' : 256,
'near' : 0.01,
'far' : 100,
'fov' : 45, # vertical field of view in degrees

'use_data_augmentation' : True,

# Multiplicative noise
'gamma_shape' : 1000.,
'gamma_scale' : 0.001,

# Additive noise
'gaussian_scale' : 0.005, # 5mm standard dev
'gp_rescale_factor' : 4,
'gaussian_scale_range' : [0., 0.003],
'gp_rescale_factor_range' : [12, 20],

# Random ellipse dropout
'ellipse_dropout_mean' : 10, 
'ellipse_gamma_shape' : 5.0, 
'ellipse_gamma_scale' : 1.0,

# Random high gradient dropout
'gradient_dropout_left_mean' : 15, 
'gradient_dropout_alpha' : 2., 
'gradient_dropout_beta' : 5.,

# Random pixel dropout
'pixel_dropout_alpha' : 1., 
'pixel_dropout_beta' : 10.,    
}

class binDataset(Dataset):
    def __init__(self, file_list, for_training=False):
        # Loads the data from the h5 file, where file is the path to the .h5 file
        self.files = []
        self.lens = []
        # Determines whether to apply data randomizations for training
        self.for_training = for_training
        # Load data from each file
        for i in range(len(file_list)):
            self.files.append(h5py.File(file_list[i], 'r'))
            self.lens.append(self.files[i]['frame0_data'].shape[0])

    def __len__(self):
        return sum(self.lens)

    def add_noise_to_depth(self, depth_img, noise_params):
        """ Add noise to depth image. 
            This is adapted from the DexNet 2.0 code.
            Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py

            @param depth_img: a [H x W] set of depth z values
        """
        depth_img = depth_img.copy()

        # Multiplicative noise: Gamma random variable
        multiplicative_noise = np.random.gamma(noise_params['gamma_shape'], noise_params['gamma_scale'])
        depth_img = multiplicative_noise * depth_img

        return depth_img

    def add_noise_to_xyz(self, xyz_img, depth_img, noise_params):
        """ Add (approximate) Gaussian Process noise to ordered point cloud

            @param xyz_img: a [H x W x 3] ordered point cloud
        """
        xyz_img = xyz_img.copy()

        H, W, C = xyz_img.shape

        # Additive noise: Gaussian process, approximated by zero-mean anisotropic Gaussian random variable,
        #                 which is rescaled with bicubic interpolation.
        gp_rescale_factor = np.random.randint(noise_params['gp_rescale_factor_range'][0],
                                            noise_params['gp_rescale_factor_range'][1])
        gp_scale = np.random.uniform(noise_params['gaussian_scale_range'][0],
                                    noise_params['gaussian_scale_range'][1])

        small_H, small_W = (np.array([H, W]) / gp_rescale_factor).astype(int)
        additive_noise = np.random.normal(loc=0.0, scale=gp_scale, size=(small_H, small_W, C))
        additive_noise = cv2.resize(additive_noise, (W, H), interpolation=cv2.INTER_CUBIC)
        xyz_img[depth_img > 0, :] += additive_noise[depth_img > 0, :]

        return xyz_img

    def dropout_random_ellipses(self, depth_img, noise_params):
        """ Randomly drop a few ellipses in the image for robustness.
            This is adapted from the DexNet 2.0 code.
            Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py

            @param depth_img: a [H x W] set of depth z values
        """
        depth_img = depth_img.copy()

        # Sample number of ellipses to dropout
        num_ellipses_to_dropout = np.random.poisson(noise_params['ellipse_dropout_mean'])

        # Sample ellipse centers
        nonzero_pixel_indices = np.array(np.where(depth_img > 0)).T # Shape: [#nonzero_pixels x 2]
        dropout_centers_indices = np.random.choice(nonzero_pixel_indices.shape[0], size=num_ellipses_to_dropout)
        dropout_centers = nonzero_pixel_indices[dropout_centers_indices, :] # Shape: [num_ellipses_to_dropout x 2]

        # Sample ellipse radii and angles
        x_radii = np.random.gamma(noise_params['ellipse_gamma_shape'], noise_params['ellipse_gamma_scale'], size=num_ellipses_to_dropout)
        y_radii = np.random.gamma(noise_params['ellipse_gamma_shape'], noise_params['ellipse_gamma_scale'], size=num_ellipses_to_dropout)
        angles = np.random.randint(0, 360, size=num_ellipses_to_dropout)

        # Dropout ellipses
        for i in range(num_ellipses_to_dropout):
            center = dropout_centers[i, :]
            x_radius = np.round(x_radii[i]).astype(int)
            y_radius = np.round(y_radii[i]).astype(int)
            angle = angles[i]

            # dropout the ellipse
            mask = np.zeros_like(depth_img)
            mask = cv2.ellipse(mask, tuple(center[::-1]), (x_radius, y_radius), angle=angle, startAngle=0, endAngle=360, color=1, thickness=-1)
            depth_img[mask == 1] = 0

        return depth_img

    def random_color_warp(self, image, d_h=None, d_s=None, d_l=None):
        """ Given an RGB image [H x W x 3], add random hue, saturation and luminosity to the image

            Code adapted from: https://github.com/yuxng/PoseCNN/blob/master/lib/utils/blob.py
        """
        H, W, _ = image.shape

        image_color_warped = np.zeros_like(image)

        # Set random hue, luminosity and saturation which ranges from -0.1 to 0.1
        if d_h is None:
            d_h = (np.random.uniform() - 0.5) * 0.2 * 256
        if d_l is None:
            d_l = (np.random.uniform() - 0.5) * 0.2 * 256
        if d_s is None:
            d_s = (np.random.uniform() - 0.5) * 0.2 * 256

        # Convert the RGB to HLS
        hls = cv2.cvtColor(image.round().astype(np.uint8), cv2.COLOR_RGB2HLS)
        h, l, s = cv2.split(hls)

        # Add the values to the image H, L, S
        # new_h = (np.round((h + d_h)) % 256).astype(np.uint8)
        new_h = np.round(np.clip(h + d_h, 0, 255)).astype(np.uint8)
        new_l = np.round(np.clip(l + d_l, 0, 255)).astype(np.uint8)
        new_s = np.round(np.clip(s + d_s, 0, 255)).astype(np.uint8)

        # Convert the HLS to RGB
        new_hls = cv2.merge((new_h, new_l, new_s)).astype(np.uint8)
        new_im = cv2.cvtColor(new_hls, cv2.COLOR_HLS2RGB)

        image_color_warped = new_im.astype(np.float32)

        return image_color_warped

    # Computes point cloud from depth image and camera intrinsics
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
    
    def __getitem__(self, index):
        # For now, we only care about the more packed frame
        cat = 'frame1_'

        # Find the right file to consider and the matching index
        file_now = 0
        while index >= self.lens[file_now]:
            index -= self.lens[file_now]
            file_now += 1
        self.file = self.files[file_now]

        # Loads the color image
        rgb = self.file[cat + 'data'][index][...,0:3]
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Applies color jittering
        if self.for_training:
            rgb = self.random_color_warp(rgb)

        # Loads the depth image and camera intrinsics
        depth = self.file[cat + 'depth'][index]

        # Finds all valid pixels for standardization
        mask = depth >= -1000
        depth[depth < -1000] = 0
        intrinsic = json.loads(self.file[cat + 'metadata'][index])['camera']['intrinsic_matrix']

        # Adds noise and cuts out random ellipses from depth
        if self.for_training:
            depth = self.add_noise_to_depth(self.dropout_random_ellipses(depth, data_loading_params), data_loading_params)

        # Computes the point cloud
        xyz = self.compute_xyz(depth, intrinsic).astype(np.float32)

        # Adds noise to the point cloud
        if self.for_training:
            xyz = self.add_noise_to_xyz(xyz, depth, data_loading_params)

        # Loads the label masks
        label = self.file[cat + 'mask'][index]
        label = (label + 1) % 65536
        label %= np.max(label)

        # Randomly flip the image/point cloud/label
        if self.for_training:
            if np.random.uniform() < .5:
                rgb = np.fliplr(rgb).copy()
                depth = np.fliplr(depth).copy()
                xyz = np.fliplr(xyz).copy()
                label = np.fliplr(label).copy()
        
        # Grab un-normalized images for refinement network
        rgb_n = rgb.copy()
        xyz_n = xyz.copy()

        # Standardize image and point cloud
        rgb[..., 0] = (rgb[..., 0] - np.mean(rgb[..., 0], where=mask)) / np.std(rgb[..., 0], where=mask) * mask
        rgb[..., 1] = (rgb[..., 1] - np.mean(rgb[..., 1], where=mask)) / np.std(rgb[..., 1], where=mask) * mask
        rgb[..., 2] = (rgb[..., 2] - np.mean(rgb[..., 2], where=mask)) / np.std(rgb[..., 2], where=mask) * mask
        xyz[..., 0] = (xyz[..., 0] - np.mean(xyz[..., 0], where=mask)) / np.std(xyz[..., 0], where=mask) * mask
        xyz[..., 1] = (xyz[..., 1] - np.mean(xyz[..., 1], where=mask)) / np.std(xyz[..., 1], where=mask) * mask
        xyz[..., 2] = (xyz[..., 2] - np.mean(xyz[..., 2], where=mask)) / np.std(xyz[..., 2], where=mask) * mask

        # Turn everything into tensors and reshape as needed
        im_tensor = torch.from_numpy(rgb) / 255.0
        im_tensor_n = torch.from_numpy(rgb_n) / 255.0
        pixel_mean = torch.tensor(PIXEL_MEANS / 255.0).float()
        im_tensor_n -= pixel_mean
        image_blob = im_tensor.permute(2, 0, 1)
        image_blob_n = im_tensor_n.permute(2, 0, 1)
        if self.for_training:
            sample = {'image_color': image_blob}
            sample['image_color_n'] = image_blob_n
        else:
            sample = {'image_color': image_blob.unsqueeze(0)}
            sample['image_color_n'] = image_blob_n.unsqueeze(0)

        depth_blob = torch.from_numpy(xyz).permute(2, 0, 1)
        depth_blob_n = torch.from_numpy(xyz_n).permute(2, 0, 1)
        if self.for_training:
            sample['depth'] = depth_blob
            sample['depth_n'] = depth_blob_n
        else:
            sample['depth'] = depth_blob.unsqueeze(0)
            sample['depth_n'] = depth_blob_n.unsqueeze(0)
            
        label_blob = torch.from_numpy(label)
        if self.for_training:
            sample['label'] = label_blob
        else:
            sample['label'] = label_blob.unsqueeze(0)

        return sample

class npDataset(Dataset):
    def __init__(self, dict, bounds=None, transforms=None):
        # Loads the data from the h5 file, where file is the path to the .h5 file
        self.dict = dict
        self.name = 'Numpy dataset'
        self.num_classes = 2
        self.bounds = bounds
        self.transforms = transforms
        self.lens = len(dict['frame0_rgb'])
        self.has_labels = 'frame0_label' in self.dict.keys()
    
    def __len__(self):
        return self.lens

    # Computes point cloud from depth image and camera intrinsics
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
    
    def __getitem__(self, index):
        cat = 'frame0_'       

        # Loads the color image
        rgb = self.dict[cat + 'rgb'][index][...,0:3]
        if 'swap_color' in self.transforms:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Loads the depth image and camera intrinsics
        depth = self.dict[cat + 'depth'][index]
        if 'fill_holes' in self.transforms:
            depth[depth == 0] = depth[20 + 810, 20 + 1080]

        # Computes the point cloud
        intrinsic = np.array(self.dict[f'{cat}info'][index]['K']).reshape(3,3)
        xyz = self.compute_xyz(depth, intrinsic).astype(np.float32)
        if 'scale_xyz' in self.transforms:
            xyz /= 1000.

        # Loads the label masks
        if self.has_labels:
            label = self.dict[cat + 'label'][index]

        # Crops the image
        if self.bounds is not None:
            xyz = xyz[self.bounds[0]:self.bounds[1], self.bounds[2]:self.bounds[3], :]
            rgb = rgb[self.bounds[0]:self.bounds[1], self.bounds[2]:self.bounds[3], :]
            if self.has_labels:
                label = label[self.bounds[0]:self.bounds[1], self.bounds[2]:self.bounds[3]]
        
        if 'rotate' in self.transforms:
            rgb = np.rot90(rgb, k=3).copy()
            xyz = np.rot90(xyz, k=3).copy()
            if self.has_labels:
                label = np.rot90(label, k=3).copy()
        
        rgb_n = rgb.copy()
        xyz_n = xyz.copy()

        # Standardize image and point cloud
        rgb[..., 0] = (rgb[..., 0] - np.mean(rgb[...,0])) / np.std(rgb[...,0])
        rgb[..., 1] = (rgb[..., 1] - np.mean(rgb[...,1])) / np.std(rgb[...,1])
        rgb[..., 2] = (rgb[..., 2] - np.mean(rgb[...,2])) / np.std(rgb[...,2])
        xyz[..., 0] = (xyz[..., 0] - np.mean(xyz[...,0])) / np.std(xyz[...,0])
        xyz[..., 1] = (xyz[..., 1] - np.mean(xyz[...,1])) / np.std(xyz[...,1])
        xyz[..., 2] = (xyz[..., 2] - np.mean(xyz[...,2])) / np.std(xyz[...,2])

        # Turn everything into tensors and reshape as needed
        im_tensor = torch.from_numpy(rgb) / 255.0
        im_tensor_n = torch.from_numpy(rgb_n) / 255.0
        pixel_mean = torch.tensor(PIXEL_MEANS / 255.0).float()
        im_tensor_n -= pixel_mean
        image_blob = im_tensor.permute(2, 0, 1)
        image_blob_n = im_tensor_n.permute(2, 0, 1)
        sample = {'image_color': image_blob.unsqueeze(0)}
        sample['image_color_n'] = image_blob_n.unsqueeze(0)

        depth_blob = torch.from_numpy(xyz).permute(2, 0, 1)
        depth_blob_n = torch.from_numpy(xyz_n).permute(2, 0, 1)
        sample['depth'] = depth_blob.unsqueeze(0)
        sample['depth_n'] = depth_blob_n.unsqueeze(0)

        if self.has_labels:
            label_blob = torch.from_numpy(label)
            sample['label'] = label_blob.unsqueeze(0)

        return sample