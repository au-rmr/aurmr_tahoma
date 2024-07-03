import torch
import torch.nn as nn
import torch.nn.functional as F
import os


from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image
import numpy as np
import cv2
import re
import random
from torch.utils.data.dataset import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
import torch.nn as nn
import torch.nn.functional as F

class CropTransform:
    def __init__(self, crop_dim):
        self.crop_dim = crop_dim

    def __call__(self, x):
        return x[
            self.crop_dim[0] : self.crop_dim[1], self.crop_dim[2] : self.crop_dim[3]
        ]


class CustomSingleDataset(Dataset):
    def __init__(self, image, depth, segmask):
        self.image = image
        self.segmask = segmask
        self.depth_image = depth
        cv2.imwrite("/tmp/rgb_grasp.png", image)
        cv2.imwrite("/tmp/segmask_grasp.png", segmask)

        self.resize_transform = transforms.Resize(
            (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
        )
        self.normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.batch_size = 1000
        self.patch_size = 32

    def _get_patch_indices(self, segmask):
        numpy_mask = segmask
        mask_id = np.unique(numpy_mask)[1]
        padding_size = [
            self.patch_size,
            self.patch_size,
            self.patch_size,
            self.patch_size,
        ]
        mask = numpy_mask == int(mask_id)
        mask = torch.from_numpy(mask).unsqueeze(0)
        mask = self.resize_transform(mask)
        mask = F.pad(mask, padding_size, "constant", 0).squeeze(0)

        non_zero_coords = torch.nonzero(mask, as_tuple=False)
        non_zero_coords = [tuple(coord.tolist()) for coord in non_zero_coords]
        non_zero_coords = random.sample(
            non_zero_coords, min(len(non_zero_coords), self.batch_size)
        )

        return non_zero_coords

    def _extract_patch(self, image, path_idx, pad_size):
        y, x = path_idx
        patch = image[
            y - self.patch_size // 2 : y + self.patch_size // 2,
            x - self.patch_size // 2 : x + self.patch_size // 2,
        ]
        return patch

    def __getitem__(self, idx):
        patch_indices = self._get_patch_indices(self.segmask)

        batch_patches = []
        padding_size = [
            self.patch_size,
            self.patch_size,
            self.patch_size,
            self.patch_size,
        ]

        for patch_idx in patch_indices:
            depth_image = self.depth_image
            depth_image = torch.from_numpy(depth_image).unsqueeze(0).div(1000).to(torch.float32)
            depth_image = self.resize_transform(depth_image)
            depth_image = F.pad(depth_image, padding_size, "constant", 0).squeeze(0)

            patch = self._extract_patch(
                depth_image, patch_idx, padding_size[0]
            )
            batch_patches.append(patch)

        batch_patches = torch.stack(batch_patches)

        return batch_patches, patch_indices, self.depth_image

    def __len__(self):
        return 1


class SimpleGQCNN(nn.Module):
    def __init__(self):
        super(SimpleGQCNN, self).__init__()
        # Image stream
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        # Calculate the output size after convolutions and pooling
        self.fc3_input_size = self._get_conv_output_size((1, 32, 32))
        self.fc3 = nn.Linear(self.fc3_input_size, 1024)

        # Merge stream
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 1)  # Output: single value (e.g., quality score)

    def _get_conv_output_size(self, shape):
        # Generate a dummy tensor to get the output size after conv layers
        dummy_input = torch.zeros(1, *shape)
        with torch.no_grad():
            x = F.relu(self.conv1_1(dummy_input))
            x = F.max_pool2d(x, kernel_size=1, stride=1)
            if True:
                x = F.local_response_norm(x, size=2)

            x = F.relu(self.conv1_2(x))
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            if True:
                x = F.local_response_norm(x, size=2)

            x = F.relu(self.conv2_1(x))
            x = F.max_pool2d(x, kernel_size=1, stride=1)

            x = F.relu(self.conv2_2(x))
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            if True:
                x = F.local_response_norm(x, size=2)

        return x.numel()

    def forward(self, x_image):
        x = F.relu(self.conv1_1(x_image))
        x = F.max_pool2d(x, kernel_size=1, stride=1)
        if True:
            x = F.local_response_norm(x, size=2)

        x = F.relu(self.conv1_2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        if True:
            x = F.local_response_norm(x, size=2)

        x = F.relu(self.conv2_1(x))
        x = F.max_pool2d(x, kernel_size=1, stride=1)

        x = F.relu(self.conv2_2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        if True:
            x = F.local_response_norm(x, size=2)

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc3(x))

        x = F.relu(self.fc4(x))
        x = self.fc5(
            x
        )  # Output layer without activation function for regression output

        return x



class TrainDexNetModel:
    def __init__(self, image, depth, mask):
        self.image = image
        self.mask = mask
        self.depth = depth

    def flatten_state_dict(self, state_dict):
        flat_state_dict = {}
        for key, value in state_dict.items():
            if 'state_dict' in key:
                # Remove redundant 'state_dict' from keys
                new_key = key.split('state_dict.')[-1]
                flat_state_dict[new_key] = value
            else:
                flat_state_dict[key] = value
        return flat_state_dict

    def init_model(self):
        model = SimpleGQCNN()
        model.load_state_dict(torch.load("/home/aurmr/workspaces/RGB_grasp_soofiyan_ws/src/NBV-GP/huggingface/Depth-Anything/weights/checkpoint_dexnet_hardware_state_dict.pth"))

        model = model.to("cuda")
        return model


    def save(self, model):

        from torch.nn.parallel import DistributedDataParallel as DDP
        if isinstance(model, DDP):
            model = model.module
        torch.save(model.state_dict(), "/tmp/foo.pth")

    def run_dexnet_hardware_comm_inference(self):
        dataset = CustomSingleDataset(self.image, self.depth, self.mask)

        inference_dataloader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0
        )
        model = self.init_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():  # Disable gradient computation
            for images, patch_indices, depth_image in inference_dataloader:
                images = images.to(device)

                images = images.squeeze(0).unsqueeze(1)
                score = model(images)

                indices_ = torch.argmax(score.squeeze(1))
                max_indices = patch_indices[indices_]

                indices_ = torch.max(score.squeeze(1))
        return max_indices
