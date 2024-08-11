import torch
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

import torch.nn.functional as F
from torchvision.transforms import Compose

import cv2
import numpy as np

from functools import lru_cache


@lru_cache()
def load_depth_anything(device):
    depth_anything = DepthAnything.from_pretrained("LiheYoung/depth_anything_vitl14")
    return depth_anything.to(device).eval()

def estimate_depth(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    depth_anything = load_depth_anything(DEVICE)


    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])


    h, w = image.shape[:2]

    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        depth = depth_anything(image)

    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

    depth = depth.cpu().numpy().astype(np.uint8)

    return depth

