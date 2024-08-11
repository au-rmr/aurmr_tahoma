#!/usr/bin/env python3

import os

import numpy as np
import torch
import cv2

import matplotlib.pyplot as plt

import rich_click as click
from rich.progress import track

from aurmr_setup.utils.workspace_utils import get_active_workspace_path

from click_prompt import filepath_option
from depth_utils import estimate_depth
from gqcnn_hardware_comms import TrainDexNetModel, SimpleGQCNN


def get_default_dataset_path():
    workpace_path = get_active_workspace_path()
    return  f"{workpace_path}/src/Paper evaluation/Dataset/"


@click.command()
@filepath_option("--dataset-folder", default=get_default_dataset_path())
def cli(dataset_folder):

    for filename in track(os.listdir(dataset_folder)):
        if not filename.startswith("rgb_img_"):
            continue

        image_suffix = filename[8:]
        if not os.path.isfile(os.path.join(dataset_folder, "depth_img_" + image_suffix)):
            print("Depth image does not exist.")
            continue
        dexnet_with_depth_estimation(dataset_folder, filename)
 



def query_dexnet(rgb_image, depth, mask):

    object_hardware_comm = TrainDexNetModel(rgb_image, depth, mask)
    grasp_points = object_hardware_comm.run_dexnet_hardware_comm_inference()

    grasp_points[:, 0] = grasp_points[:, 0] * rgb_image.shape[0]
    grasp_points[:, 1] = grasp_points[:, 1] * rgb_image.shape[1]

    return grasp_points


def dexnet_with_depth_estimation(folder, image_name):

    image_suffix = image_name[8:]

    rgb_image = cv2.imread(os.path.join(folder, image_name)) 
    mask = cv2.imread(os.path.join(folder, "segmask_img_" + image_suffix), 0)
    depth = estimate_depth(rgb_image)
    depth = depth.astype(np.float32)

    original_depth = cv2.imread(os.path.join(folder, "depth_img_" + image_suffix), cv2.IMREAD_UNCHANGED)
    original_depth = original_depth.astype(np.float32)

    min_bin_distance = 1100
    max_bin_distance = 1370

    depth = (1.0 - (depth / 255)) * (max_bin_distance - min_bin_distance) + min_bin_distance


    grasp_points = query_dexnet(rgb_image, depth, mask)
    original_grasp_points = query_dexnet(rgb_image, original_depth, mask)


    depth_colorized = cv2.applyColorMap(depth.astype(np.uint8), cv2.COLORMAP_INFERNO)
    original_depth_colorized = cv2.applyColorMap(original_depth.astype(np.uint8), cv2.COLORMAP_INFERNO)

    fig, axs = plt.subplots(1, 4, figsize=(12, 3))
    
    for ax in axs:
        ax.axis("off")


    NUM_GRASPS = 100

    masked_rgb_image = rgb_image[:, :, ::-1] 
    masked_rgb_image[:, :, 0] = masked_rgb_image[:, :, 0] * 0.7 * mask + masked_rgb_image[:, :, 0] * 0.3 
    masked_rgb_image[:, :, 1] = masked_rgb_image[:, :, 1] * 0.7 * mask + masked_rgb_image[:, :, 1] * 0.3
    masked_rgb_image[:, :, 2] = masked_rgb_image[:, :, 2] * 0.7 * mask + masked_rgb_image[:, :, 2] * 0.3
    axs[0].imshow(masked_rgb_image)


    for grasp_point in grasp_points[:NUM_GRASPS]:
        axs[0].add_patch(plt.Circle((grasp_point[1], grasp_point[0]), radius=2, color="orange", fill=False, alpha=0.5))
    for grasp_point in original_grasp_points[:NUM_GRASPS]:
        axs[0].add_patch(plt.Circle((grasp_point[1], grasp_point[0]), radius=2, color="teal", fill=False, alpha=0.5))
    for grasp_point in grasp_points[:1]:
        axs[0].add_patch(plt.Circle((grasp_point[1], grasp_point[0]), radius=2, color="red", fill=False, alpha=1.0))
    for grasp_point in original_grasp_points[:1]:
        axs[0].add_patch(plt.Circle((grasp_point[1], grasp_point[0]), radius=2, color="blue", fill=False, alpha=1.0))
    axs[1].imshow(mask * 128, cmap="gray")
    axs[2].imshow(depth_colorized)
    axs[3].imshow(original_depth_colorized)
    plt.tight_layout()
    fig.savefig(f"/tmp/grasp_point_{image_suffix}")
    #plt.show()

if __name__ == "__main__":
    import torch
    import numpy as np
    torch.random.seed()
    np.random.seed()
    cli()
