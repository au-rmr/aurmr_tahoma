import os
import cv2
import argparse
import random

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from typing import Tuple, List

join = os.path.join


def mouse_click_event(event, x, y, flags, param):
    global top_left_click_x, top_left_click_y, bottom_right_click_x, bottom_right_click_y
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Left button clicked at ({x}, {y})")
        top_left_click_x = x
        top_left_click_y = y
    elif event == cv2.EVENT_MBUTTONDOWN:
        print(f"Middle button clicked at ({x}, {y})")
        bottom_right_click_x = x
        bottom_right_click_y = y


def load_sample(args, image_id: int, mask_id: int) -> List[np.ndarray]:
    # Load the image
    rgb_image_path = join(args.input, f"{image_id}_rgb_image.png")  # "src/segnetv2_mask2_former/Mask_Results/1699238205rgb_first.png"
    image = cv2.imread(rgb_image_path)

    # Load the image
    depth_image_path = join(args.input, f"{image_id}_depth_image.png")  # "src/segnetv2_mask2_former/Mask_Results/1699238205depth.png"
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    if args.mouse_click:
        depth_image = depth_image.astype(np.float32)/1000.0
    else:
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
        depth_image = depth_image.astype(np.float32) / 255

    # Load the mask
    segmask_path = join(args.input, f"{image_id}_mask_{mask_id}.png")  # "src/segnetv2_mask2_former/Mask_Results/1699238205mask_first.png"
    mask = cv2.imread(segmask_path, 0)
    
    # cv2.imshow("Image Window", image)
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     cv2.destroyWindow("My Window")
    #     exit()

    return [image, depth_image, mask]


def parse_user_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # String/Path arguments
    parser.add_argument('-i', '--input', help='Path to input dataset directory', default='/home/aurmr/Documents/CVPR_Nov2023/Benchmark_Data/Processed_Data')
    parser.add_argument('-o', '--output', help='Path to output dataset directory', default='/home/aurmr/Documents/CVPR_Nov2023/Benchmark_Data/GPT_Output')
    parser.add_argument('--points', help='GPT inference output', default='1, 2, 3')
    # Integer/Quantity arguments
    parser.add_argument('--x_step', default=20, type=int, help='Step size along width in pixels for annotation')
    parser.add_argument('--y_step', default=20, type=int, help='Step size along height in pixels for annotation')
    # Boolean/Action arguments
    parser.add_argument('--mouse_click', action='store_true', help='True: expects image bounds from mouse click to crop; False: receives a cropped image')
    parser.add_argument('--single', action='store_true', help='True: a single image with numbers of grasping points; False: a set of 6 images with grasping point circle')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_user_args()
    
    # image_id, mask_id
    file_ids = [[int(f.split('.')[0].split('_')[0]), int(f.split('.')[0].split('_')[-1])] for f in os.listdir(args.input) if f.split('.')[0].split('_')[1] == 'mask'] 
    image_ids = [ids[0] for ids in file_ids]
    mask_ids = [ids[1] for ids in file_ids]

    for image_id, mask_id in zip(image_ids, mask_ids):
        # cv2.namedWindow("Image Window")
        # cv2.setMouseCallback("Image Window", mouse_click_event)

        image, depth_image, mask = load_sample(args, image_id, mask_id)

        if args.mouse_click:
            image_crop = image[top_left_click_y:bottom_right_click_y, top_left_click_x:bottom_right_click_x]
            mask_crop = mask[top_left_click_y:bottom_right_click_y, top_left_click_x:bottom_right_click_x]
            depth_image_crop = depth_image[top_left_click_y:bottom_right_click_y, top_left_click_x:bottom_right_click_x]

            target_id = 1
            segmask = np.zeros_like(mask_crop)
            segmask[mask_crop == target_id] = 1
            shrink_size = 3
            kernel = np.ones((shrink_size*2 + 1, shrink_size*2 + 1), np.uint8)
            segmask = cv2.erode(segmask, kernel, iterations = 1)

            new_rgb_image = cv2.resize(image_crop, (image_crop.shape[1]*2, image_crop.shape[0]*2))
        else:
            image_crop = image
            depth_image_crop = depth_image
            segmask = mask

        # cv2.imshow("Image Window", image_crop)
        # cv2.waitKey(2000)
        image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        # cv2.imshow("Image Window", image_crop)
        # cv2.waitKey(2000)

        target_coords = np.column_stack(np.where(segmask != 0))

        y_min, x_min = target_coords.min(axis=0)
        y_max, x_max = target_coords.max(axis=0)

        z = depth_image_crop[int((y_max-y_min)/2), int((x_max-x_min)/2)] - 0.1
        # print(f"{z=}")
        cx = depth_image_crop.shape[1] / 2
        cy = depth_image_crop.shape[0] / 2
        fx = 903.14697265625
        fy = 903.5240478515625
        # height, width = depth_image_crop.shape

        # Compute the object's actual sizes (height, width)
        top_left_3D_x = (x_min - cx) / fx
        top_left_3D_y = (y_min - cy) / fy
        x1 = top_left_3D_x * z
        y1 = top_left_3D_y * z

        bottom_right_3D_x = (x_max - cx) / fx
        bottom_right_3D_y = (y_max - cy) / fy
        x2 = bottom_right_3D_x * z
        y2 = bottom_right_3D_y * z

        width = y2-y1
        height = x2-x1 + 0.035
        # width = 0.075
        # height = 0.220
        print(f"width: {width}")
        print(f"height: {height}")

        # image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)

        # depth_im_pil = Image.fromarray(depth_image_crop)
        new_width, new_height = image_crop.shape[1]*6, image_crop.shape[0]*6
        image = cv2.resize(image_crop, (new_width, new_height))
        num_points_per_side_x = int((x_max - x_min)/args.x_step)
        num_points_per_side_y = int((y_max - y_min)/args.y_step) 

        x_min_grid = 0
        y_min_grid = 0
        x_max_grid = (x_max - x_min)
        y_max_grid = (y_max - y_min)
        x_spacing = (x_max - x_min) // num_points_per_side_x
        y_spacing = (y_max - y_min) // num_points_per_side_y

        # Generate grid points
        grid_points = [(y, x) for x in range(x_min + x_spacing // 2, x_max, x_spacing) 
                            for y in range(y_min + y_spacing // 2, y_max, y_spacing)]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        font_thickness = 4
        circle_radius = 40
        circle_thickness = -1
        color = (255, 255, 255)
        concatenate_count = 0

        points = np.array(args.points.split(' '))
        # points = np.array([7, 8, 10, 11, 12])
        imgs = []

        for i, point in enumerate(grid_points, 1):
            if(segmask[point[0], point[1]] == 0):
                continue  # Do not include numbers outside of the object contour

            image_viz = image.copy()
            if(not args.single):
                if(i in points):
                    image_viz = cv2.circle(image_viz, (point[1]*6, point[0]*6), 200, (255, 255, 255), 20)
                    final_image = cv2.resize(image_viz, (image.shape[1]//3, image.shape[0]//3))
                    concatenate_count += 1

                    imgs.append(Image.fromarray(np.uint8(final_image)))
            else:
                image = cv2.putText(image, str(i), (int(point[1]*6-circle_radius/1.5), int(point[0]*6+circle_radius/2)), 
                                    font, font_scale, color, font_thickness, cv2.LINE_AA)
                final_image = cv2.resize(image, (image.shape[1]//3, image.shape[0]//3))
                final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)

        # cv2.imshow("Image Window", final_image)
        # cv2.waitKey(2000)

        if(not args.single):
            min_img_shape = sorted([(min(img.width, img.height), img.width, img.height) for img in imgs])[0]
            imgs_comb = [img.resize((min_img_shape[1], min_img_shape[2])) for img in imgs]
            new_im = Image.new('RGB', (3 * min_img_shape[1], 2 * min_img_shape[2]))
            for i in range(2):
                for j in range(3):
                    new_im.paste(imgs_comb[i*3 + j], (min_img_shape[1] * j, min_img_shape[2] * i))

            # new_im.paste(imgs_comb[0], (0, 0))
            # new_im.paste(imgs_comb[1], (min_img_shape[1], 0))
            # new_im.paste(imgs_comb[2], (min_img_shape[1]*2, 0))
            # new_im.paste(imgs_comb[3], (0, min_img_shape[2]))
            # new_im.paste(imgs_comb[4], (min_img_shape[1], min_img_shape[2]))
            # new_im.paste(imgs_comb[5], (min_img_shape[1]*2, min_img_shape[2]))

            image_path = join(args.output, f"{image_id}_rgb_image_different_set_{mask_id}.png")  # 'src/segnetv2_mask2_former/VLM_Results/rgb_image_concatenate.png'
            new_im.save(image_path)
        else:
            # Save annotated (text) RGB image
            rgb_image_path = join(args.output, f"{image_id}_rgb_image_texted_{mask_id}.png")  # 'src/segnetv2_mask2_former/VLM_Results/rgb_image.png'
            cv2.imwrite(rgb_image_path, final_image)
            
            if args.mouse_click:
                output_dir = '/'.join(args.output.split('/')[:-1].append('New_Data'))

                # Find max index of image/mask
                filenames = os.listdir(output_dir)
                if len(filenames) == 0:
                    image_id = 0
                    mask_id = 0
                else:
                    file_ids = [[int(f.split('.')[0].split('_')[0]), int(f.split('.')[0].split('_')[-1])] for f in filenames if f.split('.')[0].split('_')[1] == 'mask']
                    image_id = max([ids[0] for ids in file_ids]) + 1
                    mask_id = max([ids[1] for ids in file_ids]) + 1

                # Save RGB image
                rgb_image_path = join(output_dir, f"{image_id}_rgb_image.png")
                cv2.imwrite(rgb_image_path, new_rgb_image)
                # Save colored depth image
                depth_image = cv2.applyColorMap(np.uint8(depth_image*255), cv2.COLORMAP_JET)
                depth_image_path = join(output_dir, f"{image_id}_depth_image.png")  # 'src/segnetv2_mask2_former/VLM_Results/depth.png'
                cv2.imwrite(depth_image_path, depth_image)
                # Save new segmask
                segmask_path = join(output_dir, f"{image_id}_mask_{mask_id}.png")  # 'src/segnetv2_mask2_former/VLM_Results/segmask.png'
                cv2.imwrite(segmask_path, segmask*255)
            
            # plt.imshow(final_image)
            # plt.show()
            # plt.close()