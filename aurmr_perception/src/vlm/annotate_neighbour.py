import cv2
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt

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

cv2.namedWindow("Image Window")
cv2.setMouseCallback("Image Window", mouse_click_event)

# Load the image
rgb_image_path = "src/segnetv2_mask2_former/Mask_Results/1699238205rgb_first.png"
image = cv2.imread(rgb_image_path)

# Load the mask
segmask_path = "src/segnetv2_mask2_former/Mask_Results/1699238205mask_first.png"
mask = cv2.imread(segmask_path, 0)

depth_image_path = "src/segnetv2_mask2_former/Mask_Results/1699238205depth.png"
depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
depth_image = depth_image.astype(np.float32)/1000.0

depth_image = cv2.applyColorMap(np.uint8(depth_image*255), cv2.COLORMAP_JET)


cv2.imshow("Image Window", image)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyWindow("My Window")
    exit()

image_rgb = image[top_left_click_y:bottom_right_click_y, top_left_click_x:bottom_right_click_x]
mask = mask[top_left_click_y:bottom_right_click_y, top_left_click_x:bottom_right_click_x]
depth_image = depth_image[top_left_click_y:bottom_right_click_y, top_left_click_x:bottom_right_click_x]

target_id = 1

segmask = np.zeros_like(mask)
segmask[mask == target_id] = 1
shrink_size = 3
kernel = np.ones((shrink_size*2 + 1, shrink_size*2 + 1), np.uint8)
segmask = cv2.erode(segmask, kernel, iterations = 1)

target_coords = np.column_stack(np.where(segmask != 0))
x_min, y_min = target_coords.min(axis=0)
x_max, y_max = target_coords.max(axis=0)

image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

depth_im_pil = Image.fromarray(depth_image)
new_width, new_height = image_rgb.shape[1]*6, image_rgb.shape[0]*6
image = cv2.resize(image_rgb, (new_width, new_height))
num_points_per_side_x = int((x_max - x_min)/20)
num_points_per_side_y = int((y_max - y_min)/20) 

x_min_grid = 0
y_min_grid = 0
x_max_grid = (x_max - x_min)
y_max_grid = (y_max - y_min)
x_spacing = (x_max - x_min) // num_points_per_side_x
y_spacing = (y_max - y_min) // num_points_per_side_y
# Generate grid points
grid_points = [(x, y) for x in range(x_min + x_spacing // 2, x_max, x_spacing) 
                      for y in range(y_min + y_spacing // 2, y_max, y_spacing)]


image = np.array(image)
concatenate_count = 0

imgs = []
target_point = 10

for i, point in enumerate(grid_points, 1):  # Start enumerating from number 1
    if(segmask[point[0], point[1]] == 0):
        continue

    if(i == target_point):
        print("Image Pixel 1x1", top_left_click_x+int((point[1]*6)/6), top_left_click_y+int((point[0]*6-120)/6))
        image_viz = image.copy()
        image_viz = cv2.circle(image_viz, (point[1]*6, point[0]*6-120), 200, (255, 255, 255), 20)
        final_image = cv2.resize(image_viz, (image.shape[1]//3, image.shape[0]//3))
        imgs.append(Image.fromarray(np.uint8(final_image)))
        final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)

        print("Image Pixel 1x2", top_left_click_x+int((point[1]*6-120)/6), top_left_click_y+int((point[0]*6)/6))
        image_viz = image.copy()
        image_viz = cv2.circle(image_viz, (point[1]*6-120, point[0]*6), 200, (255, 255, 255), 20)
        final_image = cv2.resize(image_viz, (image.shape[1]//3, image.shape[0]//3))
        imgs.append(Image.fromarray(np.uint8(final_image)))
        final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)

        print("Image Pixel 1x3", top_left_click_x+int((point[1]*6+120)/6), top_left_click_y+int((point[0]*6)/6))
        image_viz = image.copy()
        image_viz = cv2.circle(image_viz, (point[1]*6+120, point[0]*6), 200, (255, 255, 255), 20)
        final_image = cv2.resize(image_viz, (image.shape[1]//3, image.shape[0]//3))
        imgs.append(Image.fromarray(np.uint8(final_image)))
        final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)

        print("Image Pixel 2x1", top_left_click_x+int((point[1]*6)/6), top_left_click_y+int((point[0]*6+120)/6))
        image_viz = image.copy()
        image_viz = cv2.circle(image_viz, (point[1]*6, point[0]*6+120), 200, (255, 255, 255), 20)
        final_image = cv2.resize(image_viz, (image.shape[1]//3, image.shape[0]//3))
        imgs.append(Image.fromarray(np.uint8(final_image)))
        final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)

        print("Image Pixel 2x2", top_left_click_x+int((point[1]*6)/6), top_left_click_y+int((point[0]*6)/6))
        image_viz = image.copy()
        image_viz = cv2.circle(image_viz, (point[1]*6, point[0]*6), 200, (255, 255, 255), 20)
        final_image = cv2.resize(image_viz, (image.shape[1]//3, image.shape[0]//3))
        imgs.append(Image.fromarray(np.uint8(final_image)))
        final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)

min_img_shape = sorted([(min(i.width, i.height), i.width, i.height) for i in imgs])[0]
imgs_comb = [i.resize((min_img_shape[1], min_img_shape[2])) for i in imgs]
new_im = Image.new('RGB', (3 * min_img_shape[1], 2 * min_img_shape[2]))
new_im.paste(imgs_comb[0], (0, 0))
new_im.paste(imgs_comb[1], (min_img_shape[1], 0))
new_im.paste(imgs_comb[2], (min_img_shape[1]*2, 0))
new_im.paste(imgs_comb[3], (0, min_img_shape[2]))
new_im.paste(imgs_comb[4], (min_img_shape[1], min_img_shape[2]))
new_im.save(f'src/segnetv2_mask2_former/VLM_Results/rgb_image_concatenate_neighbours_neighbours.png')