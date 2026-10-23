import os

IMAGE_PATH = '/home/aurmr/workspaces/test_ros_bare_bone/src/aurmr_perception/VITA_my/data/bin_3F/'

folder_path = IMAGE_PATH
image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
num_images = len(image_files)
num_images= num_images+1

print('Number of PNG images in folder:', num_images)



files = os.listdir(folder_path)

for file_name in files:
    file_name = file_name
    
print(type(file_name))

file_name_new = file_name[:18] + str(num_images) + file_name[19:]

print("Original string:", file_name)
print("Modified string:", file_name_new)

import cv2
out_label = cv2.imread('/home/aurmr/workspaces/test_ros_bare_bone/src/aurmr_perception/VITA_my/vis_real/frame_R50_v26_c17_csf/scene_03_full/result.png', 0)
cv2.imwrite(os.path.join(IMAGE_PATH,file_name_new),out_label)        



folder_path = IMAGE_PATH
image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
num_images = len(image_files)
num_images= num_images+1
files = os.listdir(folder_path)
for file_name in files:
    file_name = file_name
file_name_new = file_name[:18] + str(num_images) + file_name[19:]
cv2.imwrite(os.path.join(IMAGE_PATH,file_name_new),out_label)        
