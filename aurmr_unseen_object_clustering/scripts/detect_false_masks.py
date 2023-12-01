import cv2
import numpy as np
import imutils
from timeit import default_timer as timer

mask = cv2.imread("/home/aurmr/workspaces/aurmr_demo_perception/src/segnetv2_mask2_former/Mask_Results/mask.png", cv2.IMREAD_GRAYSCALE)
start = timer()
n = 3
mask[170:201, 125:180] = 5
# cv2.imshow("initial mask", mask*30)
# cv2.waitKey(0)
print(np.max(mask))

if(np.max(mask) > n):
    for i in range(np.max(mask) - n):
        areas = np.array([np.sum(mask == i) for i in range(1, np.max(mask) + 1)])
        smallest_mask_id = np.argmin(areas) + 1
        smallest_mask = np.zeros_like(mask)
        smallest_mask[mask == smallest_mask_id] = 1

        cnts = cv2.findContours(smallest_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        c = max(cnts, key=cv2.contourArea)
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        top_left_smallest = [extLeft[0], extTop[1]]
        bot_right_smallest = [extRight[0], extBot[1]]

        if(areas[smallest_mask_id-1] < 250):
            mask[mask == smallest_mask_id] = 0
            mask[mask > smallest_mask_id] -= 1
            continue

        inside_bound = 0
        for j in range(1, np.max(mask)+1):
            if(j == smallest_mask_id):
                continue
            compare_mask = np.zeros_like(mask)
            compare_mask[mask == j] = 1
            cnts = cv2.findContours(compare_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)
            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])
            top_left = [extLeft[0], extTop[1]]
            bot_right = [extRight[0], extBot[1]]

            if(top_left[0] <= top_left_smallest[0]+10 and bot_right[0] >= bot_right_smallest[0]-10):
                if(top_left[1] <= top_left_smallest[1]+10 and bot_right[1] >= bot_right_smallest[1]-10):
                    print("inside bounds")
                    if(inside_bound == 0):
                        bounds_smallest_top_left = top_left
                        bounds_smallest_bot_right = bot_right
                        inside_bound = j
                    else:
                        if(top_left[0] >= bounds_smallest_top_left[0] and bot_right[0] <= bounds_smallest_bot_right[0]):
                            if(top_left[1] >= bounds_smallest_top_left[1] and bot_right[1] <= bounds_smallest_bot_right[1]):
                                bounds_smallest_top_left = top_left
                                bounds_smallest_bot_right = bot_right
                                inside_bound = j
        if(inside_bound > 0):
            mask[mask == smallest_mask_id] = inside_bound
            mask[mask > smallest_mask_id] -= 1
        else:
            mask[mask == smallest_mask_id] = 0
            mask[mask > smallest_mask_id] -= 1

areas = np.array([np.sum(mask == i) for i in range(1, np.max(mask) + 1)])
print(areas)
end = timer()
print(end - start)
cv2.imshow("final first mask", mask*30)
cv2.waitKey(0)
            
            ## 125 170 180 201
            

# while np.max(mask) > n:
#     areas = np.array([np.sum(mask == i) for i in range(1, np.max(mask) + 1)])
#     # Finds the ID of the largest cluster
#     idx_max = np.argmax(areas) + 1
#     if(n == 0):
#         mask[mask == idx_max] = 0
#         # removing the embeddings as well from the indexof mask subtracting 1 for the object index
#         continue
#     print("Unique mask and areas ",np.unique(mask), areas)
#     nonzero_idx = np.where(np.sum(mask == idx_max, axis=1) > 0)
           

#     c1 = nonzero_idx[0][0]
#     c2 = nonzero_idx[-1][0]

#     print(f"I think that the bounds for the largest mask are {c1} to {c2}")

#     # Find the smallest area under the largest mask

#     idx_min = np.argmin(areas) + 1
#     nonzero_idx_under = np.where(np.sum(mask == idx_min, axis=1) > 0)
#     print(nonzero_idx_under)
#     c1_under = nonzero_idx_under[0][0]
#     c2_under = nonzero_idx_under[-1][0]

#     # While the area found isn't beneath the largest mask
#     while c1_under < (c1 - 10) or c2_under > (c2 + 10):
#         # Remove it from areas
#         areas = np.delete(areas, idx_min - 1)
#         mask[mask == idx_min] = 0
#         mask[mask > idx_min] -= 1
#         print(areas)
        

#         # Recalculate smallest component
#         idx_min = np.argmin(areas) + 1
#         nonzero_idx_under = np.where(np.sum(mask == idx_min, axis=1) > 0)
#         c1_under = nonzero_idx_under[0][0]
#         c2_under = nonzero_idx_under[-1][0]
#         # If we've gone through all areas, break
#         if areas.shape[0] == 0 or areas.shape[0]==n:
#             break
    
#     while c1_under >= (c1 - 10) and c2_under <= (c2 + 10):
#         # Remove it from areas
#         areas = np.delete(areas, idx_min - 1)
#         mask[mask == idx_min] = idx_max
#         mask[mask > idx_min] -= 1
#         print(areas)
        

#         # Recalculate smallest component
#         idx_min = np.argmin(areas) + 1
#         nonzero_idx_under = np.where(np.sum(mask == idx_min, axis=1) > 0)
#         c1_under = nonzero_idx_under[0][0]
#         c2_under = nonzero_idx_under[-1][0]
#         # If we've gone through all areas, break
#         if areas.shape[0] == 0 or areas.shape[0]==n:
#             break
    
#     if areas.shape[0] == n:
#         break
#     cv2.imshow("before change mask", mask*30)
#     cv2.waitKey(0)
#     # Otherwise, a mask was found. Merge them.
#     print(idx_min, idx_max)
#     # n += 1
# # If there are STILL too many masks
# #       remove the smallest
# while np.max(mask) > n:
#     print("Merge method failed. Removing excess")
#     # Calculate areas for each mask
#     areas = np.array([np.sum(mask == i) for i in range(1, np.max(mask) + 1)])
#     idx_min = np.argmin(areas) + 1
#     # Update masks
#     mask[mask == idx_min] = 0
#     mask[mask > idx_min] -= 1


# # If there are too few masks
# #       split the largest in half along the vertical axis
# while np.max(mask) < n:
#     print(f"We only see {np.max(mask)} of {n} masks")

# # cv2.imshow("final mask", mask*30)
# # cv2.waitKey(0)