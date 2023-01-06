import numpy as np
import cv2
# import h5py
import matplotlib.pyplot as plt
import json
from scipy.optimize import linear_sum_assignment

MIN_MATCH_COUNT = 4


def match_masks(im1, im2, mask1, mask2):
    # Convert the images to greyscale
    im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)

    mask_recs = np.zeros(shape=(np.max(mask1), np.max(mask2)))

    sift = cv2.SIFT_create()
    k2, d2 = sift.detectAndCompute(im2, None)

    for i in range(1, np.max(mask1) + 1):
        # Subset the image from the mask
        im1_now = im1 * (mask1 == i)

        k1, d1 = sift.detectAndCompute(im1_now, None)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(d1, d2, 2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        
        if len(good) > MIN_MATCH_COUNT:
            # src_pts = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.round(dst_pts).astype(int)
        
        else:
            print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
            dst_pts = None

        for j in range(1, np.max(mask2) + 1):
            if dst_pts is not None:
                n_hits = np.sum(mask2[dst_pts[:, 0, 1], dst_pts[:, 0, 0]] == j)
                mask_recs[i-1, j-1] = n_hits
            else:
                mask_recs[i - 1, j - 1] = 0
        
        # idx = np.argmax(mask_recs[i-1, :])
        # im2_now = im2 * (mask2 == idx + 1)
        # img3 = cv2.drawMatches(im1_now,k1,im2_now,k2,good,None)
        # plt.imshow(img3, 'gray'), plt.show()

    print("mask_recs", mask_recs)
    # Find the mask in the destination image that best matches the soruce mask
    row_ind, col_ind = linear_sum_assignment(-mask_recs)

    # mask_recs = np.argmax(mask_recs, axis=1) + 1

    return row_ind + 1


# def score(idx, mask_recs):
#     md1 = json.loads(data['frame0_metadata'][idx])
#     md2 = json.loads(data['frame1_metadata'][idx])
#     src_name2id = md1['name_to_id_map']
#     src_id2name=  {v: k for k, v in src_name2id.items()}
#     dst_name2id = md2['name_to_id_map']
#     dst_id2name=  {v: k for k, v in dst_name2id.items()}

#     num_good = 0
#     for i in range(len(mask_recs)):
#         if src_id2name[i + 1] == dst_id2name[mask_recs[i]]:
#             num_good += 1
#             # print("Good")
#         # else:
#             # print(f"Picked {dst_id2name[mask_recs[i]]} but wanted {src_id2name[i + 1]}")
#     return num_good / len(mask_recs)


# # file = "/home/thomas/Desktop/not-flipped/10000.h5"
# file = '/home/thomas/Desktop/ku_data/0714_1/train1_shard_000000.h5'
# # file = "/home/thomas/Desktop/UnseenObjectClustering/data/demo/5k/test_shard_000000.h5"

# data = h5py.File(file, 'r')


# n = 1000
# n_bad = 0
# s = np.array([])

# for q in range(n):
#     if q % 50 == 0:
#         print(f"On {q} of {n}")
#     # Get two images
#     im1 = data['frame0_data'][q]
#     im2 = data['frame1_data'][q]

#     # Get their corresponding masks
#     mask1 = data['frame0_mask'][q]
#     mask2 = data['frame1_mask'][q]

#     # Format the masks correctly
#     mask1 = (mask1 + 1) % 256
#     mask2 = (mask2 + 1) % 256
#     mask1[mask1 == np.max(mask1)] = 0
#     mask2[mask2 == np.max(mask2)] = 0

#     # If mask1 is empty, skip
#     if np.max(mask1) == 0:
#         n_bad += 1
#         continue

#     # plt.imshow(im1)
#     # plt.show()
#     # plt.imshow(mask1)
#     # plt.show()
#     # plt.imshow(im2)
#     # plt.show()

#     # Grab mask predictions
#     mask_recs = match_masks(im1, im2, mask1, mask2)

#     # Score mask recs
#     s = np.append(s, score(q, mask_recs))

#     # Plot two subplots with mask 1 and its recommended mask
#     # print(f"Object {q} score: {s[-1]}")

#     # for i in range(1, np.max(mask1) + 1):
#     #     md1 = json.loads(data['frame0_metadata'][q])
#     #     md2 = json.loads(data['frame1_metadata'][q])
#     #     src_name2id = md1['name_to_id_map']
#     #     src_id2name=  {v: k for k, v in src_name2id.items()}
#     #     dst_name2id = md2['name_to_id_map']
#     #     dst_id2name=  {v: k for k, v in dst_name2id.items()}

#     #     gt_idx = dst_name2id[src_id2name[i]]

#     #     fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
#     #     ax1.imshow(mask1 == i)
#     #     ax2.imshow((mask2 == mask_recs[i - 1]))
#     #     ax3.imshow((mask2 == gt_idx))
#     #     plt.show()

# print(np.mean(s))
# print(n_bad)
# # print(s)