import numpy as np
import cv2
# import h5py
import matplotlib.pyplot as plt
import json
from scipy.optimize import linear_sum_assignment

MIN_MATCH_COUNT = 4

def match_masks_muru(im1, im2, mask1, mask2, global_counter=None, thre=5):
    # Convert the images to greyscale
    # im1_rgb = im1.copy()
    # im2_rgb = im2.copy()
    # im1_hls = (cv2.cvtColor(im1, cv2.COLOR_RGB2HLS))
    # im2_hls = (cv2.cvtColor(im2, cv2.COLOR_RGB2HLS))
    im1_hsv = (cv2.cvtColor(im1, cv2.COLOR_RGB2HSV))
    im2_hsv = (cv2.cvtColor(im2, cv2.COLOR_RGB2HSV))
    im1 = (cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)*255).astype("uint8")
    im2 = (cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)*255).astype("uint8")

    # Will hold mask recommendations
    mask_recs = np.zeros(shape=(np.max(mask1), np.max(mask2)))

    sift = cv2.SIFT_create()
    # plt.imshow(im2)
    # plt.savefig("temp_im.png")
    k2, d2 = sift.detectAndCompute(im2, None)

    for i in range(1, np.max(mask1) + 1):
        # Subset the image from the mask
        im1_now = im1 * (mask1 == i)

        k1, d1 = sift.detectAndCompute(im1_now, None)
        # print(len(k1))
        # print(len(d1))

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
        
        # if global_counter:
        #     matched_img = cv2.drawMatches(im1_now, k1, im2, k2, good, im2, flags=2, matchesThickness=1)
        #     cv2.imwrite(f'SIFT_demo_sequence_higherosion_2f/matched_img{global_counter}_obj{i}.png', matched_img)
        
        if len(good) > MIN_MATCH_COUNT:
            # src_pts = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.round(dst_pts).astype(int)
        
        else:
            # print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
            dst_pts = None

        for j in range(1, np.max(mask2) + 1):
            if dst_pts is not None:
                n_hits = np.sum(mask2[dst_pts[:, 0, 1], dst_pts[:, 0, 0]] == j)
                # if thre:
                #     if n_hits >= thre:
                #         mask_recs[i-1, j-1] = n_hits
                #     else:
                #         mask_recs[i-1, j-1] = 0
                # else:
                mask_recs[i-1, j-1] = n_hits
            else:
                mask_recs[i - 1, j - 1] = 0
        
        # idx = np.argmax(mask_recs[i-1, :])
        # im2_now = im2 * (mask2 == idx + 1)
        # img3 = cv2.drawMatches(im1_now,k1,im2_now,k2,good,None)
        # plt.imshow(img3, 'gray'), plt.show()
    # if global_counter == 1603:
    #     breakpoint()
    # print(mask_recs)
    # Find the mask in the destination image that best matches the source mask
    row_ind, col_ind = linear_sum_assignment(-mask_recs)
    print(row_ind)
    print(col_ind)
    if thre:
        remain_row = []
        remain_col = []
        for row_idx, col_idx in zip(row_ind, col_ind):
            if mask_recs[row_idx, col_idx] < thre:
                remain_row.append(row_idx)
                remain_col.append(col_idx)
        for j in range(0, np.max(mask2)):
            if j not in col_ind:
                remain_col.append(j)
            
        score = np.zeros((len(remain_row), len(remain_col)))
        for i in range(len(remain_row)):
            im1_now = im1_hsv * np.expand_dims((mask1 == remain_row[i]+1), axis=2)
            if np.sum(mask1 == remain_row[i]+1) == 0:
                print(f"Dividing by 0 encountered for object {remain_row[i]+1}")
            avg_im1 = np.sum(im1_now) / np.sum(mask1 == remain_row[i]+1) if np.sum(mask1 == remain_row[i]+1) else 0
            for j in range(len(remain_col)):
                im2_now = im2_hsv * np.expand_dims((mask2 == remain_col[j]+1), axis=2)
                if np.sum(mask2 == remain_col[j]+1) == 0:
                    print(f"Dividing by 0 encountered for object {remain_col[j]+1}")
                avg_im2 = np.sum(im2_now) / np.sum((mask2 == remain_col[j]+1)) if np.sum(mask2 == remain_col[j]+1) else 0
                diff = abs(avg_im1 - avg_im2)
                score[i][j] = diff
        row_ind_remain, col_ind_remain = linear_sum_assignment(score)
        for i in range(len(row_ind_remain)):
            col_ind[remain_row[i]] = remain_col[col_ind_remain[i]]

    # mask_recs = np.argmax(mask_recs, axis=1) + 1
    print(row_ind)
    print(col_ind)
    return row_ind + 1, col_ind + 1

# if __name__ == "__main__":
#     from custom_datasets import frameDataset
#     file_name = '/home/thomas/Desktop/azure_data/azure_24_128.npy'
#     data = np.load(file_name, allow_pickle=True).item()
#     dataset = frameDataset(data, fill_holes=False)
#     print(dataset[0]['frame0'].keys())
#     f0_im = dataset[0]['frame0']['image_color_n'].cpu().squeeze().detach().numpy().transpose(1,2,0)[:,:,::-1]
#     f1_im = dataset[0]['frame1']['image_color_n'].cpu().squeeze().detach().numpy().transpose(1,2,0)[:,:,::-1]
#     f0_mask = dataset[0]['frame0']['label'].cpu().squeeze().detach().numpy()
#     f1_mask = dataset[0]['frame1']['label'].cpu().squeeze().detach().numpy()
#     print(match_masks(f0_im, f1_im, f0_mask, f1_mask))


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