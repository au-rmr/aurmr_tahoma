import queue
import subprocess
import encodings
from fileinput import filename
from typing import Dict, List, Set, Tuple
from unittest import result
import cv2
import matplotlib
from aurmr_perception.srv import CaptureObject, RemoveObject, GetObjectPoints, ResetBin, LoadDataset


from skimage.color import label2rgb

import numpy as np
import ros_numpy
import rospy
import message_filters
import tf2_ros
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from sensor_msgs.msg import Image, CameraInfo, PointField, PointCloud2
from std_msgs.msg import Header
from std_srvs.srv import Trigger
# from aurmr_unseen_object_clustering.tools.run_network import clustering_network
# from aurmr_unseen_object_clustering.tools.match_masks import match_masks
from aurmr_unseen_object_clustering.tools.segmentation_net import SegNet, NO_OBJ_STORED, UNDERSEGMENTATION, OBJ_NOT_FOUND, MATCH_FAILED, IN_BAD_BINS
from aurmr_dataset.io import DatasetReader, DatasetWriter
from aurmr_dataset.dataset import Dataset, Item, Entry
import pickle

import scipy.ndimage as spy


with open('/tmp/calibration_pixel_coords_pod.pkl', 'rb') as f:
    bin_bounds = pickle.load(f)

def_config = {
    # Erosion/Dilation and largest connected component
   'perform_cv_ops':False,
   'perform_cv_ops_ref':True,
   # 'rm_back':True,
   'rm_back':True,
   'rm_back_old':False,
   'kernel_size':5,
   'erosion_num':3,
   'dilation_num':4,
   'rgb_is_standardized':True,
   'xyz_is_standardized':True,
   'min_bg_change':15,
   'resize':True,
   'print_preds':False,
    'min_pixel_threshhold':30,

}


def compute_xyz(depth_img, intrinsic):
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

def numpify_pointcloud(points, im_shape):
    final_shape = list(im_shape[0:2])
    # final_shape.append(3)
    points = ros_numpy.numpify(points)

    points = np.reshape(points, final_shape)
    # points = np.vstack((points['x'],points['y'],points['z']))
    points = np.stack((points['x'],points['y'],points['z']), axis=2)
    
    print(points.shape)
    plt.imshow(points[...,2])
    plt.show()
    return points

def mask_pointcloud(points, mask):
    points = ros_numpy.numpify(points)
    points_seg = points[mask.flatten() > 0]
    points_seg = np.vstack((points_seg['x'],points_seg['y'],points_seg['z']))
    points_seg = points_seg[:, np.invert(np.isnan(points_seg[2, :]))]
    return points_seg

class PodPerceptionROS:
    def __init__(self, model, camera_name, visualize, camera_type):
        self.visualize = visualize
        self.camera_name = camera_name
        self.model = model

        self.tf2_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer)

        self.trigger_capture = rospy.Service('~capture_object', CaptureObject, self.capture_object_callback)
        self.trigger_empty_pod = rospy.Service('~capture_empty_pod', Trigger, self.empty_pod_callback)
        self.trigger_pick = rospy.Service('~pick_object', CaptureObject, self.pick_object_callback)
        self.trigger_stow = rospy.Service('~stow_object', CaptureObject, self.stow_object_callback)
        self.trigger_update = rospy.Service('~update_bin', CaptureObject, self.update_object_callback)
        self.trigger_retrieve = rospy.Service('~get_object_points', GetObjectPoints, self.get_object_callback)
        self.trigger_reset = rospy.Service('~reset_bin', ResetBin, self.reset_callback)


        rospy.loginfo(camera_type)
        if camera_type == "realsense":
            self.camera_depth_subscriber = message_filters.Subscriber(f'/{self.camera_name}/aligned_depth_to_color/image_raw', Image)
            self.camera_rgb_subscriber = message_filters.Subscriber(f'/{self.camera_name}/color/image_raw', Image)
            self.camera_info_subscriber = message_filters.Subscriber(f'/{self.camera_name}/color/camera_info', CameraInfo)
            self.camera_frame = f"{self.camera_name}_color_frame"
        elif camera_type == "azure_kinect":
            self.camera_depth_subscriber = message_filters.Subscriber(f'/{self.camera_name}/depth_to_rgb/image_raw', Image)
            self.camera_rgb_subscriber = message_filters.Subscriber(f'/{self.camera_name}/rgb/image_raw', Image)
            self.camera_info_subscriber = message_filters.Subscriber(f'/{self.camera_name}/depth_to_rgb/camera_info', CameraInfo)
            self.camera_frame = "camera_base"
            self.camera_points_frame = "rgb_camera_link"
        else:
            raise RuntimeError(f"Unknown camera type requested: {camera_type}")
        
        self.points_sub = rospy.Subscriber(f'/{self.camera_name}/points2', PointCloud2, self.points_cb)
        self.masks_pub = rospy.Publisher('~detected_masks', Image, queue_size=1, latch=True)
        self.labels_pub = rospy.Publisher('~labeled_images', Image, queue_size=1, latch=True)
        self.color_image_pub = rospy.Publisher('~colored_images', Image, queue_size=1, latch=True)
        self.camera_synchronizer = message_filters.ApproximateTimeSynchronizer([
            self.camera_depth_subscriber, self.camera_rgb_subscriber, self.camera_info_subscriber], 10, 1)
        self.camera_synchronizer.registerCallback(self.camera_callback)
        self.received_images = False
        self.rgb_image = None
        self.depth_image = None
        self.camera_info = None
        self.points_msg = None

        self.dataset_writer = DatasetWriter("/home/aurmr/workspaces/aurmr_demo")

    def play_camera_shutter(self):
        audio_file = '/usr/share/sounds/freedesktop/stereo/camera-shutter.oga'
        try:
            subprocess.call(['mplayer', audio_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            rospy.logwarn("Couldn't play sound")
            rospy.logwarn(e)

    def capture_image(self):
        self.play_camera_shutter()
        return self.rgb_image.copy(), self.depth_image.copy()


    def load_dataset(self, request):
        dataset = DatasetReader.load()

        K  = np.asarray(dataset.camera_info['depth_to_rgb']['K'] ).reshape((3, 3))

        first_entry = dataset.entries[0]

        self.net = SegNet(init_depth=first_entry.depth_image, init_info=K)
        self.model.net = self.net

        for entry in dataset.entries[1:]:
            rgb_image = entry.rgb_image 
            depth_image = entry.depth_image
            for item_added in entry.inventory():
                self.model.bins[item_added.bin_id].add_new(item_added.asin_id, rgb_image, depth_image)

        self.dataset = dataset
        return {"success": True, "message": f"load dataset {dataset.path}"}


    def capture_object_callback(self, request):
        if not request.bin_id or not request.object_id:
            return False, "bin_id and object_id are required"

        rgb_image, depth_image = self.capture_image()
        rospy.loginfo(request.bin_id)   
        result, message, mask = self.model.capture_object(request.bin_id, request.object_id, rgb_image, depth_image, self.camera_info.K)
        if result:
            new_entry = Entry(self.dataset.metadata["entries"][-1], [Item(request.asin_id, request.bin_id, 0)], rgb_image, depth_image)
            self.dataset.add(new_entry)
        return result, message, mask

    def points_cb(self, msg):
        if self.points_msg is None:
            rospy.loginfo("Points Found")
        self.points_msg = msg

    def empty_pod_callback(self, request):
        intrinsics_3x3 = np.reshape(self.camera_info.K, (3,3))
        rgb_img, depth_img = self.capture_image()
        self.net = SegNet(init_depth=depth_img, init_info=intrinsics_3x3)
        self.model.initialize_empty(rgb_img, depth_img, intrinsics_3x3)
        self.model.net = self.net
        return {"success": True, "message": "empty bin captured"}

    def get_object_callback(self, request):
        if not request.object_id or not request.frame_id:
            return False, "object_id and frame_id are required", None, None

        result, message = self.model.get_object(request.bin_id, request.object_id, self.points_msg, self.depth_image.shape)
        rospy.loginfo(f"Get Object Callback MSG: {message}")
        if not result:
            return result, message, None, None, None


        bin_id, points, mask = result
        bgr_to_rgb_image = cv2.cvtColor(self.rgb_image[:,:,:3].astype(np.uint8), cv2.COLOR_BGR2RGB)

        labled_mask = mask.astype(np.uint8)
        labled_mask[labled_mask>0] = 1
        labled_img = cv2.bitwise_and(bgr_to_rgb_image, bgr_to_rgb_image, mask=labled_mask)
        mask_msg = ros_numpy.msgify(Image, mask.astype(np.uint8), encoding="mono8")  
            
        labled_img_msg = ros_numpy.msgify(Image, labled_img.astype(np.uint8), encoding="rgb8")
        
        colored_img_msg = ros_numpy.msgify(Image, bgr_to_rgb_image, encoding="rgb8")

        common_header = Header()
        common_header.stamp = rospy.Time.now()

        mask_msg.header = common_header
        labled_img_msg.header = common_header
        colored_img_msg.header = common_header

        self.labels_pub.publish(labled_img_msg)
        self.masks_pub.publish(mask_msg)
        self.color_image_pub.publish(colored_img_msg)

        if request.mask_only:
            return True,\
               f"Mask successfully retrieved for object {request.object_id} in bin {bin_id}",\
               bin_id,\
               None,\
               mask_msg

        if request.frame_id != self.camera_points_frame:
            # Transform points to requested frame_id

            stamped_transform = self.tf2_buffer.lookup_transform(request.frame_id, self.camera_points_frame, rospy.Time(0),
                                                                 rospy.Duration(1))
            camera_to_target_mat = ros_numpy.numpify(stamped_transform.transform)
            points = np.vstack([points, np.ones(points.shape[1])])  # convert to homogenous

            points = np.matmul(camera_to_target_mat, points)[0:3, :].T  # apply transform
        else:
            points = points.T

        # Convert numpy points to a pointcloud message
        itemsize = np.dtype(np.float32).itemsize
        points = np.hstack((points, np.ones((points.shape[0], 4))))
        num_points = points.shape[0]

        data = points.astype(np.float32).tobytes()

        fields = [PointField(
            name=n, offset=i * itemsize, datatype=PointField.FLOAT32, count=1)
            for i, n in enumerate('xyzrgba')]

        header = Header(frame_id=request.frame_id, stamp=rospy.Time.now())

        pointcloud = PointCloud2(
            header=header,
            height=1,
            width=num_points,
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=(itemsize * 7),
            row_step=(itemsize * 7 * num_points),
            data=data
        )

        self.play_camera_shutter()

        return True,\
               f"Points and mask successfully retrieved for object {request.object_id} in bin {bin_id}",\
               bin_id,\
               pointcloud,\
               mask_msg

    def pick_object_callback(self, request):
        if not request.bin_id or not request.object_id:
            return False, "bin_id and object_id are required"

        result, message = self.model.remove_object(request.bin_id, request.object_id, self.rgb_image, self.depth_image, self.camera_info, self.points_msg)

        return result, message, None

    def update_object_callback(self, request):
        if not request.bin_id:
            return False, "bin_id is required"
        print("CALLBACK")
        result, message = self.model.update(request.bin_id, self.rgb_image, self.depth_image, self.camera_info, self.points_msg)

        return result, message, None


    def stow_object_callback(self, request):
        if not request.bin_id or not request.object_id:
            return False, "bin_id and object_id are required"
        rgb_image, depth_image = self.capture_image()
        result, message, mask = self.model.add_object(request.bin_id, request.object_id, rgb_image, depth_image, self.camera_info, self.points_msg)
        rospy.loginfo(f"STOW OBJECT MESSAGE: {message}")
        return result, message, mask

    def reset_callback(self, request):
        if not request.bin_id:
            return False, "bin_id is required"
    
        if self.rgb_image is None:
            return False, "No images have been streamed"

        result = self.model.reset(request.bin_id, self.rgb_image)

        return result, None

    def camera_callback(self, ros_depth_image, ros_rgb_image, ros_camera_info):
        k = np.reshape(ros_camera_info.K, (3,3))
        r = np.reshape(ros_camera_info.R, (3,3))
        d = ros_camera_info.D
        p = np.reshape(ros_camera_info.P, (3,4))

        self.received_images = True
        self.rgb_image = ros_numpy.numpify(ros_rgb_image).astype(np.float32)

        # #undistort
        # h,  w = self.rgb_image .shape[:2]
        # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(k, d, (w,h), 0, (w,h))
        # self.rgb_image = cv2.undistort(self.rgb_image, k, d).astype(np.uint8)
        # # crop the image
        # x, y, w, h = roi
        # print(roi)
        # plt.imshow(self.rgb_image)
        # plt.show()
        # self.rgb_image = self.rgb_image[y:y+h, x:x+w]
        # plt.imshow(self.rgb_image)
        # plt.show()
        #undistort
        self.depth_image = ros_numpy.numpify(ros_depth_image).astype(np.float32)
        # h,  w = self.depth_image .shape[:2]
        # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(k, d, (w,h), 1, (w,h))
        # self.depth_image = cv2.undistort(self.depth_image, k, d).astype(np.uint8)
        # # crop the image
        # x, y, w, h = roi
        # self.depth_image = self.depth_image[y:y+h, x:x+w]
        # # img_shape = ros_numpy.numpify(ros_rgb_image).shape[0:2]
        self.camera_info = ros_camera_info
        # self.camera_info.K = newcameramtx.flatten()
        # print(newcameramtx)
        # if self.visualize:
        #     rgb_im_viz = cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2BGR)
        #     cv2.imshow('rgb_image', rgb_im_viz)
        #     cv2.waitKey(1)

class DiffPodModel:
    def __init__(self, diff_threshold, segmentation_method):
        self.diff_threshold = diff_threshold
        self.latest_captures = {}
        self.latest_masks = {}
        self.occluded_table = {}
        self.bin_normals = {}
        self.segmentation_method = segmentation_method
        self.net = None
        self.bin_names = bin_bounds.keys()
        self.bins = {}
        self.H = None
        self.W = None

    def initialize_empty(self, rgb_img, depth_img, camera_intrinsics):
        init_xyz = compute_xyz(depth_img, camera_intrinsics)
        for bin in self.bin_names:
            self.bins[bin] = Bin(bin, bounds=bin_bounds[bin], init_depth=init_xyz)

        self.H, self.W, _ = rgb_img.shape

   # Computes the point cloud from a depth array and camera intrinsics


    def backproject(self, depth_image, mask, camera_intrinsics):
        # Apply mask to the depth image
        masked_depth = depth_image #* mask

        # Get intrinsics reported via camera_info (focal point and principal point offsets)
        fp_x = camera_intrinsics[0]
        fp_y = camera_intrinsics[4]
        ppo_x = camera_intrinsics[2]
        ppo_y = camera_intrinsics[5]

        # Convert the masked depth into a point cloud
        height, width = masked_depth.shape
        nx = np.linspace(0, width-1, width)
        ny = np.linspace(0, height-1, height)
        u, v = np.meshgrid(nx, ny)
        x = (u.flatten() - ppo_x) / fp_x
        y = (v.flatten() - ppo_y) / fp_y
        z = masked_depth.flatten() / 1000
        x = np.multiply(x,z)
        y = np.multiply(y,z)

        x = x[np.nonzero(z)]
        y = y[np.nonzero(z)]
        z = z[np.nonzero(z)]

        # Rearrange axes to match ROS axes
        # Camera: X- right, Y- down, Z- forward
        # ROS: X- forward, Y- left, Z- up
        x_ros = z
        y_ros = -x
        z_ros = -y

        points = np.vstack((x_ros,y_ros,z_ros))
        return points

     # Retrieves the mask with bin product id obj_id from the full camera reference frame

    def get_masks(self):
       mask = np.zeros((self.H, self.W))
       offset = 0
       for bin in self.bins.values():
           if bin.current['mask'] is not None:
               mask_bin = bin.current['mask'].copy()
               mask_bin[mask_bin > 0] += offset
               r1,r2,c1,c2 = bin.bounds
               mask[r1:r2, c1:c2] = mask_bin
 
               offset += np.max(mask_bin)
 
       return mask

    def capture_object(self, bin_id, object_id, rgb_image, depth_image, camera_intrinsics):
        if not bin_id or not object_id:
            return False

        if bin_id not in self.latest_captures:
            return False, f"Bin {bin_id} has not been reset"
        

        last_rgb = self.latest_captures[bin_id]
        current_rgb = rgb_image

        points = rospy.wait_for_message(f'/camera_lower_right/points2', PointCloud2)
        points = mask_pointcloud(points, mask)

        if bin_id not in self.points_table:
            self.points_table[bin_id] = {}
            self.masks_table[bin_id] = {}

        self.points_table[bin_id][object_id] = points
        self.masks_table[bin_id][object_id] = mask


        # Matches masks with previous mask labels
        # The (i-1)-th entry of recs is the mask id of the best match for mask i in the second image
        if not self.latest_masks:
            self.latest_masks[bin_id] = np.zeros(mask.shape, dtype=np.uint8)

        self.latest_captures[bin_id] = current_rgb
        self.latest_masks[bin_id] = mask

        return True, f"Object {object_id} in bin {bin_id} has been captured with {points.shape[0]} points.", ros_numpy.msgify(Image, mask.astype(np.uint8), encoding="mono8")

    def add_object(self, bin_id, object_id, rgb_image, depth_image, camera_intrinsics, points_msg):
        if not bin_id or not object_id:
            return False, "bin_id and object_id are required"
        intrinsics_3x3 = np.reshape(camera_intrinsics.K, (3,3))
        bin = self.bins[bin_id]
        xyz = compute_xyz(depth_image, intrinsics_3x3)
        # check whether the stow is valid (there is a meaningful pixel difference)
        if not bin.new_object_detected(xyz) and False: #FIXME
            print(f"No new object detected: Did you store an object in bin {bin_id}?\nPlease try again")
            if bin_id not in self.occluded_table:
                self.occluded_table[bin_id] = {}
            self.occluded_table[bin_id][object_id] = True
            msg = f"Object {object_id} in bin {bin_id} was not detected, it is either occluded or was never placed."
            mask_ret = None
            success = False
            return success, msg, mask_ret

        bin.add_new(object_id, rgb_image, xyz)

        # After object with product id obj_id is stowed in bin with id bin_id, 
        #           updates perception using the current camera information (rgb_raw, depth_raw, intrinsic)
        # FAILURES:
        #       1. No object is stored. Covered using depth difference in bin.new_object_detected CODE NO_OBJ_STORED
        #       2. Fewer than n objects are segmented (undersegmentation). Covered in refine_masks CODE UNDERSEGMENTATION

        # Check if in bad bins
        if self.bins[bin_id].bad:
            return False, "Bin is already bad. Not updating masks", None
  

        # Current mask recommendations on the bin
        mask_crop, embeddings = self.net.segment(bin.get_history_images())

        if mask_crop is None:
            bin.bad = True
            return False, "Segmentation failed", None

        # plt.imshow(mask_crop)
        # plt.title("Predicted object mask before refinement")
        # plt.show()

        # Make sure that the bin only has segmentations for n objects
        mask_crop, embeddings = self.net.refine_masks(mask_crop, bin.n, embeddings)
        
        # mask2vis = self.vis_masks(bin.current['rgb'], mask_crop)
        # plt.imshow(mask2vis)
        # plt.title(f"Masks in the scene. There should be {bin.n}")
        # plt.show()

        if mask_crop is None:
            print(f"Bin {bin_id} added to bad bins. CAUSE Undersegmentation")
            bin.bad = True
            return False, f"Bin {bin_id} added to bad bins. CAUSE Undersegmentation", None

        #mask2vis = self.vis_masks(bin.current['rgb'], mask_crop)
        # plt.imshow(mask2vis)
        # plt.title(f"Masks in the scene (stow). There should be {bin.n} but there are {np.unique(mask_crop)}")
        # plt.show()

  
        # Find the recommended matches between the two frames
        if bin.prev and bin.prev['mask'] is not None:
            prev_rgb = bin.prev["rgb"]
            cur_rgb = bin.current["rgb"]
            prev_mask = bin.prev["mask"]
            prev_emb = bin.prev["embeddings"]
            recs, match_failed, row_recs = self.net.match_masks_using_embeddings(prev_rgb,cur_rgb, prev_mask, mask_crop, prev_emb, embeddings)

            if(match_failed):
                print(f"WARNING: Embeddings Matching Failure on bin {bin_id}. But not Appending to bad bins yet.")
                recs, sift_failed, row_recs = self.net.match_masks_using_sift(prev_rgb,cur_rgb, prev_mask, mask_crop)
                match_failed = False

                if sift_failed:
                    print(f"WARNING: SIFT Matching Failure on bin {bin_id}. Appending to bad bins.")
                    bin.bad = True
  
            # Find the index of the new object (not matched)
            for i in range(1, bin.n + 1):
                if i not in recs:
                    recs = np.append(recs, i)
                    break
            
            # Update the new frame's masks
            mask, embed = self.net.update_masks(mask_crop, recs, embeddings)
        else:
            # This should only happen at the first insertion
            assert(bin.n == 1)
            # No matching is necessary because there is only one object in the scene
            mask = mask_crop.copy()
            embed = embeddings
        
        bin.update_current(mask=mask, embeddings=embed)
  
        #mask = self.net.get_masks()
    
        if False:
            rospy.loginfo("No stow occurred. Returning.")
            return True, f"Object {object_id} in bin {bin_id} hasn't been detected.", ros_numpy.msgify(Image, mask.astype(np.uint8), encoding="mono8")
        # plt.imshow(mask)
        # plt.title(f"get_object all masks for {object_id} in {bin_id}")
        # plt.show()
        rospy.loginfo(np.unique(mask, return_counts=True))
        # mask = (mask != 0).astype(np.uint8)
        cv2.imwrite("/tmp/mask.bmp", mask)
        cv2.imwrite("/tmp/image.png", rgb_image)
        # plt.imshow(mask)
        # plt.show()

        if not bin.bad:
            bin.refresh_points_table(points_msg)
            mask = bin.get_obj_mask(object_id)
            num_points = bin.points_table[object_id].shape[0]
            mask_ret = ros_numpy.msgify(Image, mask.astype(np.uint8), encoding="mono8")
            msg = f"Object {object_id} in bin {bin_id} has been stowed with {num_points} points."
            success = True
        else:
            msg = f"Object {object_id} in bin {bin_id} could not be SIFT matched. Bin state is lost."
            mask_ret = None
            success = False
        self.latest_captures[bin_id] = rgb_image
        self.latest_masks[bin_id] = mask
        # plt.imshow(mask)
        # plt.title(f"add_object {object_id} in {bin_id}")
        # plt.show()

        return success, msg, mask_ret

    def get_object(self, bin_id, object_id, points_msg, im_shape):
        
        if not object_id:
            return False, "object_id is required"

        if bin_id in self.occluded_table and object_id in self.occluded_table[bin_id]:
            return False, f"Object {object_id} in bin {bin_id} was occluded or never stowed and could not be found"

        if bin_id not in self.bin_names:
            return False, f"Bin {bin_id} was not found"
        bin = self.bins[bin_id]

        if object_id not in bin.object_ids_counts.keys():
            return False, f"Object {object_id} was not found in bin {bin_id}"
        
        if bin.bad:
            msg = f"Object {object_id} in bin {bin_id} was could not be segmented, but was selected by user"
            return False, msg
        else:
            mask = bin.masks_table[object_id]
            msg = f"Object {object_id} in bin {bin_id} was found"

        points = bin.points_table[object_id]
        # plt.imshow(mask)
        # plt.title(f"get_object {object_id} in {bin_id}")
        # plt.show()
        return (bin_id, points, mask), msg

    def remove_object(self, bin_id, object_id, rgb_image, depth_image, camera_intrinsics, points_msg):
        if bin_id not in self.bin_names:
            return False, f"Bin {bin_id} was not found"

        bin = self.bins[bin_id]
        if object_id not in bin.points_table.keys():
            return False, f"Object {object_id} was not found in bin {bin_id}"


        intrinsics_3x3 = np.reshape(camera_intrinsics.K, (3,3))
        bin.remove(object_id, rgb_image, compute_xyz(depth_image,intrinsics_3x3))

        self.latest_captures[bin_id] = rgb_image

         # Current mask recommendations on the bin
        mask_crop, embeddings = self.segment(bin_id)


        # mask2vis = self.vis_masks(bin.current['rgb'], mask_crop)
        # plt.imshow(mask2vis)
        # plt.title(f"Masks in the scene (pick). There should be {bin.n}")
        # plt.show()
  
        # Make sure that the bin only has segmentations for n objects
        mask_crop, embeddings = self.refine_masks(mask_crop, bin.n, embeddings)

        if mask_crop is None:
            print(f"Adding {bin_id} to bad bins. Reason: Undersegmentation")
            bin.bad = True
            return UNDERSEGMENTATION

        # Optional visualization
        # if self.config['print_preds']:
        #     plt.imshow(mask_crop)
        #     plt.title("Cropped mask prediction after pick")
        #     plt.show()
        try:
            # FIXME: THIS NEEDS NEW BIN API FOR GETTING N OF THE REMOVED OBJECT
            old_mask = bin.prev['mask'].copy()
            old_embeddings = bin.prev['embeddings'].copy()
            # Remove the object that is no longer in the scene
            old_mask[old_mask == obj_n] = 0
            old_mask[old_mask > obj_n] -= 1
            old_embeddings = np.delete(old_embeddings, obj_n-1, 0)
        except Exception as e:
            print(e)
            old_embeddings = np.ones((1, 256), dtype = np.float64)
            old_mask = np.zeros((256, 256), dtype = np.uint8)

        # Find object correspondence between scenes
        recs, match_failed, row_recs = self.match_masks_using_embeddings(bin.prev['rgb'],bin.current['rgb'], old_mask, mask_crop, old_embeddings, embeddings)
        # print("matching method embeddings and sift", recs, recs_)
        
        if(match_failed):
            print(f"WARNING: Embeddings Matching Failure on bin {bin_id}. But not Appending to bad bins yet.")
            recs, sift_failed, row_recs = self.net.match_masks_using_sift(bin.prev['rgb'],bin.current['rgb'], old_mask, mask_crop)
            match_failed = False

            if sift_failed:
                print(f"Adding {bin_id} to bad bins. CAUSE: Unconfident matching")
                bin.bad = True
                return MATCH_FAILED

        # Checks that SIFT could accurately mask objects
        # if recs is None:
        #     print(f"SIFT can not confidently determine matches for bin {bin_id}. Reset bin to continue.")
        #     self.bad_bins.append(bin_id)
        #     return 1

        mask, emb = self.net.update_masks(mask_crop, recs, embeddings)
        bin.update_current(mask=mask, embeddings=emb)


        if not bin.bad:
            bin.refresh_points_table(points_msg)

        return True, f"Object {object_id} in bin {bin_id} has been removed"

    def update(self, bin_id, rgb_image, depth_image, camera_intrinsics, points_msg):
        if bin_id not in self.bin_names:
            rospy.logwarn("no bin id")
            return False, f"Bin {bin_id} was not found"

        self.latest_captures[bin_id] = rgb_image
        bin = self.bins[bin_id]

        intrinsics_3x3 = np.reshape(camera_intrinsics.K, (3,3))        

        # Update the current state of the bin and scene with the new images
        rgb = rgb_image.astype(np.uint8)
        xyz = compute_xyz(depth_image, intrinsics_3x3)

        bin.update_current(rgb, xyz)

        # Current mask recommendations on the bin
        mask_crop, embeddings = self.net.segment(bin.get_history_images())
        
        if mask_crop is None:
            bin.bad = True
            return False, "bin state was lost"

        # Make sure that the bin only has segmentations for n objects
        mask_crop, embeddings = self.net.refine_masks(mask_crop, bin.n, embeddings)
        

        if mask_crop is None:
            bin.bad = True
            return UNDERSEGMENTATION, "bin state was lost"

        # if mask_crop is None:
        #     print(f"Segmentation could not find objects in bin {bin_id}")
        #     input("Need to reset bin. Take out all items and reput them in.")
        #     self.bad_bins.append(bin_id)
        #     self.reset_bin(bin_id)

        # Find the recommended matches between the two frames
        if bin.prev['mask'] is not None:
            recs, match_failed, row_recs = self.net.match_masks_using_embeddings(bin.prev['rgb'],bin.current['rgb'], bin.prev['mask'], mask_crop, bin.prev['embeddings'], embeddings)
            # print("matching method embeddings and sift", recs, recs_)

            if(match_failed):
                print(f"WARNING: Embeddings Matching Failure on bin {bin_id}. But not Appending to bad bins yet.")
                recs, sift_failed, row_recs = self.net.match_masks_using_sift(bin.prev['rgb'],bin.current['rgb'], bin.prev['mask'], mask_crop)
                match_failed = False

                if sift_failed:
                    print(f"WARNING: SIFT Matching Failure on bin {bin_id}. Appending to bad bins.")
                    bin.bad = True
            
            # if recs is None:
            #     print(f"SIFT could not confidently match bin {bin_id}")
            #     figure, axis = plt.subplots(2,)
            #     axis[0].imshow(bin.last['rgb'])
            #     axis[1].imshow(bin.current['rgb'])
            #     plt.title("Frames that sift failed on")
            #     plt.show()
            #     self.bad_bins.append(bin_id)
            
            # Update the new frame's masks
            mask, emb = self.net.update_masks(mask_crop, recs, embeddings)
            bin.update_current(mask=mask, embeddings=emb)
            # plt.imshow(bin.current['mask'])
            # plt.title("After update")
            # plt.show()
        
        else:
            print(bin.prev)
            print("ERROR: THIS SHOULD NEVER HAPPEN!!!")
            return False, "Something went very wrong"


        if not bin.bad:
            bin.refresh_points_table(points_msg)

        return True, f"{bin_id} was updated"

    def reset(self, bin_id, rgb_image):
        if not bin_id:
            return False

        self.latest_captures[bin_id] = rgb_image
        self.bins[bin_id].reset()
        

        return True


class Bin:
    def __init__(self, bin_id: str, bounds=None, init_depth=None, config=def_config):
        self.bin = bin_id
        self.bounds = bounds
        self.config = config
        
        self.init_depth = init_depth[bounds[0]:bounds[1], bounds[2]:bounds[3]]
        self.H, self.W, _ = init_depth.shape
        self.bg_mask = np.ones(self.init_depth.shape, dtype=np.uint8)
        #FIXME SHOULD THIS HAVE ANYTHING?
        self.history = [{'rgb':np.zeros(shape=(self.bounds[1] - self.bounds[0], self.bounds[3] - self.bounds[2], 3), dtype=np.uint8), 'depth':None, 'mask':None, 'embeddings':np.array([]), 'action': None, "object_id": None, "n": None}]
        self.points_table = {}
        self.masks_table = {}
        self.bad = False

    def crop(self, img_array):
        # RGB might come in with a 4th dim
        return img_array[self.bounds[0]:self.bounds[1], self.bounds[2]:self.bounds[3], 0:3]

    def uncrop(self, img_array):
        full = np.zeros((self.H, self.W), dtype=np.uint8)
        r1, r2, c1, c2 = self.bounds
        full[r1:r2, c1:c2] = img_array
        return full

    def add_new(self, object_id, rgb=None, xyz=None, mask=None, embeddings=None):
        new_entry = {"object_id": object_id, "rgb": None, "depth": None, "mask": None, "embeddings": None, "object_id": object_id, "action": "stow", "n": self.n + 1}
        if rgb is not None:
            new_entry['rgb'] = self.crop(rgb)
        if xyz is not None:
            new_entry['depth'] = self.crop(xyz)
            self.bg_mask = np.abs(new_entry['depth'][...,2] - self.init_depth[...,2]) < self.config['min_bg_change']
        new_entry['embeddings'] = embeddings
  
        # plt.imshow(self.bg_mask)
        # plt.title("Background Mask")
        # plt.show()
        new_entry['mask'] = mask
        self.history.append(new_entry)

    def update_current(self, rgb=None, depth=None, mask=None, embeddings=None):
        if rgb is not None:
            self.history[-1]['rgb'] = self.crop(rgb)
        if depth is not None:
            self.history[-1]['depth'] = self.crop(depth)
            self.bg_mask = np.abs(self.history[-1]['depth'][...,2] - self.init_depth[...,2]) < self.config['min_bg_change']

        if embeddings is not None:
            self.history[-1]['embeddings'] = embeddings
        
        if mask is not None:
            self.history[-1]['mask'] = mask
            
        # plt.imshow(self.bg_mask)
        # plt.title("Background Mask")
        # plt.show()
        

    def remove(self, object_id: str, rgb, depth):
        del self.points_table[object_id]
        del self.masks_table[object_id]
        self.history.append({"rgb": None, "depth": None, "mask": None, "embeddings": None, "action": "pick", "object_id": object_id, "n": None})

    def reset(self):
        self.masks_table.clear()
        self.points_table.clear()
        self.bg_mask = np.ones(self.init_depth.shape, dtype=np.uint8)
        self.history = [{'rgb':np.zeros(shape=(self.bounds[1] - self.bounds[0], self.bounds[3] - self.bounds[2], 3), dtype=np.uint8), 'depth':None, 'mask':None, 'embeddings':np.array([]), "action": None, "object_id": None}]
        self.bad = False

    def new_object_detected(self, xyz) -> bool:
        # Crop the current depth down to bin size
        r1,r2,c1,c2 = self.bounds
        depth_now = xyz[r1:r2, c1:c2, 2]
        
        if self.history[-1]['depth'] is None:
            depth_bin = self.init_depth[...,2]
        else:
            depth_bin = self.history[-1]['depth'][...,2]
        

        mask = (np.abs(depth_bin - depth_now) > self.config['min_pixel_threshhold'])
        kernel = np.ones(shape=(9,9), dtype=np.uint8)
        mask_now = spy.binary_erosion(mask, structure=kernel, iterations=2)

        if np.sum(mask_now) > 0:
            return True
        return False

    @property
    def current(self):
        if len(self.history) == 0:
            return None
        return self.history[-1]
    
    @property
    def prev(self):
        if len(self.history) < 2:
            return None
        return self.history[-2]
    
    @property
    def n(self):
        count = 0
        for snapshot in self.history:
            if snapshot["action"] == "stow":
                count += 1
            elif snapshot["action"] == "pick":
                count -= 1
        return count
    
    @property
    def object_ids_counts(self) -> "Dict[str]":
        contained = defaultdict(int)
        for snapshot in self.history:
            if snapshot["action"] == "stow":
                contained[snapshot["object_id"]] += 1
            elif snapshot["action"] == "pick":
                contained[snapshot["object_id"]] -= 1
        return contained
    
    def get_history_images(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        return [(snap["rgb"], snap["depth"]) for snap in self.history]
    
    def get_obj_mask(self, obj_id):
        for entry in reversed(self.history):
            if entry["object_id"] == obj_id:
                mask_crop = entry["mask"]
                obj_id = entry["n"]
                break
        else:
            return None
        return self.uncrop(np.array((mask_crop == id)).astype(np.uint8))
    

    def refresh_points_table(self, points_msg):
        for o_id in self.object_ids_counts.keys():
            mask = self.get_obj_mask(o_id)
            self.masks_table[o_id] = mask
            self.points_table[o_id] = mask_pointcloud(points_msg, mask)

       
