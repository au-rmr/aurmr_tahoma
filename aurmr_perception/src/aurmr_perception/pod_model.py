import os
import subprocess
import encodings
from fileinput import filename
from typing import Any, Dict, List, Set, Tuple
from unittest import result
from aurmr_perception.bin_model import BinModel
import cv2
import matplotlib
from aurmr_perception.srv import ActOnBins, CaptureObject, RemoveObject, GetObjectPoints, ResetBin, LoadDataset


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
from aurmr_dataset.io import DatasetReader, DatasetWriter, DatasetAppender
from aurmr_dataset.dataset import Dataset, Item, Action, CameraData, CameraConfig
import pickle

import scipy.ndimage as spy

from aurmr_perception.util import compute_xyz, mask_pointcloud


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


class PodPerceptionROS:
    def __init__(self, model: "PodModel", dataset_path: str, camera_name: str, visualize, camera_type):
        self.visualize = visualize
        self.camera_name = camera_name
        self.model = model

        self.dataset_path = dataset_path
        self.dataset_writer = DatasetAppender(dataset_path)

        self.tf2_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer)

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

        self.rgb_image = None
        self.depth_image = None
        self.camera_info = None
        self.camera_synchronizer = message_filters.ApproximateTimeSynchronizer([
            self.camera_depth_subscriber, self.camera_rgb_subscriber, self.camera_info_subscriber], 3, .1)
        self.camera_synchronizer.registerCallback(self.camera_callback)

        self.wait_for_camera_data()

        self.trigger_capture = rospy.Service('~capture_object', CaptureObject, self.capture_object_callback)
        self.trigger_empty_pod = rospy.Service('~capture_empty_pod', Trigger, self.empty_pod_callback)
        self.trigger_pick = rospy.Service('~pick_object', CaptureObject, self.pick_object_callback)
        self.trigger_stow = rospy.Service('~stow_object', CaptureObject, self.stow_object_callback)
        self.act_on_bins_srv = rospy.Service('~act_on_bins', ActOnBins, self.act_on_bins_callback)
        self.trigger_update = rospy.Service('~update_bin', CaptureObject, self.update_object_callback)
        self.trigger_retrieve = rospy.Service('~get_object_points', GetObjectPoints, self.get_object_callback)
        self.trigger_reset = rospy.Service('~reset_bin', ResetBin, self.reset_callback)
        self.load_dataset_srv = rospy.Service('~load_dataset', LoadDataset, self.load_dataset_callback)

        self.masks_pub = rospy.Publisher('~detected_masks', Image, queue_size=1, latch=True)
        self.labels_pub = rospy.Publisher('~labeled_images', Image, queue_size=1, latch=True)
        self.color_image_pub = rospy.Publisher('~colored_images', Image, queue_size=1, latch=True)


    def wait_for_camera_data(self):
        while not rospy.is_shutdown():
            missing_something = ""
            if self.camera_info is None:
                missing_something += " camera_info"
            if self.rgb_image is None:
                missing_something += " rgb image"
            if self.depth_image is None:
                missing_something += " depth image"

            if not missing_something:
                break
            rospy.logwarn_throttle(5.0, f"Waiting for{missing_something}. Services not started")
        # Note that this will also print when killing the node with Ctrl-C
        rospy.loginfo("Camera data received")

    def camera_callback(self, ros_depth_image, ros_rgb_image, ros_camera_info):
        self.received_images = True
        # Azure connect's image encoding is brga8. We'll drop the useless a channel here
        self.rgb_image = ros_numpy.numpify(ros_rgb_image)[...,0:3]
        self.depth_image = ros_numpy.numpify(ros_depth_image).astype(np.float32)
        self.camera_info = ros_camera_info

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

    def capture_pointcloud(self):
        # FIXME: hardcoded for azure kinect
        # You don't want to subscribe to pointclouds longer than we need too; bandwidth intensive
        return rospy.wait_for_message(f'/camera_lower_right/points2', PointCloud2)

    def load_dataset_callback(self, request):
        dataset = DatasetReader(request.dataset_path).load()

        self.model.initialize_with_data(dataset)

        self.dataset_path = request.dataset_path

        return {"success": True, "message": f"loaded dataset {request.dataset_path}"}

    def act_on_bins_callback(self, request):
        rgb_image, depth_image = self.capture_image()
        actions = zip(request.bin_ids, request.asins, request.actions)
        result, message = self.model.act_on_bins(actions, rgb_image, depth_image, self.camera_info.K)
        self.dataset_writer.write(self.model.dataset)
        return result, message

    def capture_object_callback(self, request):
        if not request.bin_id or not request.object_id:
            return False, "bin_id and object_id are required"

        rgb_image, depth_image = self.capture_image()
        result, message, mask = self.model.capture_object(request.bin_id, request.object_id, rgb_image, depth_image, self.camera_info.K)
        self.dataset_writer.write(self.model.dataset)
        return result, message, mask

    def empty_pod_callback(self, request):
        rgb_image, depth_image = self.capture_image()
        prev_bin_bounds = self.model.dataset.metadata["bin_bounds"]
        new_dataset = Dataset()
        new_dataset.metadata["bin_bounds"] = prev_bin_bounds
        new_dataset.camera_configs = {self.camera_info.name: CameraConfig(intrinsics={"K": self.camera_info.K, "D": self.camera_info.D, "R": self.camera_info.R, "P": self.camera_info.P})}
        new_dataset.add([], {self.camera_info.name: CameraData(rgb_image=rgb_image, depth_image=depth_image)})
        self.model.initialize_with_data(new_dataset)
        # FIXME: this will fail if the user initialized with a specific dataset
        self.dataset_writer = DatasetAppender(self.dataset_path)
        self.dataset_writer.write(self.model.dataset)
        return {"success": True, "message": "empty bin captured"}

    def get_object_callback(self, request):
        if not request.object_id or not request.frame_id:
            return False, "object_id and frame_id are required", None, None

        self.play_camera_shutter()
        result, message = self.model.get_object(request.bin_id, request.object_id, None, self.depth_image.shape)
        self.dataset_writer.write(self.model.dataset)
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


        return True,\
               f"Points and mask successfully retrieved for object {request.object_id} in bin {bin_id}",\
               bin_id,\
               pointcloud,\
               mask_msg

    def pick_object_callback(self, request):
        if not request.bin_id or not request.object_id:
            return False, "bin_id and object_id are required"

        rgb_image, depth_image = self.capture_image()
        points_msg = self.capture_pointcloud()
        result, message = self.model.remove_object(request.bin_id, request.object_id, rgb_image, depth_image, self.camera_info, points_msg)
        self.dataset_writer.write(self.model.dataset)
        return result, message, None

    def update_object_callback(self, request):
        if not request.bin_id:
            return False, "bin_id is required"
        rgb_image, depth_image = self.capture_image()
        points_msg = self.capture_pointcloud()
        result, message = self.model.update(request.bin_id, rgb_image, depth_image, self.camera_info, points_msg)
        self.dataset_writer.write(self.model.dataset)
        return result, message, None

    def stow_object_callback(self, request):
        if not request.bin_id or not request.object_id:
            return False, "bin_id and object_id are required"
        rgb_image, depth_image = self.capture_image()
        points_msg = self.capture_pointcloud()
        result, message, mask = self.model.add_object(request.bin_id, request.object_id, rgb_image, depth_image, self.camera_info, points_msg)

        self.dataset_writer.write(self.model.dataset)
        return result, message, mask

    def reset_callback(self, request):
        if not request.bin_id:
            return False, "bin_id is required"

        rgb_image, depth_image = self.capture_image()
        result, message = self.model.reset(request.bin_id, rgb_image)
        self.dataset_writer.write(self.model.dataset)
        return result, message


class PodModel:

    def __init__(self, dataset, camera_name) -> None:
        self.dataset = dataset
        self.bin_bounds = self.dataset.metadata["bin_bounds"]
        self.bin_names = self.bin_bounds.keys()
        self.bins = {}
        self.H = None
        self.W = None
        self.camera_name = camera_name

    def initialize_with_data(self, dataset: Dataset):
        self.dataset = dataset
        bin_bounds = dataset.metadata["bin_bounds"]
        self.bin_bounds = bin_bounds
        self.bin_names = self.bin_bounds.keys()
        for bin in self.bin_names:
            self.bins[bin] = BinModel(dataset, bin, bounds=bin_bounds[bin])


        for entry in dataset.entries:

            camera_data = entry.camera_data[self.camera_name]

            rgb_image = camera_data.rgb_image
            depth_image = camera_data.depth_image
            points_msg = None
            camera_info = None #dataset.metadata[]

            for action in entry.actions:
                if action.action_type == "stow":
                    self.add_object(action.item.bin_id, action.item.asin_id, rgb_image, depth_image, camera_info, points_msg)
                else:
                    print("ERROR ACTION NOT SUPPORTED")


        self.H, self.W = self.dataset.entries[-1].camera_data[self.camera_name].depth_image.shape

    def get_masks(self):
       """ Retrieves the mask with bin product id obj_id from the full camera reference frame"""
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

    def capture_object(self, bin_id, asin, rgb_image, depth_image, camera_intrinsics):
        raise NotImplementedError()

    def act_on_bins(self, actions: Tuple[str, str, str], rgb_image, depth_image, camera_intrinsics):
        raise NotImplementedError()

    def add_object(self, bin_id, asin, rgb_image, depth_image, camera_intrinsics, points_msg):
        raise NotImplementedError()

    def get_object(self, bin_id, asin, points_msg, im_shape):
        raise NotImplementedError()

    def remove_object(self, bin_id, asin, rgb_image, depth_image, camera_intrinsics, points_msg):
        raise NotImplementedError()

    def update(self, bin_id, rgb_image, depth_image, camera_intrinsics, points_msg):
        raise NotImplementedError()

    def reset(self, bin_id, rgb_image):
        raise NotImplementedError()


class DummyPodModel(PodModel):
    def __init__(self, dataset, camera_name) -> None:
        super().__init__(dataset, camera_name)

    def capture_object(self, bin_id, asin, rgb_image, depth_image, camera_intrinsics):
        return True

    def act_on_bins(self, actions: Tuple[str, str, str], rgb_image, depth_image, camera_intrinsics):
        to_record = []
        for bin_id, asin, action_type in actions:
            to_record.append(Action(action_type, Item(asin, self.dataset.next_item_seq(bin_id, asin)), bin_id))
        self.dataset.add(to_record, {self.camera_name: CameraData(rgb_image=rgb_image, depth_image=depth_image)})
        return True, ""

    def add_object(self, bin_id, asin, rgb_image, depth_image, camera_intrinsics, points_msg):
        actions = [Action("stow", Item(asin, self.dataset.next_item_seq(bin_id, asin)), bin_id)]
        self.dataset.add(actions, {self.camera_name: CameraData(rgb_image=rgb_image, depth_image=depth_image)})
        return True, "", None

    def get_object(self, bin_id, asin, points_msg, im_shape):
        return True, ""

    def remove_object(self, bin_id, asin, rgb_image, depth_image, camera_intrinsics, points_msg):
        actions = [Action("pick", Item(asin, self.dataset.next_item_seq(bin_id, asin)), bin_id)]
        self.dataset.add(actions, {self.camera_name: CameraData(rgb_image=rgb_image, depth_image=depth_image)})
        return True, ""

    def update(self, bin_id, rgb_image, depth_image, camera_intrinsics, points_msg):
        return True, ""

    def reset(self, bin_id, rgb_image):
        return True, ""
