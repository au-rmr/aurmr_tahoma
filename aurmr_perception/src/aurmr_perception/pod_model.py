import queue
import subprocess
import encodings
from fileinput import filename
from unittest import result
import cv2
import matplotlib
from aurmr_perception.srv import CaptureObject, CaptureTable, RemoveObject, GetObjectPoints, GetObjectMasks, ResetBin, LoadDataset


from skimage.color import label2rgb

import numpy as np
import ros_numpy
import rospy
import message_filters
import tf2_ros
import matplotlib.pyplot as plt
from collections import defaultdict
from sensor_msgs.msg import Image, CameraInfo, PointField, PointCloud2
from demo.segnetv2_demo import SegnetV2
from std_msgs.msg import Header
from std_srvs.srv import Trigger
# from aurmr_unseen_object_clustering.tools.run_network import clustering_network
# from aurmr_unseen_object_clustering.tools.match_masks import match_masks
from aurmr_unseen_object_clustering.tools.segmentation_net import SegNet, NO_OBJ_STORED, UNDERSEGMENTATION, OBJ_NOT_FOUND, MATCH_FAILED, IN_BAD_BINS, def_config
from copy import deepcopy


NO_OBJ_STORED = 1
UNDERSEGMENTATION = 2
OBJ_NOT_FOUND = 3
MATCH_FAILED = 4
IN_BAD_BINS = 5

class PodPerceptionROS:
    def __init__(self, model, camera_name, visualize, camera_type, gripper_camera, gripper_camera_name):
        self.visualize = visualize
        self.camera_name = camera_name
        self.gripper_camera_name = gripper_camera_name
        self.model = model

        self.tf2_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer)

        self.trigger_capture = rospy.Service('~capture_object', CaptureObject, self.capture_object_callback)
        self.trigger_capture_table = rospy.Service('~capture_table', CaptureTable, self.capture_table_callback)
        self.trigger_empty_pod = rospy.Service('~capture_empty_pod', Trigger, self.empty_pod_callback)
        self.trigger_pick = rospy.Service('~pick_object', CaptureObject, self.pick_object_callback)
        self.trigger_stow = rospy.Service('~stow_object', CaptureObject, self.stow_object_callback)
        self.trigger_update = rospy.Service('~update_bin', CaptureObject, self.update_object_callback)
        self.trigger_retrieve = rospy.Service('~get_object_points', GetObjectPoints, self.get_object_callback)
        self.trigger_retrieve_table = rospy.Service('~get_masks_table', GetObjectMasks, self.get_mask_table_callback)
        self.trigger_reset = rospy.Service('~reset_bin', ResetBin, self.reset_callback)
        self.trigger_load = rospy.Service('~load_dataset', LoadDataset, self.load_dataset)


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

        if gripper_camera:
            self.gripper_camera_rgb_subscriber = message_filters.Subscriber(f'/{self.gripper_camera_name}/image_raw', Image)
            self.gripper_camera_info_subscriber = message_filters.Subscriber(f'/{self.gripper_camera_name}/camera_info', CameraInfo)

        self.points_sub = rospy.Subscriber(f'/{self.camera_name}/points2', PointCloud2, self.points_cb)
        self.masks_pub = rospy.Publisher('~detected_masks', Image, queue_size=1, latch=True)
        self.labels_pub = rospy.Publisher('~labeled_images', Image, queue_size=1, latch=True)
        self.color_image_pub = rospy.Publisher('~colored_images', Image, queue_size=1, latch=True)
        self.camera_synchronizer = message_filters.ApproximateTimeSynchronizer([
            self.camera_depth_subscriber, self.camera_rgb_subscriber, self.camera_info_subscriber], 10, 1)
        self.camera_synchronizer.registerCallback(self.camera_callback)

        if gripper_camera:
            self.gripper_camera_synchronizer = message_filters.ApproximateTimeSynchronizer([
                self.gripper_camera_rgb_subscriber, self.gripper_camera_info_subscriber], 10, 1)
            self.gripper_camera_synchronizer.registerCallback(self.gripper_camera_callback)
        self.received_images = False
        self.rgb_image = None
        self.depth_image = None
        self.camera_info = None
        self.points_msg = None
        if gripper_camera:
            self.gripper_received_images = False
            self.gripper_rgb_image = None
            self.gripper_camera_info = None



    def play_camera_shutter(self):
        audio_file = '/usr/share/sounds/freedesktop/stereo/camera-shutter.oga'
        try:
            subprocess.call(['mplayer', audio_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except:
            pass

    def load_dataset(self, request):
        from aurmr_dataset.io import DatasetReader

        bin_mapping = {
            "P-9-H051H030": "1H",
            "P-6-H835J238": "1G",
            "P-8-H758H650": "1F",
            "P-8-H588H170": "1E",
            "P-9-H051H031": "2H",
            "P-6-H835J239": "2G",
            "P-8-H758H651": "2F",
            "P-8-H588H171": "2E",
            "P-9-H051H032": "3H",
            "P-6-H835J240": "3G",
            "P-8-H758H652": "3F",
            "P-8-H588H172": "3E",
            "P-9-H051H033": "4H",
            "P-6-H835J241": "4G",
            "P-8-H758H653": "4F",
            "P-8-H588H173": "4E",
            "P-8-H588H494" : "1E",
            "P-8-H588H495" : "2E",
            "P-8-H588H496" : "3E",
            "P-8-H588H497" : "4E",
            "P-8-H758H974" : "1F",
            "P-8-H758H975" : "2F",
            "P-8-H758H976" : "3F",
            "P-8-H758H977" : "4F",
            "P-8-H888H454" : "1G",
            "P-8-H888H455" : "2G",
            "P-8-H888H456" : "3G",
            "P-8-H888H457" : "4G",
            "P-9-H051H354" : "1H",
            "P-9-H051H355" : "2H",
            "P-9-H051H356" : "3H",
            "P-9-H051H357" : "4H",
            "P-9-M223R307": "1D",
            "P-9-M223R831": "1E",
            "P-9-M503R832": "2E",
            "P-9-M095R784": "2C"

        }
        dataset = DatasetReader(request.dataset_path).load()

        camera_name = f"{self.camera_name}_depth"

        K  = np.asarray(dataset.camera_info[camera_name]['K'] ).reshape((3, 3))

        first_entry = dataset.entries[0]
        first_depth = first_entry.camera_data[self.camera_name].depth_image
        updated_config = deepcopy(def_config)
        updated_config["bounds"] = dataset.metadata["bin_bounds"]
        first_depth = first_depth.astype(np.float32)


        self.net = SegNet(config=updated_config, init_depth=first_depth, init_info=K)
        # (henrifung)NOTE: PERHAPS OTHER THINGS NEED TO BE CLEARED
        self.model.reset_model()
        self.model.net = self.net

        for entry in dataset.entries:

            camera_data = entry.camera_data[self.camera_name]

            rgb_image = camera_data.rgb_image[:,:,::-1]
            depth_image = camera_data.depth_image
            points_msg = ros_numpy.point_cloud2.array_to_pointcloud2(camera_data.np_xyz_points)
            camera_info = dataset.camera_info


            depth_image = depth_image.astype(np.float32)

            for action in entry.actions:
                # Hack so we can keep changes isolated from rest of file
                class MockCameraInfo():
                    def __init__(self) -> None:
                        self.K = K
                if action.action_type == "stow":
                    simple_bin_id = bin_mapping.get(action.item.bin_id, None)
                    if simple_bin_id is None:
                        print("Can't simplify bin id ", action.item.bin_id)
                        continue
                    print(action.item.bin_id, simple_bin_id)
                    self.model.add_object(simple_bin_id, action.item.asin_id, rgb_image, depth_image, MockCameraInfo(), points_msg)
                else:
                    print("ERROR ACTION NOT SUPPORTED")


        return {"success": True, "message": f"loaded dataset {request.dataset_path} with {len(dataset.inventory())} objects in inventory"}


    def capture_object_callback(self, request):
        if not request.bin_id or not request.object_id:
            return False, "bin_id and object_id are required"

        rospy.loginfo(request.bin_id)
        result, message, mask = self.model.capture_object(request.bin_id, request.object_id, self.rgb_image, self.depth_image, self.camera_info.K)

        if result and self.visualize:
            bin_im_viz = self.model.latest_captures[request.bin_id]
            bin_im_viz = cv2.cvtColor(bin_im_viz, cv2.COLOR_RGB2BGR)
            # cv2.imshow('latest_bin_capture', bin_im_viz)
            # cv2.waitKey(1)

            mask_im_viz = self.model.latest_masks[request.bin_id].astype(float)
            # cv2.imshow('latest_mask', mask_im_viz)
            # cv2.waitKey(1)
        self.play_camera_shutter()
        return result, message, mask

    def capture_table_callback(self, request):
        result, message, masks = self.model.capture_table_object(self.gripper_rgb_image.astype(np.uint8))
        self.play_camera_shutter()
        return result, message, masks

    def points_cb(self, request):
        if self.points_msg is None:
            rospy.loginfo("Points Found")
        self.points_msg = request

    def empty_pod_callback(self, request):
        import os
        # filename = '/tmp/empty_bin.bmp'
        # if os.path.exists(filename):
        #     self.depth_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        #     rospy.loginfo(f"Using cached empty pod image. Delete {filename} to regenerate the image.")
        # elif self.depth_image is not None:
        #     cv2.imwrite(filename, self.depth_image)

        #     rospy.loginfo("Caching Empty Pod Image for Later use")

        # self.net = SegNet(init_depth=self.model.numpify_pointcloud(self.points_msg, self.rgb_image.shape))

        intrinsics_3x3 = np.reshape(self.camera_info.K, (3,3))
        print(intrinsics_3x3)
        print(type(intrinsics_3x3))
        self.net = SegNet(init_depth=self.depth_image, init_info=intrinsics_3x3)
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
            print(points.shape)
            points = np.matmul(camera_to_target_mat, points)[0:3, :].T  # apply transform
        else:
            points = points.T
        print(points.shape)
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

    def get_mask_table_callback(self, request):
        return True, "Table Mask Retrieved", self.model.get_masks_table()

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
        result, message, mask = self.model.add_object(request.bin_id, request.object_id, self.rgb_image, self.depth_image, self.camera_info, self.points_msg)
        self.play_camera_shutter()
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

    def gripper_camera_callback(self, ros_rgb_image, ros_camera_info):
        k = np.reshape(ros_camera_info.K, (3,3))
        r = np.reshape(ros_camera_info.R, (3,3))
        d = ros_camera_info.D
        p = np.reshape(ros_camera_info.P, (3,4))



        self.gripper_received_images = True
        self.gripper_rgb_image = ros_numpy.numpify(ros_rgb_image).astype(np.float32)

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
        # h,  w = self.depth_image .shape[:2]
        # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(k, d, (w,h), 1, (w,h))
        # self.depth_image = cv2.undistort(self.depth_image, k, d).astype(np.uint8)
        # # crop the image
        # x, y, w, h = roi
        # self.depth_image = self.depth_image[y:y+h, x:x+w]
        # # img_shape = ros_numpy.numpify(ros_rgb_image).shape[0:2]
        self.gripper_camera_info = ros_camera_info
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
        self.points_table = {}
        self.masks_table = {}
        self.occluded_table = {}
        self.object_bin_queues = defaultdict(ObjectBinQueue)
        self.bin_normals = {}
        self.segmentation_method = segmentation_method
        self.net = None
        self.table_net = SegnetV2()
        self.table_mask = None
        self.mask_pub = rospy.Publisher("~masks", Image, queue_size=5)

    def numpify_pointcloud(self, points, im_shape):
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

    def mask_pointcloud(self, points, mask):
        # Points may or may not be (points x 1), so we flatten to be sure
        points = ros_numpy.numpify(points).flatten()
        np.save("/tmp/points1.npy", points)
        points_seg = points[mask.flatten() > 0]
        points_seg = np.vstack((points_seg['x'],points_seg['y'],points_seg['z']))
        points_seg = points_seg[:, np.invert(np.isnan(points_seg[2, :]))]
        return points_seg

    def backproject(self, depth_image, mask, camera_intrinsics):
        points = rospy.wait_for_message(f'/camera_lower_right/points2', PointCloud2)
        points = ros_numpy.numpify(points)
        np.save("/tmp/points1.npy", points)
        points_seg = points[mask.flatten() > 0]
        points_seg = np.vstack((points_seg['x'],points_seg['y'],points_seg['z']))
        points_seg = points_seg[:, np.invert(np.isnan(points_seg[0, :]))]
        return points_seg
        #[INFO] [1660938607.362794]: {'names': ['x', 'y', 'z', 'rgb'], 'formats': ['<f4', '<f4', '<f4', '<f4'], 'offsets': [0, 4, 8, 16], 'itemsize': 32}
        #[INFO] [1660938607.364375]: (12582912,)
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

    def capture_table_object(self, rgb_image):
        if rgb_image is None:
            return False, f"Table RGB image is None"
        full_masks_ret = np.zeros_like(rgb_image[:, :, 0])
        bin_crop = [100,250,35,500]
        rgb_image_croppped = rgb_image[bin_crop[0]:bin_crop[1], bin_crop[2]:bin_crop[3], :]
        masks, full_masks = self.table_net.mask_generator(rgb_image_croppped)
        full_masks_ret[bin_crop[0]:bin_crop[1], bin_crop[2]:bin_crop[3]] = full_masks
        if not masks:
            return True, f"No masks were found", None
        self.table_mask = full_masks_ret
        # viz = self.table_net.vis_masks(rgb_image, full_masks_ret)
        # plt.imsave('viz.png', viz)
        # plt.show()
        return True, f"Table has been segmented successfully.", ros_numpy.msgify(Image, full_masks_ret, encoding="mono8")

    def capture_object(self, bin_id, object_id, rgb_image, depth_image, camera_intrinsics):
        if not bin_id or not object_id:
            return False

        if bin_id not in self.latest_captures:
            return False, f"Bin {bin_id} has not been reset"


        last_rgb = self.latest_captures[bin_id]
        current_rgb = rgb_image

        # Assigns a mask to the current image, stored in mask
        if(self.segmentation_method == "clustering"):
            pass
        elif(self.segmentation_method == "pixel_difference"):
            # Get the difference of the two captured RGB images
            difference = cv2.absdiff(last_rgb, current_rgb)

            # Threshold the difference image to get the initial mask
            mask = difference.sum(axis=2) >= self.diff_threshold

            # Group the masked pixels and leave only the group with the largest area
            (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8, cv2.CV_32S)
            areas = np.array([stats[i, cv2.CC_STAT_AREA] for i in range(len(stats))])
            max_area = np.max(areas[1:])
            max_area_idx = np.where(areas == max_area)[0][0]
            mask[np.where(labels != max_area_idx)] = 0.0
        else:
            raise RuntimeError(f"Unknown segmentation method requested: {self.segmentation_method}")
        points = rospy.wait_for_message(f'/camera_lower_right/points2', PointCloud2)
        points = self.mask_pointcloud(points, mask)

        if bin_id not in self.points_table:
            self.points_table[bin_id] = {}
            self.masks_table[bin_id] = {}

        self.points_table[bin_id][object_id] = points
        self.masks_table[bin_id][object_id] = mask
        self.object_bin_queues[object_id].put(bin_id)

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

        rospy.loginfo(bin_id + str(type(bin_id)))
        # plt.imshow(rgb_image.astype(np.uint8))
        # plt.show()
        # status = self.net.stow(bin_id, object_id, rgb_image, self.numpify_pointcloud(points_msg, rgb_image.shape))
        intrinsics_3x3 = np.reshape(camera_intrinsics.K, (3,3))
        status = self.net.stow(bin_id, object_id, rgb_image, depth_raw=depth_image, info=intrinsics_3x3)
        if status == NO_OBJ_STORED:
            if bin_id not in self.occluded_table:
                self.occluded_table[bin_id] = {}
            self.occluded_table[bin_id][object_id] = True
            msg = f"Object {object_id} in bin {bin_id} was not detected, it is either occluded or was never placed."
            mask_ret = None
            success = False
            return success, msg, mask_ret
        mask = self.net.get_masks()
        self.mask_pub.publish(ros_numpy.msgify(Image, mask.astype(np.uint8), encoding="mono8"))
        cv2.imwrite(f"/tmp/{bin_id}_{object_id}.png", mask)
        if status == 1:
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


        if bin_id not in self.points_table:
            self.points_table[bin_id] = {}
            self.masks_table[bin_id] = {}
        if bin_id not in self.net.bad_bins:
            for o_id in self.points_table[bin_id].keys():
                mask = self.net.get_obj_mask(o_id)
                self.masks_table[bin_id][o_id] = mask
                self.points_table[bin_id][o_id] = self.mask_pointcloud(points_msg, mask)

            mask = self.net.get_obj_mask(object_id)
            self.masks_table[bin_id][object_id] = mask
            self.points_table[bin_id][object_id] = self.mask_pointcloud(points_msg, mask)
            num_points = self.points_table[bin_id][object_id].shape[0]
            mask_ret = ros_numpy.msgify(Image, mask.astype(np.uint8), encoding="mono8")
            msg = f"Object {object_id} in bin {bin_id} has been stowed with {num_points} points."
            success = True
        else:
            self.masks_table[bin_id][object_id] = None
            self.points_table[bin_id][object_id] = None
            msg = f"Object {object_id} in bin {bin_id} could not be SIFT matched. Bin state is lost."
            mask_ret = None
            success = False
        self.object_bin_queues[object_id].put(bin_id)
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

        if self.object_bin_queues[object_id].empty():
            return False, f"Object {object_id} was not found"

        if bin_id not in self.points_table:
            return False, f"Bin {bin_id} was not found"

        if object_id not in self.points_table[bin_id]:
            return False, f"Object {object_id} was not found in bin {bin_id}"


        if bin_id in self.net.bad_bins:
            msg = f"Object {object_id} in bin {bin_id} could not be segmented, and was added to bad bins already during stowing/picking"
            return False, msg
            mask = self.net.get_obj_mask(object_id)
            self.points_table[bin_id][object_id] = self.mask_pointcloud(points_msg, mask)
        else:
            mask = self.masks_table[bin_id][object_id]
            msg = f"Object {object_id} in bin {bin_id} was found"

        points = self.points_table[bin_id][object_id]
        # plt.imshow(mask)
        # plt.title(f"get_object {object_id} in {bin_id}")
        # plt.show()
        return (bin_id, points, mask), msg

    def get_masks_table(self):
        return ros_numpy.msgify(Image, self.table_mask.astype(np.uint8), encoding="mono8")

    def remove_object(self, bin_id, object_id, rgb_image, depth_image, camera_intrinsics, points_msg):
        if not bin_id or not object_id:
            return False, "bin_id and object_id are required"

        if bin_id not in self.points_table:
            return False, f"Bin {bin_id} was not found"

        if object_id not in self.points_table[bin_id]:
            return False, f"Object {object_id} was not found in bin {bin_id}"

        del self.points_table[bin_id][object_id]
        self.object_bin_queues[object_id].remove(bin_id)
        self.latest_captures[bin_id] = rgb_image

        intrinsics_3x3 = np.reshape(camera_intrinsics.K, (3,3))
        # self.net.pick(object_id, rgb_image, self.numpify_pointcloud(points_msg, rgb_image.shape))
        self.net.pick(object_id, rgb_image, depth_raw=depth_image, info=intrinsics_3x3)
        if bin_id not in self.net.bad_bins:
            for o_id in self.points_table[bin_id].keys():
                mask = self.net.get_obj_mask(o_id)
                self.masks_table[bin_id][o_id] = mask
                self.points_table[bin_id][o_id] = self.mask_pointcloud(points_msg, mask)

        return True, f"Object {object_id} in bin {bin_id} has been removed"

    def update(self, bin_id, rgb_image, depth_image, camera_intrinsics, points_msg):
        print(bin_id)
        print(self.points_table.keys())
        if not bin_id :
            rospy.logwarn("bin id not given")
            return False, "bin_id is required"

        if bin_id not in self.points_table:
            rospy.logwarn("no bin id")
            return False, f"Bin {bin_id} was not found"

        self.latest_captures[bin_id] = rgb_image

        print("CALLBACK2")
        intrinsics_3x3 = np.reshape(camera_intrinsics.K, (3,3))
        # self.net.update(bin_id, rgb_image, self.numpify_pointcloud(points_msg, rgb_image.shape))
        res = self.net.update(bin_id, rgb_image, depth_raw=depth_image, info=intrinsics_3x3)
        if res == UNDERSEGMENTATION:
            return False, f"Segmentation in {bin_id} undersegmented. Causes not limited to model failure, potential result of occlusion, object fell out during manip"
        print("DONE UPDATING")
        if bin_id not in self.net.bad_bins:
            for o_id in self.points_table[bin_id].keys():
                mask = self.net.get_obj_mask(o_id)
                # plt.imshow(mask)
                # plt.title("Mask in update")
                # plt.show()
                self.masks_table[bin_id][o_id] = mask
                self.points_table[bin_id][o_id] = self.mask_pointcloud(points_msg, mask)

        return True, f"{bin_id} was updated"

    def reset(self, bin_id, rgb_image):
        if not bin_id:
            return False

        self.latest_captures[bin_id] = rgb_image

        return True

    def reset_model(self):
        self.latest_captures = {}
        self.latest_masks = {}
        self.points_table = {}
        self.masks_table = {}
        self.occluded_table = {}
        self.object_bin_queues = defaultdict(ObjectBinQueue)
        self.bin_normals = {}
        self.net = None
        self.table_net = SegnetV2()
        self.table_mask = None
        self.mask_pub = rospy.Publisher("~masks", Image, queue_size=5)

class ObjectBinQueue:
    def __init__(self):
        self.queue = []

    def put(self, bin_id):
        self.queue.append(bin_id)

    def get(self, remove=True):
        result = self.queue[-1]
        if remove:
            self.queue = self.queue[:-1]
        return result

    def remove(self, bin_id):
        self.queue.reverse()
        self.queue.remove(bin_id)
        self.queue.reverse()

    def empty(self):
        return len(self.queue) == 0