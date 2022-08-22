import subprocess
import encodings
from fileinput import filename
from unittest import result
import cv2
import matplotlib
from aurmr_perception.srv import CaptureObject, RemoveObject, GetObjectPoints, ResetBin
import numpy as np
import ros_numpy
import rospy
import message_filters
import tf2_ros
import matplotlib.pyplot as plt
from collections import defaultdict
from sensor_msgs.msg import Image, CameraInfo, PointField, PointCloud2
from std_msgs.msg import Header

from aurmr_unseen_object_clustering.tools.run_network import clustering_network

class PodPerceptionROS:
    def __init__(self, model, camera_name, visualize):
        self.visualize = visualize
        self.camera_name = camera_name
        self.model = model

        self.tf2_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer)

        self.trigger_capture = rospy.Service('~capture_object', CaptureObject, self.capture_object_callback)
        self.trigger_remove = rospy.Service('~remove_object', RemoveObject, self.remove_object_callback)
        self.trigger_retrieve = rospy.Service('~get_object_points', GetObjectPoints, self.get_object_callback)
        self.trigger_reset = rospy.Service('~reset_bin', ResetBin, self.reset_callback)

        self.camera_depth_subscriber = message_filters.Subscriber(f'/{self.camera_name}/aligned_depth_to_color/image_raw', Image)
        self.camera_rgb_subscriber = message_filters.Subscriber(f'/{self.camera_name}/color/image_raw', Image)
        self.camera_info_subscriber = message_filters.Subscriber(f'/{self.camera_name}/color/camera_info', CameraInfo)

        self.camera_synchronizer = message_filters.ApproximateTimeSynchronizer([
            self.camera_depth_subscriber, self.camera_rgb_subscriber, self.camera_info_subscriber], 10, 1)
        self.camera_synchronizer.registerCallback(self.camera_callback)
        self.camera_frame = f"{self.camera_name}_color_frame"
        self.received_images = False
        self.rgb_image = None
        self.depth_image = None
        self.camera_info = None


    def play_camera_shutter(self):
        audio_file = '/usr/share/sounds/Yaru/stereo/battery-low.oga'
        subprocess.call(['mplayer', audio_file])

    def capture_object_callback(self, request):
        if not request.bin_id or not request.object_id:
            return False, "bin_id and object_id are required"

        result, message, mask = self.model.capture_object(request.bin_id, request.object_id, self.rgb_image, self.depth_image, self.camera_info.K)

        if result and self.visualize:
            bin_im_viz = self.model.latest_captures[request.bin_id]
            bin_im_viz = cv2.cvtColor(bin_im_viz, cv2.COLOR_RGB2BGR)
            cv2.imshow('latest_bin_capture', bin_im_viz)
            cv2.waitKey(1)

            mask_im_viz = self.model.latest_masks[request.bin_id].astype(float)
            cv2.imshow('latest_mask', mask_im_viz)
            cv2.waitKey(1)

        self.play_camera_shutter()
        return result, message, mask

    def get_object_callback(self, request):
        if not request.object_id or not request.frame_id:
            return False, "object_id and frame_id are required", None, None

        result, message = self.model.get_object(None, request.object_id)

        if not result:
            return result, message, None, None

        bin_id, points, mask = result

        mask = ros_numpy.msgify(Image, mask.astype(np.uint8), encoding="mono8")

        if request.mask_only:
            return True,\
               f"Mask successfully retrieved for object {request.object_id} in bin {bin_id}",\
               bin_id,\
               None,\
               mask

        if request.frame_id != self.camera_frame:
            # Transform points to requested frame_id

            stamped_transform = self.tf2_buffer.lookup_transform(request.frame_id, self.camera_frame, rospy.Time(0),
                                                                 rospy.Duration(1))
            camera_to_target_mat = ros_numpy.numpify(stamped_transform.transform)

            points = np.vstack([points, np.ones(points.shape[1])])  # convert to homogenous
            points = np.matmul(camera_to_target_mat, points)[0:3, :].T  # apply transform

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
               mask

    def remove_object_callback(self, request):
        if not request.bin_id or not request.object_id:
            return False, "bin_id and object_id are required"

        result, message = self.model.remove_object(request.bin_id, request.object_id)

        return result, message

    def reset_callback(self, request):
        if not request.bin_id:
            return False, "bin_id is required"
    
        import os
        filename = '/tmp/empty_bin.png'
        if os.path.exists(filename):
            img = cv2.imread(filename)
            self.rgb_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            rospy.loginfo(f"Using cached empty pod image. Delete {filename} to regenerate the image.")
        elif self.rgb_image is not None:
            img = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(filename, img)
            
            rospy.loginfo("Caching Empty Pod Image for Later use")
            
        if self.rgb_image is None:
            return False, "No images have been streamed"

        result = self.model.reset(request.bin_id, self.rgb_image)

        return result, None

    def camera_callback(self, ros_depth_image, ros_rgb_image, ros_camera_info):
        self.received_images = True
        self.rgb_image = ros_numpy.numpify(ros_rgb_image)
        self.depth_image = ros_numpy.numpify(ros_depth_image)
        self.camera_info = ros_camera_info
        # rospy.loginfo("GOT IMAGES")
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
        self.object_bin_queues = defaultdict(ObjectBinQueue)
        self.bin_normals = {}
        self.segmentation_method = segmentation_method

    def capture_object(self, bin_id, object_id, rgb_image, depth_image, camera_intrinsics):
        if not bin_id or not object_id:
            return False

        if bin_id not in self.latest_captures:
            return False, f"Bin {bin_id} has not been reset"

        last_rgb = self.latest_captures[bin_id]
        current_rgb = rgb_image

        if(self.segmentation_method == "clustering"):
            net = clustering_network()
            camera_intrinsics_3x3 = np.reshape(camera_intrinsics, (3,3))
            mask = net.run_net(current_rgb, depth_image, camera_intrinsics_3x3)
            mask = mask*(255//np.max(mask))
            rospy.loginfo(np.unique(mask, return_counts=True))
            cv2.imwrite("/tmp/mask.bmp", mask)
            # plt.imshow(mask)
            # plt.show()
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
        # Apply mask to the depth image
        masked_depth = depth_image * mask

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

        if bin_id not in self.points_table:
            self.points_table[bin_id] = {}
            self.masks_table[bin_id] = {}

        self.points_table[bin_id][object_id] = np.vstack((x_ros,y_ros,z_ros))
        self.masks_table[bin_id][object_id] = mask
        self.object_bin_queues[object_id].put(bin_id)

        self.latest_captures[bin_id] = current_rgb
        self.latest_masks[bin_id] = mask

        return True, f"Object {object_id} in bin {bin_id} has been captured with {x.shape[0]} points.", ros_numpy.msgify(Image, mask.astype(np.uint8), encoding="mono8")

    def get_object(self, bin_id, object_id):
        if not object_id:
            return False, "object_id is required"

        if self.object_bin_queues[object_id].empty():
            return False, f"Object {object_id} was not found"

        if not bin_id:
            bin_id = self.object_bin_queues[object_id].get(remove=False)

        if bin_id not in self.points_table:
            return False, f"Bin {bin_id} was not found"

        if object_id not in self.points_table[bin_id]:
            return False, f"Object {object_id} was not found in bin {bin_id}"

        points = self.points_table[bin_id][object_id]
        mask = self.masks_table[bin_id][object_id]
        return (bin_id, points, mask), None

    def remove_object(self, bin_id, object_id, rgb_image):
        if not bin_id or not object_id:
            return False, "bin_id and object_id are required"

        if bin_id not in self.points_table:
            return False, f"Bin {bin_id} was not found"

        if object_id not in self.points_table[bin_id]:
            return False, f"Object {object_id} was not found in bin {bin_id}"

        del self.points_table[bin_id][object_id]
        self.object_bin_queues[object_id].remove(bin_id)
        self.latest_captures[bin_id] = rgb_image

        return True, None

    def reset(self, bin_id, rgb_image):
        if not bin_id:
            return False

        self.latest_captures[bin_id] = rgb_image

        return True


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
