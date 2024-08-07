import threading
import logging

import itertools

import time
from typing import Dict
from typing import Any

import rospy

import numpy as np
import ros_numpy

import message_filters
from rospy_message_converter import message_converter

from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import PointCloud2

from cv_bridge import CvBridge

from dataclasses import dataclass
import image_geometry


@dataclass
class CameraData:
    """
    Camera images
    """

    rgb_image: np.ndarray = None
    depth_image: np.ndarray = None

    np_xyz_points: np.ndarray = None
    np_colors: np.ndarray = None

    def camera_data(self) -> Dict[str, bool]:
        return {
            "rgb": self.rgb_image is not None,
            "depth": self.depth_image is not None,
            "point_cloud": self.has_point_cloud(),
        }

    def has_images(self) -> bool:
        return (self.rgb_image is not None) and (self.depth_image is not None)

    def has_point_cloud(self) -> bool:
        return (self.np_xyz_points is not None) and (self.np_colors is not None)

    @property
    def point_cloud(self):
        # ..todo:: convert data type
        dtype = np.dtype({'names': ['x', 'y', 'z', 'r', 'g', 'b'],
                'formats': [np.dtype('float32'), np.dtype('float32'), np.dtype('float32'), np.dtype('uint8'), np.dtype('uint8'), np.dtype('uint8')]})
        return np.concatenate([self.np_xyz_points, self.np_colors], axis=2, dtype=dtype)



logger = logging.getLogger(__name__)


class Camera:
    def __init__(self, camera_name: str, config: Dict[str, Any]):
        self.cv = threading.Condition()
        self.camera_name = camera_name
        self.config = {"rgb": True, "depth": True, "point_cloud": True}
        self.config["name"] = camera_name
        self.config.update(config)
        self.new_camera_data = None
        self.request_stamp = None
        self.cv_bridge = CvBridge()
        self.metadata = {}

    def __str__(self):
        return f"{self.camera_name}"

    def is_complete(self):
        for k in ["rgb", "depth", "point_cloud"]:
            camera_data = self.new_camera_data.camera_data()
            if self.config[k] and not camera_data[k]:
                return False
        return True

    def get_camera_data(self):
        with self.cv:
            self.cv.wait_for(lambda: self.is_complete() or rospy.is_shutdown())
            self.request_stamp = None
            camera_data, self.camera_data = self.new_camera_data, None
            metadata, self.metadata = self.metadata, {}
            return camera_data, metadata

    def request_capture(self) -> None:
        with self.cv:
            # Images queue up, so we use this capture request timestamp
            # to ensure the images we pull are at least _after_ the request was made
            self.request_stamp = time.time()
            self.new_camera_data = CameraData()

    def get_camera_info(self, timeout: int = 10) -> Dict[str, Any]:

        all_topics = itertools.chain(*rospy.get_published_topics())
        all_topics = set(all_topics)

        def _read(camera_name: str) -> CameraInfo:
            topic_name = f"{camera_name}/camera_info"
            logging.debug("Getting info for %s", topic_name)
            if not topic_name in all_topics:
                logger.warning("Topic does not exist: %s", topic_name)
                return None
            info = rospy.wait_for_message(
                f"{camera_name}/camera_info", CameraInfo, timeout=timeout
            )
            return message_converter.convert_ros_message_to_dictionary(info)

        camera_info = {}
        if self.config["rgb"]:
            camera_name = self.config.get("rgb_name", f"/{self.camera_name}/rgb")
            camera_info["rgb"] = _read(camera_name)

        if self.config["depth"]:
            camera_name = self.config.get("depth_name", f"/{self.camera_name}/depth")
            camera_info["depth"] = _read(camera_name)

        if self.config["rgb"] and self.config["depth"]:
            camera_name = self.config.get(
                "rgb_to_depth_name", f"/{self.camera_name}/rgb_to_depth"
            )
            camera_info["rgb_to_depth"] = _read(camera_name)

        print(camera_info)
        return camera_info


    def subscribe_to_ros_service(self) -> None:
        if self.config["rgb"] and self.config["depth"]:
            topic_name = self.config.get(
                "depth_to_rgb_name", f"/{self.camera_name}/depth_to_rgb/image_raw"
            )
            camera_depth_subscriber = message_filters.Subscriber(f"{topic_name}", Image)
            topic_name = self.config.get(
                "rgb_name", f"/{self.camera_name}/rgb/image_raw"
            )
            camera_rgb_subscriber = message_filters.Subscriber(f"{topic_name}", Image)
            camera_synchronizer = message_filters.ApproximateTimeSynchronizer(
                [camera_depth_subscriber, camera_rgb_subscriber], 10, slop=0.5
            )
            camera_synchronizer.registerCallback(self.rgb_and_depth_callback)
        elif self.config["rgb"]:
            topic_name = self.config.get(
                "rgb_name", f"/{self.camera_name}/rgb/image_raw"
            )
            camera_subscriber = message_filters.Subscriber(f"{topic_name}", Image)
            camera_subscriber.registerCallback(self.rgb_callback)
        elif self.config["depth"]:
            topic_name = self.config.get(
                "depth_name", f"/{self.camera_name}/depth/image_raw"
            )
            camera_subscriber = message_filters.Subscriber(f"{topic_name}", Image)
            camera_subscriber.registerCallback(self.depth_callback)

        logger.debug("Subscribing to point cloud topic")

        if self.config["point_cloud"]:
            topic_name = self.config.get(
                "point_cloud_name", f"/{self.camera_name}/points2"
            )
            pointcloud_subscriber = rospy.Subscriber(
                f"{topic_name}", PointCloud2, self.pointcloud_callback
            )

    def pointcloud_callback(self, pc_msg: PointCloud2):
        if not self.new_camera_data or not self.request_stamp:
            return
        elif self.new_camera_data.has_point_cloud():
            return
        elif self.request_stamp > pc_msg.header.stamp.to_time():
            return

        pc = ros_numpy.numpify(pc_msg)
        points = np.zeros((pc.shape[0], 3))
        points[:, 0] = pc["x"]
        points[:, 1] = pc["y"]
        points[:, 2] = pc["z"]

        rgb = pc["rgb"]
        rgb.dtype = np.uint32
        colors = np.zeros((pc.shape[0], 3))
        colors[:, 0] = np.right_shift(np.bitwise_and(rgb, 0x00FF0000), 16)
        colors[:, 1] = np.right_shift(np.bitwise_and(rgb, 0x0000FF00), 8)
        colors[:, 2] = np.bitwise_and(rgb, 0x000000FF)
        colors = colors / 255

        with self.cv:
            self.new_camera_data.np_xyz_points = points
            self.new_camera_data.np_colors = colors.astype(np.float64)

            self.metadata["point_cloud"] = (
                message_converter.convert_ros_message_to_dictionary(pc_msg.header)
            )

            self.cv.notify()

    def depth_callback(self, depth_msg: Image):
        if not self.new_camera_data:
            return
        elif self.new_camera_data.camera_data["depth"]:
            return
        elif self.request_stamp > depth_msg.header.stamp.to_time():
            return

        depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, "passthrough")

        with self.cv:
            self.new_camera_data.depth_image = depth
            self.metadata["depth_image"] = (
                message_converter.convert_ros_message_to_dictionary(depth_msg.header)
            )
            self.metadata["depth_image_width"] = depth.shape[1]
            self.metadata["depth_image_height"] = depth.shape[0]
            self.metadata["depth_timestamp"] = depth_msg.header.stamp.to_time()

            self.cv.notify()

    def rgb_callback(self, rgb_msg: Image):
        if not self.new_camera_data or not self.request_stamp:
            return
        elif self.new_camera_data.camera_data()["rgb"]:
            return
        elif self.request_stamp > rgb_msg.header.stamp.to_time():
            return
        rgb = self.cv_bridge.imgmsg_to_cv2(rgb_msg, "rgb8")

        with self.cv:
            self.new_camera_data.rgb_image = rgb

            self.metadata["color_image"] = (
                message_converter.convert_ros_message_to_dictionary(rgb_msg.header)
            )
            self.metadata["rgb_image_width"] = rgb.shape[1]
            self.metadata["rgb_image_height"] = rgb.shape[0]
            self.metadata["rgb_timestamp"] = rgb_msg.header.stamp.to_time()

            self.cv.notify()

    def rgb_and_depth_callback(self, depth_msg: Image, rgb_msg: Image):
        if not self.new_camera_data or not self.request_stamp:
            return
        elif self.new_camera_data.has_images():
            return
        elif self.request_stamp > depth_msg.header.stamp.to_time() or self.request_stamp > rgb_msg.header.stamp.to_time():
            return

        rgb = self.cv_bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
        depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, "passthrough")

        with self.cv:
            self.new_camera_data.rgb_image = rgb
            self.new_camera_data.depth_image = depth

            self.metadata["depth_image"] = (
                message_converter.convert_ros_message_to_dictionary(depth_msg.header)
            )
            self.metadata["color_image"] = (
                message_converter.convert_ros_message_to_dictionary(rgb_msg.header)
            )
            self.metadata["rgb_image_width"] = rgb.shape[1]
            self.metadata["rgb_image_height"] = rgb.shape[0]
            self.metadata["depth_image_width"] = depth.shape[1]
            self.metadata["depth_image_height"] = depth.shape[0]
            self.metadata["depth_timestamp"] = depth_msg.header.stamp.to_time()
            self.metadata["rgb_timestamp"] = rgb_msg.header.stamp.to_time()

            self.cv.notify()
