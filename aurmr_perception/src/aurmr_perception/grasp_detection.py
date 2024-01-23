import math

import numpy as np
import ros_numpy
import rospy
import tf2_ros
import cv2
from std_msgs.msg import ColorRGBA
from aurmr_perception.srv import DetectGraspPoses
from geometry_msgs.msg import PoseStamped, Quaternion, Pose, Point, Vector3
from tf_conversions import transformations
from aurmr_perception.visualization import create_gripper_pose_markers
from sensor_msgs.msg import PointCloud2

# See moveit_grasps for inspiration on improved grasp ranking and filtering:
# https://ros-planning.github.io/moveit_tutorials/doc/moveit_grasps/moveit_grasps_tutorial.html
from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import sys

from scipy.spatial.transform import Rotation as R
from copy import deepcopy
import geometry_msgs

class HeuristicGraspDetector:
    def __init__(self, grasp_offset, bin_normal):
        # print("grasp offset", grasp_offset)
        # print("bin normal", bin_normal)
        self.grasp_offset = grasp_offset
        self.bin_normal = np.array(bin_normal)
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

    def detect(self, points):
        """
        Hacked together for the first grasp
        :param points:
        :return:
        """
        # compute the average point of the pointcloud

        points_sort_z = np.flip(points[points[:, 2].argsort()], axis=0)
        POINTS_TO_KEEP_FACTOR = .8
        keep_idxs = np.arange(int(points.shape[0]*(1-POINTS_TO_KEEP_FACTOR)), points.shape[0])
        points_lowered = points_sort_z[keep_idxs]
        rospy.loginfo(points.shape)
        rospy.loginfo(points_lowered.shape)
        center = np.mean(points_lowered, axis=0)

        # stamped_transform = self.tf2_buffer.lookup_transform("base_link", "rgb_camera_link", rospy.Time(0),
        #                                                 rospy.Duration(1))
        # camera_to_target_mat = ros_numpy.numpify(stamped_transform.transform)
        # center = np.vstack([center, np.ones(center.shape[1])])  # convert to homogenous
        # points = np.matmul(camera_to_target_mat, points)[0:3, :].T  # apply transform

        # center[0] = center[0] - 0.02
        # POD_OFFSET = -0.1

        transform= self.tf_buffer.lookup_transform('base_link', 'pod_base_link', rospy.Time())
        POD_OFFSET = 0.1
        RGB_TO_DEPTH_FRAME_OFFSET = -0.015
        DEPTH_TILT = -transform.transform.translation.z-0.03
        # center[2] += center[0]*np.sin(DEPTH_TILT)
        center[2] += DEPTH_TILT
        center[1] -= RGB_TO_DEPTH_FRAME_OFFSET
        center[0] = transform.transform.translation.x - POD_OFFSET

        # NOTE(nickswalker,4-29-22): Hack to compensate for the chunk of points that don't get observed
        # due to the lip of the bin
        #center[2] -= 0.02
        print(center, points.shape, center.shape)
        align_to_bin_orientation = transformations.quaternion_from_euler(math.pi / 2., -math.pi / 2., math.pi / 2.)

        poses_stamped = [(center, align_to_bin_orientation)]
        print(poses_stamped)
        return poses_stamped


class GraspDetectionROS:
    def __init__(self, detector):
        self.detector = detector
        self.detect_grasps = rospy.Service('~detect_grasps', DetectGraspPoses, self.detect_grasps_cb)

        self.dections_viz_pub = rospy.Publisher("~detected_grasps", MarkerArray, latch=True, queue_size=1)
        self.points_viz_pub = rospy.Publisher("~detected_pts", PointCloud2, latch=True, queue_size=5)
        self.camera_depth_subscriber = rospy.Subscriber('/camera_lower_right/depth_to_rgb/image_raw', Image, self.depth_callback)
        self.camera_rgb_subscriber = rospy.Subscriber('/camera_lower_right/rgb/image_raw', Image, self.rgb_callback)
        self.depth_img = []
        self.rgb_img = []
        self.grasp_offset = 0.125
        self.bin_normal = np.array([-1,0,0])
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        self.bridge = CvBridge()
        self.grasp_viz_perception = rospy.Publisher("selected_grasp_pose_perception", geometry_msgs.msg.PoseStamped,
                                        queue_size=1, latch=True)

    def depth_callback(self, data):
        try:
            self.depth_img = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
            # self.depth_img = np.array(self.depth_img, dtype=np.float32)
            # self.depth_img = self.depth_img.astype(np.float32)
        except CvBridgeError as e:
            print(e)

    def rgb_callback(self, data):
        try:
            self.rgb_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def visualize_grasps(self, poses_stamped):
        color = ColorRGBA(r = 1, g = 0, b = 0, a = 1)
        scale=Vector3(x=.05, y=.05, z=.05)
        # print(poses_stamped)
        markers = Marker(header = poses_stamped[0].header, pose=poses_stamped[0].pose, type=1, scale=scale, color=color)
        self.dections_viz_pub.publish(MarkerArray(markers=[markers]))


    def get_quaternion_from_euler(self, roll, pitch, yaw):
        """
        Convert an Euler angle to a quaternion.

        Input
            :param roll: The roll (rotation around x-axis) angle in radians.
            :param pitch: The pitch (rotation around y-axis) angle in radians.
            :param yaw: The yaw (rotation around z-axis) angle in radians.
        mask2former_frame
        Output
            :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
        """
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return [qx, qy, qz, qw]

    def transfor_pose(self, pose_stamped, to_frame):
        try:
            output_pose_stemped = self.tf_buffer.transform(pose_stamped, to_frame, rospy.Duration(1))
            return output_pose_stemped
        except(tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            raise

    def clamp(self, num, min_value, max_value):
        clamp_value = num
        if(num < 0):
            if(num <= min_value):
                clamp_value = min_value
            elif(num >= max_value):
                clamp_value = 0.
        else:
            if(num <= min_value):
                clamp_value = 0.
            elif(num >= max_value):
                clamp_value = max_value
        # clamp_value = max(min(num, max_value), min_value)
        # if(clamp_value <= min_value):
        #     clamp_value = 0
        return clamp_value

    def detect_grasps_cb(self, request):
        cv_image = self.bridge.imgmsg_to_cv2(request.mask, desired_encoding='passthrough')
        pts = ros_numpy.numpify(request.points)
        self.points_viz_pub.publish(request.points)
        pts = np.stack([pts['x'],
                        pts['y'],
                        pts['z']], axis=1)
        detections = self.detector.detect(pts)
        stamped_detections = []
        for position, orientation in detections:
            as_quat = Quaternion(x=orientation[0], y=orientation[1],
                                            z=orientation[2], w=orientation[3])
            as_pose = Pose(position=Point(x=position[0], y=position[1], z=position[2]), orientation=as_quat)
            stamped_detections.append(PoseStamped(header=request.points.header, pose=as_pose))

        self.visualize_grasps(stamped_detections)
        return True, "", stamped_detections
