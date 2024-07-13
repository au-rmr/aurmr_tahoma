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

from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import sys

from normal_std.inference_grasp_main import run_normal_std

from scipy.spatial.transform import Rotation as R
from copy import deepcopy
import geometry_msgs

from hardware_inference_comms import TrainRGBModel

from aurmr_perception.gqcnn_hardware_comms import TrainDexNetModel
from aurmr_perception.gqcnn_hardware_comms import SimpleGQCNN
from aurmr_perception.gqcnn_hardware_comms import CustomSingleDataset

import matplotlib.pyplot as plt

import pickle

import os, time

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class HeuristicGraspDetector:
    def __init__(self, grasp_offset, bin_normal):
        # print("grasp offset", grasp_offset)
        # print("bin normal", bin_normal)
        self.grasp_offset = grasp_offset
        self.bin_normal = np.array(bin_normal)
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

    def detect(self, points):
        np.save("/tmp/points", points)

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
    def __init__(self, detector, grasp_method):
        self.detector = detector
        if(grasp_method == "normal"):
            self.detect_grasps_normal = rospy.Service('~detect_grasps', DetectGraspPoses, self.detect_grasps_normal_cb)
        elif(grasp_method == "RGB_grasp"):
            self.detect_grasps_rgb = rospy.Service('~detect_grasps', DetectGraspPoses, self.detect_rgb_grasps_cb)
        elif(grasp_method == "dexnet"):
            self.detect_grasps_dexnet = rospy.Service('~detect_grasps', DetectGraspPoses, self.detect_dexnet_grasps_cb)
        elif(grasp_method == "centroid"):
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
        self.grasp_viz_perception = rospy.Publisher("~selected_grasp_pose_perception", geometry_msgs.msg.PoseStamped,
                                        queue_size=1, latch=True)

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

        with open('/tmp/calibration_pixel_coords_pod.pkl', 'rb') as f:
            self.bin_bounds = pickle.load(f)

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

    def transform_pose(self, pose_stamped, to_frame):
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
        # cv2.imwrite("/home/aurmr/workspaces/aurmr_demo_perception/src/segnetv2_mask2_former/Mask_Results/grasp_pre_mask.png", cv_image)
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
        print(request.points.header)
        self.visualize_grasps(stamped_detections)
        return True, "", stamped_detections

    def detect_grasps_normal_cb(self, request):
        cv_image = self.bridge.imgmsg_to_cv2(request.mask, desired_encoding='passthrough')

        # convert point cloud to mask
        fx, fy = 1940.1367, 1940.1958
        cx, cy = 2048.7397, 1551.3889
        width = 4096
        height = 3072

        cv_image_new = np.zeros([height, width])
        # cv2.imwrite("/home/aurmr/workspaces/aurmr_demo_perception/src/segnetv2_mask2_former/Mask_Results/zero_cv_image.png", cv_image_new)
        pts = ros_numpy.numpify(request.points)
        pts = np.stack([pts['x'],
                        pts['y'],
                        pts['z']], axis=1)
        points_sort_z = np.flip(pts[pts[:, 2].argsort()], axis=0)

        points_lowered = pts

        print("rgb shape in grasp", self.rgb_img.shape)
        pts = ros_numpy.numpify(request.points)
        self.points_viz_pub.publish(request.points)
        pts = np.stack([pts['x'],
                        pts['y'],
                        pts['z']], axis=1)
        detections = self.detector.detect(pts)

        # transform from base link to rgb camera link
        inference_object = run_normal_std()
        point, euler_angles = inference_object.inference(self.rgb_img, self.depth_img, cv_image)


        transform= self.tf_buffer.lookup_transform('base_link', 'pod_base_link', rospy.Time())
        transform_camera_to_base_link= self.tf_buffer.lookup_transform('base_link', 'rgb_camera_link', rospy.Time())
        POD_OFFSET = 0.1
        RGB_TO_DEPTH_FRAME_OFFSET = 0.015
        DEPTH_TILT = -transform.transform.translation.z-0.03
        # center[2] += center[0]*np.sin(DEPTH_TILT)
        # point[2] += DEPTH_TILT

        point[2] = transform.transform.translation.x - POD_OFFSET - transform_camera_to_base_link.transform.translation.x
        point[1] -= point[2]*np.sin(DEPTH_TILT)
        point[0] -= RGB_TO_DEPTH_FRAME_OFFSET
        print("bind id", request.bin_id)
        print("points", point)

        # clamp euler englaes within 15 and 45 but if less than 15 it will be 0
        clamped_euler_angles = [0., 0., 0.]
        if(euler_angles[0] < 0):
            clamped_euler_angles[0] = self.clamp(euler_angles[0], -30.*math.pi/180., -12.*math.pi/180.)
        else:
            clamped_euler_angles[0] = self.clamp(euler_angles[0], 12.*math.pi/180., 30.*math.pi/180.)
        if(euler_angles[1] < 0):
            clamped_euler_angles[1] = self.clamp(euler_angles[1], -30.*math.pi/180., -12.*math.pi/180.)
        else:
            clamped_euler_angles[1] = 0.0

        # Comment this if you want to use the normal of the object
        clamped_euler_angles = [0., 0., 0.]

        # clamped_euler_angles[1] = 0.0
        print("post euler angles", clamped_euler_angles[0]*180/math.pi, clamped_euler_angles[1]*180/math.pi)
        # transform from the 3d pose to rgb_camera_link
        r_cam = R.from_euler('xyz', [clamped_euler_angles[0], clamped_euler_angles[1], 0.], degrees=False)
        orientation_viz = r_cam.as_quat()

        as_quat_pose = Quaternion(x=orientation_viz[0], y=orientation_viz[1],
                                        z=orientation_viz[2], w=orientation_viz[3])
        as_pose_pose = Pose(position=Point(x=point[0], y=point[1], z=point[2]), orientation=as_quat_pose)
        t_header= deepcopy(request.points.header)
        t_header.frame_id = 'rgb_camera_link_offset'
        vis_pose = PoseStamped(header=t_header, pose=as_pose_pose)
        self.grasp_viz_perception.publish(vis_pose)

        # transform from base ink to 3d pose
        # position in base_line
        output_pose_stamped = self.transform_pose(vis_pose, 'base_link')

        r_prefix = R.from_euler('xyz', [math.pi/2., -math.pi/2., math.pi/2.], degrees=False)
        r3 = r_prefix * r_cam
        orientation = r3.as_quat()
        output_pose_stamped.pose.orientation = Quaternion(x=orientation[0], y=orientation[1],
                                        z=orientation[2], w=orientation[3])

        return True, "", [output_pose_stamped]


    def detect_rgb_grasps_cb(self, request):
        cv_image = self.bridge.imgmsg_to_cv2(request.mask, desired_encoding='passthrough')
        # cv2.imwrite("/home/aurmr/workspaces/aurmr_demo_perception/src/segnetv2_mask2_former/Mask_Results/grasp_pre_mask.png", cv_image)

        # convert point cloud to mask
        fx, fy = 1940.1367, 1940.1958
        cx, cy = 2048.7397, 1551.3889
        width = 4096
        height = 3072

        cv_image_new = np.zeros([height, width])
        # cv2.imwrite("/home/aurmr/workspaces/aurmr_demo_perception/src/segnetv2_mask2_former/Mask_Results/zero_cv_image.png", cv_image_new)
        pts = ros_numpy.numpify(request.points)
        pts = np.stack([pts['x'],
                        pts['y'],
                        pts['z']], axis=1)
        points_sort_z = np.flip(pts[pts[:, 2].argsort()], axis=0)

        points_lowered = pts

        print("rgb shape in grasp", self.rgb_img.shape)
        pts = ros_numpy.numpify(request.points)
        self.points_viz_pub.publish(request.points)
        pts = np.stack([pts['x'],
                        pts['y'],
                        pts['z']], axis=1)
        detections = self.detector.detect(pts)
        crop_bin_bounds = self.bin_bounds[request.bin_id]
        # assert np.array_equal(crop_bin_bounds, np.array([1097, 1407, 1752, 2149]))
        rgb_image_cropped = self.rgb_img[crop_bin_bounds[0]:crop_bin_bounds[1], crop_bin_bounds[2]:crop_bin_bounds[3], ...]
        segmask_cropped = cv_image[crop_bin_bounds[0]:crop_bin_bounds[1], crop_bin_bounds[2]:crop_bin_bounds[3], ...]
        depth_cropped = self.depth_img[crop_bin_bounds[0]:crop_bin_bounds[1], crop_bin_bounds[2]:crop_bin_bounds[3], ...]

        object_hardware_comm = TrainRGBModel(rgb_image_cropped, segmask_cropped)
        grasp_point, grasp_angle = object_hardware_comm.train_model()

        print(grasp_point, crop_bin_bounds)
        grasp_point[0] = grasp_point[0]*rgb_image_cropped.shape[1]/224
        grasp_point[1] = grasp_point[1]*rgb_image_cropped.shape[0]/224


        seconds = time.time()

        result = time.localtime(seconds)
        img_path = "/home/aurmr/workspaces/RGB_grasp_soofiyan_ws/src/Paper evaluation/Dataset"
        cv2.imwrite(os.path.join(img_path, f"rgb_img_{result.tm_mday}_{result.tm_hour}_{result.tm_min}_{result.tm_sec}.png"), rgb_image_cropped)
        cv2.imwrite(os.path.join(img_path, f"segmask_img_{result.tm_mday}_{result.tm_hour}_{result.tm_min}_{result.tm_sec}.png"), segmask_cropped)
        cv2.imwrite(os.path.join(img_path, f"depth_img_{result.tm_mday}_{result.tm_hour}_{result.tm_min}_{result.tm_sec}.png"), depth_cropped)

        fig, axs = plt.subplots(1, 1, figsize=(12, 3))
        axs.imshow(rgb_image_cropped)
        axs.add_patch(
            plt.Circle((grasp_point[0], grasp_point[1]), radius=2, color="red", fill=True)
        )
        plt.savefig(os.path.join(img_path, f"grasp_point_img_{result.tm_mday}_{result.tm_hour}_{result.tm_min}_{result.tm_sec}.png"))

        grasp_point[0] += crop_bin_bounds[2]
        grasp_point[1] += crop_bin_bounds[0]

        print(grasp_point, grasp_angle*180/np.pi, crop_bin_bounds)

        center3D_depth_value = self.depth_img[int(grasp_point[1]), int(grasp_point[0])]
        if center3D_depth_value < 0.01:
            print("Hole in depth image, setting depth to default 0.018")
            center3D_depth_value = 0.018
        pointz = center3D_depth_value/1000.0
        pointx = (grasp_point[0] - cx) * pointz / fx
        pointy = (grasp_point[1] - cy) * pointz / fy

        point = [pointx, pointy, pointz]

        transform= self.tf_buffer.lookup_transform('base_link', 'pod_base_link', rospy.Time())
        transform_camera_to_base_link= self.tf_buffer.lookup_transform('base_link', 'rgb_camera_link', rospy.Time())
        # less RGB_TO_DEPTH..... right
        # DEPTH_TILT more in - LOWER
        if(request.bin_id == "1H"):
            RGB_TO_DEPTH_FRAME_OFFSET = 0.007
            DEPTH_TILT = -transform.transform.translation.z - 0.01
        elif(request.bin_id == "2H"):
            RGB_TO_DEPTH_FRAME_OFFSET = 0.017
            DEPTH_TILT = -transform.transform.translation.z - 0.01
        elif(request.bin_id == "3H"):
            RGB_TO_DEPTH_FRAME_OFFSET = 0.022
            DEPTH_TILT = -transform.transform.translation.z - 0.01
        else:
            RGB_TO_DEPTH_FRAME_OFFSET = -0.005
            DEPTH_TILT = -transform.transform.translation.z - 0.01
        POD_OFFSET = 0.

        point[2] = transform.transform.translation.x - POD_OFFSET - transform_camera_to_base_link.transform.translation.x
        point[1] -= point[2]*np.sin(DEPTH_TILT)
        point[0] -= RGB_TO_DEPTH_FRAME_OFFSET

        clamped_euler_angles = [0., 0., 0.]

        clamped_euler_angles[0] = grasp_angle[0]
        clamped_euler_angles[1] = -grasp_angle[1]

        point[0] += 0.245*np.sin(clamped_euler_angles[0])
        point[1] += 0.245*np.sin(clamped_euler_angles[1])

        # clamped_euler_angles[1] = 0.0
        print("post euler angles", clamped_euler_angles[0]*180/math.pi, clamped_euler_angles[1]*180/math.pi)
        # transform from the 3d pose to rgb_camera_link
        r_cam = R.from_euler('xyz', [clamped_euler_angles[0], clamped_euler_angles[1], 0.], degrees=False)
        orientation_viz = r_cam.as_quat()

        as_quat_pose = Quaternion(x=orientation_viz[0], y=orientation_viz[1],
                                        z=orientation_viz[2], w=orientation_viz[3])
        as_pose_pose = Pose(position=Point(x=point[0], y=point[1], z=point[2]), orientation=as_quat_pose)
        t_header= deepcopy(request.points.header)
        t_header.frame_id = 'rgb_camera_link_offset'
        vis_pose = PoseStamped(header=t_header, pose=as_pose_pose)
        self.grasp_viz_perception.publish(vis_pose)

        output_pose_stamped = self.transform_pose(vis_pose, 'base_link')
        r_prefix = R.from_euler('xyz', [math.pi/2., -math.pi/2., math.pi/2.], degrees=False)
        r3 = r_prefix * r_cam
        orientation = r3.as_quat()
        output_pose_stamped.pose.orientation = Quaternion(x=orientation[0], y=orientation[1],
                                        z=orientation[2], w=orientation[3])
        self.visualize_grasps([output_pose_stamped])
        return True, "", [output_pose_stamped]

    def detect_dexnet_grasps_cb(self, request):
        cv_image = self.bridge.imgmsg_to_cv2(request.mask, desired_encoding='passthrough')
        # cv2.imwrite("/home/aurmr/workspaces/aurmr_demo_perception/src/segnetv2_mask2_former/Mask_Results/grasp_pre_mask.png", cv_image)

        # convert point cloud to mask
        fx, fy = 1940.1367, 1940.1958
        cx, cy = 2048.7397, 1551.3889
        width = 4096
        height = 3072

        cv_image_new = np.zeros([height, width])
        # cv2.imwrite("/home/aurmr/workspaces/aurmr_demo_perception/src/segnetv2_mask2_former/Mask_Results/zero_cv_image.png", cv_image_new)
        pts = ros_numpy.numpify(request.points)
        pts = np.stack([pts['x'],
                        pts['y'],
                        pts['z']], axis=1)
        points_sort_z = np.flip(pts[pts[:, 2].argsort()], axis=0)

        points_lowered = pts

        print("rgb shape in grasp", self.rgb_img.shape)
        pts = ros_numpy.numpify(request.points)
        self.points_viz_pub.publish(request.points)
        pts = np.stack([pts['x'],
                        pts['y'],
                        pts['z']], axis=1)
        detections = self.detector.detect(pts)

        crop_bin_bounds = self.bin_bounds[request.bin_id]
        rgb_image_cropped = self.rgb_img[crop_bin_bounds[0]:crop_bin_bounds[1], crop_bin_bounds[2]:crop_bin_bounds[3], ...]

        segmask_cropped = cv_image[crop_bin_bounds[0]:crop_bin_bounds[1], crop_bin_bounds[2]:crop_bin_bounds[3], ...]
        self.depth_img = self.depth_img.astype(np.float32)
        depth_cropped = self.depth_img[crop_bin_bounds[0]:crop_bin_bounds[1], crop_bin_bounds[2]:crop_bin_bounds[3], ...]

        object_hardware_comm = TrainDexNetModel(rgb_image_cropped, depth_cropped, segmask_cropped)
        grasp_point = object_hardware_comm.run_dexnet_hardware_comm_inference()

        print(grasp_point, crop_bin_bounds)
        grasp_point[1] = grasp_point[1]*rgb_image_cropped.shape[1]/224
        grasp_point[0] = grasp_point[0]*rgb_image_cropped.shape[0]/224

        fig, axs = plt.subplots(1, 1, figsize=(12, 3))
        axs.imshow(rgb_image_cropped)
        axs.add_patch(
            plt.Circle((grasp_point[0], grasp_point[1]), radius=2, color="red", fill=True)
        )
        plt.savefig("/tmp/grasp_point_plot_dexnet.png")

        grasp_point[0] += crop_bin_bounds[2]
        grasp_point[1] += crop_bin_bounds[0]

        print(grasp_point, crop_bin_bounds)

        center3D_depth_value = self.depth_img[int(grasp_point[1]), int(grasp_point[0])]

        pointz = center3D_depth_value/1000.0
        pointx = (grasp_point[0] - cx) * pointz / fx
        pointy = (grasp_point[1] - cy) * pointz / fy

        point = [pointx, pointy, pointz]

        transform= self.tf_buffer.lookup_transform('base_link', 'pod_base_link', rospy.Time())
        transform_camera_to_base_link= self.tf_buffer.lookup_transform('base_link', 'rgb_camera_link', rospy.Time())
        POD_OFFSET = 0.1
        RGB_TO_DEPTH_FRAME_OFFSET = 0.02
        DEPTH_TILT = -transform.transform.translation.z - 0.02

        point[2] = transform.transform.translation.x - POD_OFFSET - transform_camera_to_base_link.transform.translation.x
        point[1] -= point[2]*np.sin(DEPTH_TILT)
        point[0] -= RGB_TO_DEPTH_FRAME_OFFSET

        clamped_euler_angles = [0., 0., 0.]

        clamped_euler_angles[0] = 0.0
        clamped_euler_angles[1] = 0.0

        # point[0] += 0.265*np.sin(clamped_euler_angles[0])
        # point[1] += 0.265*np.sin(clamped_euler_angles[1])

        # clamped_euler_angles[1] = 0.0
        print("post euler angles", clamped_euler_angles[0]*180/math.pi, clamped_euler_angles[1]*180/math.pi)
        # transform from the 3d pose to rgb_camera_link
        r_cam = R.from_euler('xyz', [clamped_euler_angles[0], clamped_euler_angles[1], 0.], degrees=False)
        orientation_viz = r_cam.as_quat()

        as_quat_pose = Quaternion(x=orientation_viz[0], y=orientation_viz[1],
                                        z=orientation_viz[2], w=orientation_viz[3])
        as_pose_pose = Pose(position=Point(x=point[0], y=point[1], z=point[2]), orientation=as_quat_pose)
        t_header= deepcopy(request.points.header)
        t_header.frame_id = 'rgb_camera_link_offset'
        vis_pose = PoseStamped(header=t_header, pose=as_pose_pose)
        self.grasp_viz_perception.publish(vis_pose)

        output_pose_stamped = self.transform_pose(vis_pose, 'base_link')
        r_prefix = R.from_euler('xyz', [math.pi/2., -math.pi/2., math.pi/2.], degrees=False)
        r3 = r_prefix * r_cam
        orientation = r3.as_quat()
        output_pose_stamped.pose.orientation = Quaternion(x=orientation[0], y=orientation[1],
                                        z=orientation[2], w=orientation[3])

        return True, "", [output_pose_stamped]
