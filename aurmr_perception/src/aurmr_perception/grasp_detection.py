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

#from normal_std.inference_grasp_main import run_normal_std

from scipy.spatial.transform import Rotation as R
from copy import deepcopy
import geometry_msgs

#from hardware_inference_comms import TrainRGBModel

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
        self.grasp_viz_perception = rospy.Publisher("selected_grasp_pose_perception", geometry_msgs.msg.PoseStamped,
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

    def mean_nonzero_around_grasp(self, depth_img, grasp_point, radius=5):
        x, y = int(grasp_point[0]), int(grasp_point[1])

        x_min = max(x - radius, 0)
        x_max = min(x + radius + 1, depth_img.shape[1])
        y_min = max(y - radius, 0)
        y_max = min(y + radius + 1, depth_img.shape[0])

        region = depth_img[y_min:y_max, x_min:x_max]
        non_zero_values = region[region > 0]

        return np.mean(non_zero_values) if non_zero_values.size > 0 else 0.0

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
        output_pose_stamped = self.transfor_pose(vis_pose, 'base_link')

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

        # fixed depth for grasp point
        point[2] = transform.transform.translation.x - transform_camera_to_base_link.transform.translation.x

        clamped_euler_angles = [0., 0., 0.]

        clamped_euler_angles[0] = grasp_angle[0]
        clamped_euler_angles[1] = -grasp_angle[1]

        # pre_grasp_x_dist = 0.1
        # point[0] += pre_grasp_x_dist*np.sin(clamped_euler_angles[0])
        # point[1] += pre_grasp_x_dist*np.sin(clamped_euler_angles[1])

        # clamped_euler_angles[1] = 0.0
        print("post euler angles", clamped_euler_angles[0]*180/math.pi, clamped_euler_angles[1]*180/math.pi)
        # transform from the 3d pose to rgb_camera_link
        r_cam = R.from_euler('xyz', [clamped_euler_angles[0], clamped_euler_angles[1], 0.], degrees=False)
        orientation_viz = r_cam.as_quat()

        as_quat_pose = Quaternion(x=orientation_viz[0], y=orientation_viz[1],
                                        z=orientation_viz[2], w=orientation_viz[3])
        as_pose_pose = Pose(position=Point(x=point[0], y=point[1], z=point[2]), orientation=as_quat_pose)
        t_header= deepcopy(request.points.header)
        t_header.frame_id = 'rgb_camera_link_offset' # rgb_camera_link
        vis_pose = PoseStamped(header=t_header, pose=as_pose_pose)
        self.grasp_viz_perception.publish(vis_pose)

        output_pose_stamped = self.transfor_pose(vis_pose, 'base_link')
        r_prefix = R.from_euler('xyz', [math.pi/2., -math.pi/2., math.pi/2.], degrees=False)
        r3 = r_prefix * r_cam
        orientation = r3.as_quat()
        output_pose_stamped.pose.orientation = Quaternion(x=orientation[0], y=orientation[1],
                                        z=orientation[2], w=orientation[3])
        return True, "", [output_pose_stamped]



    def estimate_depth(self, image):

        #from aurmr_setup.utils.workspace_utils import get_active_workspace_path
        #workpace_path = get_active_workspace_path()
        #model_path = f"{workpace_path}/model_weights/depth_anything_vitl14.pth"


        if True or "save  image":
            cv2.imwrite('/tmp/rgb_input.png', image)


        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0


        import torch
        from depth_anything.dpt import DepthAnything
        from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

        import torch.nn.functional as F
        from torchvision.transforms import Compose

        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        depth_anything = DepthAnything.from_pretrained("LiheYoung/depth_anything_vitl14").to(DEVICE).eval()

        transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        h, w = image.shape[:2]

        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)


        with torch.no_grad():
            depth = depth_anything(image)

        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

        depth = depth.cpu().numpy().astype(np.uint8)

        if True or "save depth image":
            depth_colorized = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
            cv2.imwrite('/tmp/depth_estimated.png', depth_colorized)

        return depth

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

        if True or "save mask":
            mask_scaled = segmask_cropped * 128
            mask_colorized_colorized = cv2.applyColorMap(mask_scaled, cv2.COLORMAP_INFERNO)
            cv2.imwrite('/tmp/colorized_mask.png', mask_colorized_colorized)

        estimate_depth = True
        if estimate_depth:
            #depth = self.estimate_depth(self.rgb_img)

            depth = self.estimate_depth(rgb_image_cropped)

            print(depth.shape)
            print(depth.dtype)
            print(self.depth_img.dtype)
            print(self.depth_img.shape)

            print(depth.min())
            print(depth.max())
            print(self.depth_img.min())
            print(self.depth_img.max())

            depth = depth.astype(np.float32)

            print(depth.min())
            print(depth.max())

            min_bin_distance = 1100
            max_bin_distance = 1370

            depth = (1.0 - (depth / 255)) * (max_bin_distance - min_bin_distance) + min_bin_distance
            depth_cropped = depth


            if True or "save depth image":
                depth_colorized = cv2.applyColorMap(depth.astype(np.uint8), cv2.COLORMAP_INFERNO)
                cv2.imwrite('/tmp/depth_estimated.png', depth_colorized)

            print(depth_cropped.min())
            print(depth_cropped.max())

            #depth = depth.astype(np.float32)
            #depth_cropped = depth[crop_bin_bounds[0]:crop_bin_bounds[1], crop_bin_bounds[2]:crop_bin_bounds[3], ...]

        self.depth_img = self.depth_img.astype(np.float32)
        depth_cropped = self.depth_img[crop_bin_bounds[0]:crop_bin_bounds[1], crop_bin_bounds[2]:crop_bin_bounds[3], ...]

        if True or "save depth image":
            depth_colorized = cv2.applyColorMap(depth_cropped.astype(np.uint8), cv2.COLORMAP_INFERNO)
            cv2.imwrite('/tmp/depth_actual.png', depth_colorized)

        print(depth_cropped.min())
        print(depth_cropped.max())

        object_hardware_comm = TrainDexNetModel(rgb_image_cropped, depth_cropped, segmask_cropped)
        grasp_points = object_hardware_comm.run_dexnet_hardware_comm_inference()

        grasp_points[:, 1] = grasp_points[:, 1] * rgb_image_cropped.shape[1]
        grasp_points[:, 0] = grasp_points[:, 0] * rgb_image_cropped.shape[0]

        grasp_point = grasp_points[0]
        print(grasp_point, crop_bin_bounds)

        fig, axs = plt.subplots(1, 1, figsize=(12, 3))
        axs.imshow(rgb_image_cropped)
        axs.add_patch(
            plt.Circle((grasp_point[1], grasp_point[0]), radius=2, color="red", fill=True)

        )
        plt.savefig("/tmp/grasp_point_plot_dexnet.png")

        if True:


            fig, axs = plt.subplots(1, 2, figsize=(12, 3))

            for ax in axs:
                ax.axis("off")

            masked_rgb_image = rgb_image_cropped[:, :, ::-1]
            masked_rgb_image[:, :, 0] = masked_rgb_image[:, :, 0] * 0.7 * segmask_cropped + masked_rgb_image[:, :, 0] * 0.3
            masked_rgb_image[:, :, 1] = masked_rgb_image[:, :, 1] * 0.7 * segmask_cropped + masked_rgb_image[:, :, 1] * 0.3
            masked_rgb_image[:, :, 2] = masked_rgb_image[:, :, 2] * 0.7 * segmask_cropped + masked_rgb_image[:, :, 2] * 0.3
            axs[0].imshow(masked_rgb_image)

            for grasp_point in grasp_points[:1]:
                axs[0].add_patch(plt.Circle((grasp_point[1], grasp_point[0]), radius=2, color="red", fill=False, alpha=1.0))


            for grasp_point in grasp_points:
                axs[0].add_patch(plt.Circle((grasp_point[1], grasp_point[0]), radius=2, color="orange", fill=False, alpha=0.5))

            axs[1].imshow(segmask_cropped * 128, cmap="gray")
            plt.tight_layout()
            fig.savefig(f"/tmp/grasp_point.png")
            #plt.show()


        grasp_point[0] += crop_bin_bounds[2]
        grasp_point[1] += crop_bin_bounds[0]

        print(grasp_point, crop_bin_bounds)

        center3D_depth_value = self.depth_img[int(grasp_point[0]), int(grasp_point[1])]

        pointz = center3D_depth_value/1000.0

        # ..FIXME this is with respect to a custom coordinate system.
        pointx = (grasp_point[0] - cx) * pointz / fx
        pointy = (grasp_point[1] - cy) * pointz / fy

        point = [pointx, pointy, pointz]



        transform= self.tf_buffer.lookup_transform('base_link', 'pod_base_link', rospy.Time())
        transform_camera_to_base_link= self.tf_buffer.lookup_transform('base_link', 'rgb_camera_link', rospy.Time())
        POD_OFFSET = 0.1
        RGB_TO_DEPTH_FRAME_OFFSET = 0.02
        DEPTH_TILT = -transform.transform.translation.z - 0.02

        if False and "remove magic values":
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

        output_pose_stamped = self.transfor_pose(vis_pose, 'base_link')
        r_prefix = R.from_euler('xyz', [math.pi/2., -math.pi/2., math.pi/2.], degrees=False)
        r3 = r_prefix * r_cam
        orientation = r3.as_quat()
        output_pose_stamped.pose.orientation = Quaternion(x=orientation[0], y=orientation[1],
                                        z=orientation[2], w=orientation[3])

        return True, "", [output_pose_stamped]
