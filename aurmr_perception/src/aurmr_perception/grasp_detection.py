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
sys.path.append("/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/segnetv2_mask2_former/UIE_main/mask2former_frame/")
from normal_std.inference_grasp_main import run_normal_std

from scipy.spatial.transform import Rotation as R
from copy import deepcopy
import geometry_msgs
from grasp_training.vit_model.inference_script import inference

import matplotlib.pyplot as plt
from gqcnn.gqcnn_examples.policy_for_training import dexnet3
from autolab_core import (YamlConfig, Logger, BinaryImage,
                          CameraIntrinsics, ColorImage, DepthImage, RgbdImage)

from std_msgs.msg import Int16
import json
from datetime import datetime


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
    
        temp = center
        center[0] = temp[2]
        center[1] = temp[0]
        center[2] = temp[1]
        # stamped_transform = self.tf2_buffer.lookup_transform("base_link", "rgb_camera_link", rospy.Time(0),
        #                                                 rospy.Duration(1))
        # camera_to_target_mat = ros_numpy.numpify(stamped_transform.transform)
        # center = np.vstack([center, np.ones(center.shape[1])])  # convert to homogenous
        # points = np.matmul(camera_to_target_mat, points)[0:3, :].T  # apply transform

        # center[0] = center[0] - 0.02
        POD_OFFSET = -0.1
        transform= self.tf_buffer.lookup_transform('base_link', 'pod_base_link', rospy.Time())
        center[0] = transform.transform.translation.x- POD_OFFSET
        # NOTE(nickswalker,4-29-22): Hack to compensate for the chunk of points that don't get observed
        # due to the lip of the bin
        #center[2] -= 0.02
        print(points.shape, center.shape)
        print("center", center)
        position = self.grasp_offset * self.bin_normal + center
        align_to_bin_orientation = transformations.quaternion_from_euler(math.pi / 2., -math.pi / 2., math.pi / 2.)

        poses_stamped = [(position, align_to_bin_orientation)]
        print(poses_stamped)
        return poses_stamped


class GraspDetectionROS:
    def __init__(self, detector, grasp_method):
        self.detector = detector
        if(grasp_method == "normal"):
            self.detect_grasps_normal = rospy.Service('~detect_grasps', DetectGraspPoses, self.detect_grasps__normal_cb)
        elif(grasp_method == "centroid"):
            self.detect_grasps = rospy.Service('~detect_grasps', DetectGraspPoses, self.detect_grasps_cb)
        
        self.dections_viz_pub = rospy.Publisher("~detected_grasps", MarkerArray, latch=True, queue_size=1)
        self.points_viz_pub = rospy.Publisher("~detected_pts", PointCloud2, latch=True, queue_size=5)
        self.camera_depth_subscriber = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        self.camera_rgb_subscriber = rospy.Subscriber('/camera/color/image_raw', Image, self.rgb_callback)
        self.data_id_subscriber = rospy.Subscriber('/data_id_topic', Int16, self.data_id_callback)
        self.target_object_id_subscriber = rospy.Subscriber('/target_object_id', Int16, self.target_object_id_callback)
        self.depth_img = []
        self.rgb_img = []
        self.grasp_offset = 0.125
        self.bin_normal = np.array([-1,0,0])
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        self.bridge = CvBridge()
        self.grasp_viz_perception = rospy.Publisher("selected_grasp_pose_perception", geometry_msgs.msg.PoseStamped,
                                        queue_size=1, latch=True)
        self.time_series_count = 0

    def data_id_callback(self, data):
        try:
            self.data_id = data.data
        except Exception as e:
            print(e)
    
    def target_object_id_callback(self, data):
        try:
            self.target_id = data.data
        except Exception as e:
            print(e)

    def depth_callback(self, data):
        try:
            self.depth_img = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
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
        
        # cv2.imwrite("/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/segnetv2_mask2_former/Mask_Results/grasp_pre_mask.png", cv_image)
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

    def detect_grasps__normal_cb(self, request):
        cv_image = self.bridge.imgmsg_to_cv2(request.mask, desired_encoding='passthrough')
        mask_all = self.bridge.imgmsg_to_cv2(request.mask_all, desired_encoding='passthrough')
        print("object id", request.object_id)
        cv2.imwrite("/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/segnetv2_mask2_former/Mask_Results/mask_all.png", mask_all)
        
        # np.save(f"/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/real_world_data/depth_image_{self.data_id}.npy", self.depth_img)
        # convert point cloud to mask
        fx, fy = 903.14697265625, 903.5240478515625
        cx, cy = 640.839599609375, 355.6268310546875
        width = 1280
        height = 720

        cv_image_new = np.zeros([height, width])
        # cv2.imwrite("/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/segnetv2_mask2_former/Mask_Results/zero_cv_image.png", cv_image_new)
        pts = ros_numpy.numpify(request.points)
        pts = np.stack([pts['x'],
                        pts['y'],
                        pts['z']], axis=1)
    

        print("rgb shape in grasp", self.rgb_img.shape)
        pts = ros_numpy.numpify(request.points)
        self.points_viz_pub.publish(request.points)
        pts = np.stack([pts['x'],
                        pts['y'],
                        pts['z']], axis=1)

        # transform from base link to rgb camera link

        method = "dexnet"
        if(method == "vit"):
            print(self.depth_img.shape, mask_all.shape)

            # np.save(f"/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/real_world_data/depth_image_{self.data_id}.npy", self.depth_img)
            # np.save(f"/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/real_world_data/rgb_image_{self.data_id}.npy", self.rgb_img)
            # np.save(f"/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/real_world_data/segmask_{self.data_id}.npy", cv_image)

            depth_image = cv2.resize(self.depth_img, (640, 360),  interpolation = cv2.INTER_NEAREST)
            segmask_image = cv2.resize(mask_all, (640, 360),  interpolation = cv2.INTER_NEAREST)
            depth_image = depth_image/1000.0
            print(np.unique(segmask_image))

            # cv2.imshow("depth", depth_image)
            # cv2.waitKey(0)
            # cv2.imshow("segmask", segmask_image*75)
            # cv2.waitKey(0)

            inference_object = inference()
            grasp_point_model, centroid_model, output = inference_object.run_model(depth_image, segmask_image, int(request.object_id))

            grasp_point = grasp_point_model + np.array([centroid_model[0],centroid_model[1]])
            print(grasp_point)
            grasp_point[0] = grasp_point[0]*2
            grasp_point[1] = grasp_point[1]*2
            inference_object = run_normal_std()
            
            grasp_3d_point, grasp_euler_angles, center = inference_object.inference(self.rgb_img, self.depth_img, cv_image, grasp_point)

            # json_save = {
            #     "grasp point": center.tolist(),
            #     "grasp_3d_point": grasp_3d_point,
            #     "grasp_angle": grasp_euler_angles,
            #     "target object id": self.target_id,
            # }
            # print("saved this json", json_save)
            # new_dir_name = str(self.data_id)
            # data_path = "/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/real_world_data/"
            # save_dir_json = data_path+"/perception_json_data_"+new_dir_name+"_"+datetime_string+".json"
            # with open(save_dir_json, 'w') as json_file:
            #     json.dump(json_save, json_file)

            transform_from_kinect_to_realsense = np.array([0.112, 0.165, 0.49])
            point = grasp_3d_point + transform_from_kinect_to_realsense
            euler_angles = grasp_euler_angles
            print("points", point, euler_angles, transform_from_kinect_to_realsense)
        elif(method == "rgb_vit"):
            datetime_string = datetime.now().isoformat().replace(":","")[:-7]
            print(self.depth_img.shape, cv_image.shape)

            np.save(f"/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/real_world_data/depth_image_{self.data_id}.npy", self.depth_img)
            np.save(f"/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/real_world_data/rgb_image_{self.data_id}.npy", self.rgb_img)
            np.save(f"/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/real_world_data/single_object_segmask_{self.data_id}.npy", cv_image)

            depth_image = cv2.resize(self.depth_img, (640, 360),  interpolation = cv2.INTER_NEAREST)
            rgb_image = cv2.resize(self.rgb_img, (640, 360),  interpolation = cv2.INTER_NEAREST)
            segmask_image = cv2.resize(cv_image, (640, 360),  interpolation = cv2.INTER_NEAREST)
            depth_image = depth_image/1000.0
            print(np.unique(segmask_image))

            inference_object = inference()
            # grasp_point_model, centroid_model, output = inference_object.run_model(depth_image, segmask_image, 1)
            grasp_point_model, centroid_model, output = inference_object.run_model_rgb(rgb_image, depth_image, segmask_image, 1)

            grasp_point = grasp_point_model + np.array([centroid_model[0],centroid_model[1]])
            print(grasp_point)
            grasp_point[0] = grasp_point[0]*2
            grasp_point[1] = grasp_point[1]*2

            inference_object = run_normal_std()
            
            grasp_3d_point, grasp_euler_angles, center = inference_object.inference(self.rgb_img, self.depth_img, cv_image, grasp_point)

            json_save = {
                "grasp point": center.tolist(),
                "grasp_3d_point": grasp_3d_point,
                "grasp_angle": grasp_euler_angles,
                "target object id": self.target_id,
            }
            print("saved this json", json_save)
            new_dir_name = str(self.data_id)
            data_path = "/home/aurmr/workspaces/uois_dynamo_grasp_rgb/src/real_world_data/"
            save_dir_json = data_path+"/perception_json_data_"+new_dir_name+"_"+datetime_string+".json"
            with open(save_dir_json, 'w') as json_file:
                json.dump(json_save, json_file)

            transform_from_kinect_to_realsense = np.array([0.16, 0.17, 0.39])
            point = grasp_3d_point + transform_from_kinect_to_realsense
            euler_angles = grasp_euler_angles
            print("points", point, euler_angles)

        elif(method == "dexnet"):
            print(self.depth_img.shape, cv_image.shape)
            depth_img = cv2.resize(self.depth_img, (640, 360),  interpolation = cv2.INTER_NEAREST)
            segmask = cv2.resize(cv_image, (640, 360),  interpolation = cv2.INTER_NEAREST)
            depth_img = depth_img.astype(np.float32)

            cx = depth_img.shape[1]/2 + 0.839599609375/2
            cy = depth_img.shape[0]/2 - (10-5.6268310546875)/2
            fx = 903.14697265625/2
            fy = 903.5240478515625/2
            camera_intrinsics_back_cam = CameraIntrinsics(frame="realsense", fx=fx, fy=fy,
                                                           cx=cx, cy=cy, skew=0.0,
                                                           height=360, width=640)

            dexnet_object = dexnet3(camera_intrinsics_back_cam)
            dexnet_object.load_dexnet_model()

            segmask_numpy_temp = np.zeros_like(segmask).astype(np.uint8)
            segmask_numpy_temp[segmask == 1] = 1
            depth_img_temp = depth_img*segmask_numpy_temp/1000
            depth_img_temp[depth_img_temp == 0] = 0.76
            depth_img_dexnet = DepthImage(depth_img_temp, frame=camera_intrinsics_back_cam.frame)
            # plt.imshow(depth_img_temp)
            # plt.show()

            segmask_numpy_temp[segmask == 1] = 255
            segmask_dexnet = BinaryImage(
                segmask_numpy_temp, frame=camera_intrinsics_back_cam.frame)

            # plt.imshow(segmask_numpy_temp)
            # plt.show()

            action, grasps_and_predictions, unsorted_grasps_and_predictions = dexnet_object.inference(depth_img_dexnet, segmask_dexnet, None)

            grasp_point = np.array([grasps_and_predictions[0][0].center.x*2, grasps_and_predictions[0][0].center.y*2])
            inference_object = run_normal_std()
            point, euler_angles, center = inference_object.inference(self.rgb_img, self.depth_img, cv_image, grasp_point)
            point += np.array([0.107, 0.17, 0.49])
            print("points", point)
        
        elif(method == "centroid"):
            inference_object = run_normal_std()
            point, euler_angles, center = inference_object.inference(self.rgb_img, self.depth_img, cv_image, None)
            point += np.array([0.107, 0.17, 0.49])
            print("points", point)


        # clamp euler englaes within 15 and 45 but if less than 15 it will be 0
        clamped_euler_angles = [0., 0., 0.]
        if(euler_angles[0] < 0):
            clamped_euler_angles[0] = self.clamp(euler_angles[0], -30.*math.pi/180., 0.*math.pi/180.)
        else:
            clamped_euler_angles[0] = self.clamp(euler_angles[0], 0.*math.pi/180., 30.*math.pi/180.)
        if(euler_angles[1] < 0):
            clamped_euler_angles[1] = self.clamp(euler_angles[1], -30.*math.pi/180., 0.*math.pi/180.)
        else:
            clamped_euler_angles[1] = self.clamp(euler_angles[1], 0.*math.pi/180., 30.*math.pi/180.)

        if(method == "centroid"):
            clamped_euler_angles[0] = 0.0
            clamped_euler_angles[1] = 0.0
        
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

        # orientation of the end effector with respect to base link and multiplying two different quartenions
        # orientation in base_line
        r_prefix = R.from_euler('xyz', [math.pi/2., -math.pi/2., math.pi/2.], degrees=False)
        r3 = r_prefix * r_cam
        orientation = r3.as_quat()
        output_pose_stamped.pose.orientation = Quaternion(x=orientation[0], y=orientation[1],
                                        z=orientation[2], w=orientation[3])

        return True, "", [output_pose_stamped]

        # 0 - yaw 1 - pitch
        # best
        # orientation = self.get_quaternion_from_euler(-math.pi/2.+euler_angles[1], 0*math.pi/180., -math.pi/2.+euler_angles[0])

        # orientation = self.get_quaternion_from_euler(math.pi/2., -math.pi/2.-euler_angles[1], math.pi/2.+euler_angles[0])
        # orientation = self.get_quaternion_from_euler(math.pi/2., -math.pi/2.-euler_angles[1], math.pi/2.+euler_angles[0])
        # position = self.grasp_offset * self.bin_normal + points
        # position = points
        '''print("custom",position)
        print("orientation", orientation)
        stamped_detections = []
        count = 0
        for pos, orient in detections:
            if(count == 1):
                break
            as_quat = Quaternion(x=orientation[0], y=orientation[1],
                                        z=orientation[2], w=orientation[3])
            as_pose = Pose(position=Point(x=position[0], y=position[1], z=position[2]), orientation=as_quat)
            print("orient", as_quat)
            print("Position ", position[0], position[1], position[2])
            count = 1
        stamped_detections.append(PoseStamped(header=request.points.header, pose=as_pose))
        self.visualize_grasps(stamped_detections)
        return True, "", stamped_detections'''