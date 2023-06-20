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
sys.path.append("/home/aurmr/workspaces/uois_soofiyan_ws/src/segnetv2_mask2_former/UIE_main/mask2former_frame/")
from normal_std.inference_grasp_main import run_normal_std

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
        # cv2.imwrite("/home/aurmr/workspaces/uois_soofiyan_ws/src/segnetv2_mask2_former/Mask_Results/grasp_pre_mask.png", cv_image)
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
        # cv2.imwrite("/home/aurmr/workspaces/uois_soofiyan_ws/src/segnetv2_mask2_former/Mask_Results/grasp_pre_mask.png", cv_image)
        
        # convert point cloud to mask
        fx, fy = 1940.1367, 1940.1958
        cx, cy = 2048.7397, 1551.3889
        width = 4096
        height = 3072

        cv_image_new = np.zeros([height, width])
        # cv2.imwrite("/home/aurmr/workspaces/uois_soofiyan_ws/src/segnetv2_mask2_former/Mask_Results/zero_cv_image.png", cv_image_new)
        pts = ros_numpy.numpify(request.points)
        pts = np.stack([pts['x'],
                        pts['y'],
                        pts['z']], axis=1)
        points_sort_z = np.flip(pts[pts[:, 2].argsort()], axis=0)
        
        # POINTS_TO_KEEP_FACTOR = .8
        # keep_idxs = np.arange(int(pts.shape[0]*(1-POINTS_TO_KEEP_FACTOR)), pts.shape[0])
        # points_lowered = points_sort_z[keep_idxs]
        points_lowered = pts
        # center  (2380.0, 1744.5)
        # 1536.9160707029534 -294.14115334486473
        # for i in points_lowered:
        #     # print(i)
           
        #     # [-0.246, -0.012, 1.410]
        #     z = i[0]+0.246
        #     v = ((-i[1]+0.012)*fx) / z
        #     u = ((i[2]-1.410)*fy) / z

        #     pixel_pos_x = (int)(u + cx)
        #     pixel_pos_y = (int)(v + cy)
        #     cv_image_new[int(cx-pixel_pos_x)][int(cy-pixel_pos_y)] = 1 
        # print("xyz", i[0],i[1] , i[2])
        # print("uv ", pixel_pos_x, pixel_pos_y)
        # cv2.imwrite("/home/aurmr/workspaces/uois_soofiyan_ws/src/segnetv2_mask2_former/Mask_Results/new_cv_image_mask.png", cv_image_new)
        # float z = msg->points[i].z*1000.0;
        # float u = (msg->points[i].x*1000.0*focal_x) / z;
        # float v = (msg->points[i].y*1000.0*focal_y) / z;
        # int pixel_pos_x = (int)(u + centre_x);
        # int pixel_pos_y = (int)(v + centre_y);

        print("rgb shape in grasp", self.rgb_img.shape)
        pts = ros_numpy.numpify(request.points)
        self.points_viz_pub.publish(request.points)
        pts = np.stack([pts['x'],
                        pts['y'],
                        pts['z']], axis=1)
        detections = self.detector.detect(pts)
        # if(self.depth_img and self.rgb_img):

        # pts = np.stack([pts['x'],
        #                 pts['y'],
        #                 pts['z']], axis=1)
        # detections = self.detector.detect(pts)
        # cv_image1 = cv2.resize(cv_image, (1280, 720), interpolation = cv2.INTER_AREA)
        # cv2.imshow("mask", cv_image1)
        # cv2.waitkey(0)

        

        # transform from base link to rgb camera link
        inference_object = run_normal_std()
        point, euler_angles = inference_object.inference(self.rgb_img, self.depth_img, cv_image)
        print("points", point)
        # POD_OFFSET = -0.11535
        # transform= self.tf_buffer.lookup_transform('base_link', 'rgb_camera_link_offset', rospy.Time())

        #transform_via_tf = self.tf_buffer.transform(PointStamped(point=Point(point[0], point[1], point[2])))
        '''T = np.eye(4)
        T[:3, 3] = (transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z)
        T[:3, :3] = R.from_quat((transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w)).as_matrix()
        points_in_base_link = np.array((*point, 1)) @ T
        print(transform.transform.translation)
        print(T)
        position = [0.0, 0.0, 0.0]
        print("pre position", point[2] + transform.transform.translation.x, -point[0] + transform.transform.translation.y, -point[1] + transform.transform.translation.z)
        position = points_in_base_link'''
        
        # euler_angles[0] = np.clip(euler_angles[0], -45, 45)
        # euler_angles[1] = np.clip(euler_angles[1], -45, 45)
        # if(euler_angles[0] > -5 and euler_angles[0] < 5):
        #     euler_angles[0] = 0
        # if(euler_angles[1] > -5 and euler_angles[1] < 5):
        #     euler_angles[1] = 0

        # position[2] = -points[1] + transform.transform.translation.z
        # with boling
        # orientation = transformations.quaternion_from_euler(math.pi/2., -math.pi/2. + euler_angles[1], euler_angles[0] + math.pi/2.)
        #orientation = transformations.quaternion_from_euler(math.pi/2., -math.pi/2., math.pi/2.)
        # orientation = transformations.quaternion_from_euler(math.pi/2., -math.pi/2., math.pi/2.)


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
