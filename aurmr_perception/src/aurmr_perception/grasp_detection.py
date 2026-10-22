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
from scipy.spatial.transform import Rotation
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
        elif(grasp_method == "sim_suction"):
            self.detect_grasps_rgb = rospy.Service('~detect_grasps', DetectGraspPoses, self.sim_suction_cb)
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

    def sim_suction_cb(self, request):


        from aurmr_setup.utils.workspace_utils import get_active_workspace_path

        import open3d as o3d

        import torch
        import copy
        import sys

        workspace_path = get_active_workspace_path()


        from sim_suction_model.sim_suction_pointnet import ScoreNet
        from sim_suction_inference import compute_darboux_frame
        from sim_suction_inference import check_collision_with_suction_gripper_using_voxelgrid


        crop_bin_bounds = self.bin_bounds[request.bin_id]
        rgb_image_cropped = self.rgb_img[crop_bin_bounds[0]:crop_bin_bounds[1], crop_bin_bounds[2]:crop_bin_bounds[3], ...]



        fx, fy = 1940.1367, 1940.1958
        cx, cy = 2048.7397, 1551.3889
        width = 4096
        height = 3072

        print(crop_bin_bounds)
        print(self.rgb_img.dtype) #unit 8
        print(self.depth_img.dtype) # unit 16
        print(self.depth_img.shape)
        print("rgb min", self.rgb_img.min())
        print("rgb max", self.rgb_img.max())


        color = o3d.geometry.Image(self.rgb_img)
        depth = o3d.geometry.Image(self.depth_img.astype(np.float32))

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth)
        camera_parameters = {"width": width, "height": height, "fx": fx, "fy": fy, "cx": cx, "cy": cy}
        intrinsics = o3d.camera.PinholeCameraIntrinsic(**camera_parameters)
        extrinsics = np.eye(4)
        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics, extrinsics, project_valid_depth_only=False)


        points = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors)

        print("colors.shape", colors.shape)

        points = points.reshape((height, width, 3))
        colors = colors.reshape((height, width, 3))

        points_cropped = points[crop_bin_bounds[0]:crop_bin_bounds[1], crop_bin_bounds[2]:crop_bin_bounds[3], ...]
        colors_cropped = colors[crop_bin_bounds[0]:crop_bin_bounds[1], crop_bin_bounds[2]:crop_bin_bounds[3], ...]

        print(points_cropped.shape)
        print(colors_cropped.shape)


        print(points_cropped.dtype) #float 64
        print(colors_cropped.dtype)

        points_cropped = points_cropped.reshape((-1, 3))
        colors_cropped = colors_cropped.reshape((-1, 3))

        bin_point_cloud = o3d.geometry.PointCloud()
        bin_point_cloud.points = o3d.utility.Vector3dVector(points_cropped)
        bin_point_cloud.colors = o3d.utility.Vector3dVector(colors_cropped)

        transform_camera_to_base_link= self.tf_buffer.lookup_transform('base_link', 'rgb_camera_link', rospy.Time())


        pose = ros_numpy.numpify(transform_camera_to_base_link.transform)


        bin_point_cloud = bin_point_cloud.transform(pose)

        coordinate_frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))

        o3d.visualization.draw_geometries_with_custom_animation([bin_point_cloud, coordinate_frame],width=720,height=720)


        # ..todo: pass squeeze=False?
        pc = ros_numpy.numpify(request.points)

        #print(request.points)

        points = np.zeros((pc.shape[0], 3))
        points[:, 0] = pc["x"]
        points[:, 1] = pc["y"]
        points[:, 2] = pc["z"]


        #pc["r"].dtype = np.uint8
        print()
        print("Numpy dtype")
        print(pc.dtype)
        print(pc.shape)


        # normal_vector
        print(rgb_image_cropped.shape)

        colors = np.zeros((pc.shape[0], 3))
        print(colors.dtype)
        print(colors.shape)

        # ..todo:: color information is not available
        #colors[:, 0] = pc["r"] * 255.0
        #colors[:, 1] = pc["g"] * 255.0
        #colors[:, 2] = pc["b"] * 255.0


        #rotate_to_table_top = Rotation.from_euler('xyz', [0, np.pi/2., 0], degrees=False)


        def add_plane(points, colors):
            transform = self.tf_buffer.lookup_transform('base_link', f'pod_bin_{request.bin_id.lower()}',  rospy.Time())
            bin_location = np.array([transform.transform.translation.x,  transform.transform.translation.y, transform.transform.translation.z])

            x = np.linspace(-0.5, 0.5, num=500)
            y = np.linspace(-0.5, 0.5, num=500)

            xv, yv = np.meshgrid(x, y)
            zv = np.zeros_like(xv)
            plane_points = np.stack([xv.ravel(), yv.ravel(), zv.ravel()], axis=1)
            plane_points += bin_location

            print(plane_points.shape)
            print(points.shape)
            points = np.concatenate([points, plane_points], axis=0)

            colors = np.concatenate([colors, np.zeros_like(plane_points)], axis=0)
            return points, colors


        points, colors = add_plane(points, colors)


        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        # rotate so we have a table top view :)
        # point_cloud = point_cloud.rotate(rotate_to_table_top.as_matrix(), center=(0, 0, 0))



        o3d.io.write_point_cloud("/tmp/test.pcd", point_cloud)

        o3d.io.write_point_cloud("/tmp/test2.pcd", bin_point_cloud)


        point_cloud.estimate_normals()
        point_cloud = point_cloud.scale(100.0, np.zeros(3))

        points = np.concatenate([np.asarray(point_cloud.points), np.asarray(point_cloud.normals), np.asarray(point_cloud.colors)], axis=1)


        # ..copied from sim_suction_inference.py

        def pc_normalize(pc):
            centroid = np.mean(pc, axis=0)
            pc = pc - centroid
            m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
            pc = pc / m
            return pc

        score_model = ScoreNet(training=False).cuda()


        # load model

        model_path = f"{workspace_path}/src/Sim-Suction-API/Sim-Suction-Pointnet/models/MV_PCL_1550_500.model"

        model_dict = torch.load(model_path, map_location='cuda:{}'.format(0)).state_dict() #, map_location='cpu'
        new_model_dict = {}
        for key in model_dict.keys():
            new_model_dict[key.replace("module.", "")] = model_dict[key]
        score_model.load_state_dict(new_model_dict)


        score_model.eval()
        with torch.no_grad():

            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, 1)




            def create_bin():
                """
                Constructs the bounding box of a bin given the bin id

                """

                transform = self.tf_buffer.lookup_transform('base_link', f'pod_bin_{request.bin_id.lower()}',  rospy.Time())
                bin_location = np.array([transform.transform.translation.x,  transform.transform.translation.y, transform.transform.translation.z])
                bin_height = 0.24
                bin_width = 0.24
                bin_depth = 0.15
                box_depth = 0.01
                color = np.array([1, 1, 0])

                print(bin_location)
                print(f'pod_bin_{request.bin_id.lower()}')

                back = o3d.geometry.TriangleMesh.create_box(width=box_depth, height=bin_width, depth=bin_height, create_uv_map=False, map_texture_to_each_face=False)
                back = back.translate(bin_location + np.array([bin_depth, -bin_width, 0]))
                back.paint_uniform_color(color)

                left_side = o3d.geometry.TriangleMesh.create_box(width=bin_depth, height=box_depth, depth=bin_height, create_uv_map=False, map_texture_to_each_face=False)
                left_side = left_side.translate(bin_location)

                right_side = o3d.geometry.TriangleMesh.create_box(width=bin_depth, height=box_depth, depth=bin_height, create_uv_map=False, map_texture_to_each_face=False)
                right_side = right_side.translate(bin_location + np.array([0, -bin_width, 0]))

                top = o3d.geometry.TriangleMesh.create_box(width=bin_depth, height=bin_width, depth=box_depth, create_uv_map=False, map_texture_to_each_face=False)
                top = top.translate(bin_location  + np.array([0, -bin_width, bin_height]))
                top.paint_uniform_color(color)

                bottom = o3d.geometry.TriangleMesh.create_box(width=bin_depth, height=bin_width, depth=box_depth, create_uv_map=False, map_texture_to_each_face=False)
                bottom = bottom.translate(bin_location+ np.array([0, -bin_width, 0]))
                bottom.paint_uniform_color(color)

                if False:
                    voxel_grid += back
                    voxel_grid += left_side
                    voxel_grid += right_side
                    voxel_grid += top
                    voxel_grid += bottom

                back += left_side
                back += right_side
                back += top
                back += bottom


                return back

            bin_mesh = create_bin()
            bin_mesh = bin_mesh.scale(100.0, np.zeros(3))
            #bin_mesh = bin_mesh.rotate(rotate_to_table_top.as_matrix(), center=(0, 0, 0))

            coordinate_frame = coordinate_frame.scale(100.0, np.zeros(3))


            bin_voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(bin_mesh,  voxel_size=1)


            o3d.visualization.draw_geometries_with_custom_animation([voxel_grid, bin_voxel_grid, coordinate_frame] ,width=720,height=720)


            points_ori=copy.deepcopy(points)


            points=points[:, 0:6]
            points[:, 0:3] = pc_normalize(points[:, 0:3])

            # ..todo store the normmalization.

            if len(points) < 5120:
                # Calculate the number of points needed
                num_points_needed = 5120 - len(points)
                zero_padding = np.zeros((num_points_needed, points.shape[1]))
                points = np.vstack((points, zero_padding))

            points=torch.from_numpy(points).float(). unsqueeze(0)


            pc = points.cuda()

            results,_ = score_model(pc, None)

            results_dict = {
                "top_1": {},
                "top_1%": {},
                "top_5%": {},
                "top_10%": {}
            }

            if results is None:
                print("No suction grasp candidates detected.")
                return None

            top_10_indices = results['top_10%'][1]
            ind_list,t_ori_list, t_translate_list, R_list, normal_vector_list, normal_vector_opposite_list = [],[], [], [], [], []
            bin_collision_list = []


            normals=np.asarray(point_cloud.normals)
            kdtree = o3d.geometry.KDTreeFlann(point_cloud)

            for index in top_10_indices:
                t_ori, t_translate, R, normal_vector, normal_vector_opposite = compute_darboux_frame(points_ori, index, kdtree, normals)

                # Check for collision
                collision = check_collision_with_suction_gripper_using_voxelgrid(voxel_grid, t_translate, normal_vector)

                # check if collision with bin

                print(normal_vector_opposite)
                print("R", R)

                bin_collision = check_collision_with_suction_gripper_using_voxelgrid(bin_voxel_grid, t_translate, normal_vector_opposite)

                print("is in collision with scene", collision)
                print("is in collision with bin", bin_collision)

                # If no collision, append to lists
                if True or collision:
                    t_ori_list.append(t_ori)
                    t_translate_list.append(t_translate)
                    R_list.append(R)
                    normal_vector_list.append(normal_vector)
                    normal_vector_opposite_list.append(normal_vector_opposite)
                    ind_list.append(index)
                    bin_collision_list.append(bin_collision)

            if ind_list==[]:
                print("No collision-free suction grasp candidates detected.")
                return None

            results_dict["top_10%"] = {
                "t_ori": t_ori_list,
                "t_translate": t_translate_list,
                "R": R_list,
                "normal_vector": normal_vector_list,
                "normal_vector_opposite": normal_vector_opposite_list,
                "relative_indices": list(range(len(ind_list))),
                "bin_collision": bin_collision_list
            }
            for top_key in ["top_1", "top_1%", "top_5%"]:
                absolute_indices = results[top_key][1]
                relative_indices = [top_10_indices.index(idx) for idx in absolute_indices if idx in ind_list]
                results_dict[top_key]["relative_indices"] = relative_indices

            visualize = "top_10%"

            subset_relative_indices = results_dict[visualize]["relative_indices"]

            subset_data = {
                "t_ori": [results_dict["top_10%"]["t_ori"][idx] for idx in subset_relative_indices],
                "t_translate": [results_dict["top_10%"]["t_translate"][idx] for idx in subset_relative_indices],
                "R": [results_dict["top_10%"]["R"][idx] for idx in subset_relative_indices],
                "normal_vector": [results_dict["top_10%"]["normal_vector"][idx] for idx in subset_relative_indices],
                "normal_vector_opposite": [results_dict["top_10%"]["normal_vector_opposite"][idx] for idx in subset_relative_indices],
                "bin_collision":  [results_dict["top_10%"]["bin_collision"][idx] for idx in subset_relative_indices],


            }

            display = []
            show_visualization= True
            # Visualize the specified subset's results
            for i in range(len(subset_relative_indices)):
                t_ori = subset_data["t_ori"][i]
                t_translate = subset_data["t_translate"][i]
                R = subset_data["R"][i]

                if show_visualization:

                    from sim_suction_inference import create_mesh_cylinder_detection

                    collides_with_bin = bin_collision_list[i]
                    mesh = create_mesh_cylinder_detection(R, t_ori, collides_with_bin)
                    display.append(mesh)



            if show_visualization:

                display.append(bin_mesh)
                display.append(point_cloud)
                o3d.visualization.draw_geometries_with_custom_animation(display,width=720,height=720)



            r_cam = Rotation.from_matrix(R)
            point = np.asarray(t_ori) / 100.0

            # ..todo:: revert "tabletop view"
            # r_prefix = rotate_to_table_top.inv()
            # r_cam = r_prefix * r_cam
            #point = np.dot(r_prefix.as_matrix(), point)
            point = np.dot(r_cam.as_matrix(), point)

            orientation_viz = r_cam.as_quat()

            as_quat_pose = Quaternion(x=orientation_viz[0], y=orientation_viz[1],
                                            z=orientation_viz[2], w=orientation_viz[3])
            as_pose_pose = Pose(position=Point(x=point[0], y=point[1], z=point[2]), orientation=as_quat_pose)
            t_header = deepcopy(request.points.header)
            #t_header.frame_id = 'rgb_camera_link_offset'
            vis_pose = PoseStamped(header=t_header, pose=as_pose_pose)
            self.grasp_viz_perception.publish(vis_pose)


            output_pose_stamped = self.transfor_pose(vis_pose, 'base_link')


            return True, "", [output_pose_stamped]


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
            DEPTH_TILT = -transform.transform.translation.z - 0.025
        elif(request.bin_id == "2H"):
            RGB_TO_DEPTH_FRAME_OFFSET = 0.017
            DEPTH_TILT = -transform.transform.translation.z - 0.025
        elif(request.bin_id == "3H"):
            RGB_TO_DEPTH_FRAME_OFFSET = 0.022
            DEPTH_TILT = -transform.transform.translation.z - 0.025
        else:
            RGB_TO_DEPTH_FRAME_OFFSET = 0.0225
            DEPTH_TILT = -transform.transform.translation.z - 0.0275
        POD_OFFSET = 0.1

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

        output_pose_stamped = self.transfor_pose(vis_pose, 'base_link')
        r_prefix = R.from_euler('xyz', [math.pi/2., -math.pi/2., math.pi/2.], degrees=False)
        r3 = r_prefix * r_cam
        orientation = r3.as_quat()
        output_pose_stamped.pose.orientation = Quaternion(x=orientation[0], y=orientation[1],
                                        z=orientation[2], w=orientation[3])

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

        output_pose_stamped = self.transfor_pose(vis_pose, 'base_link')
        r_prefix = R.from_euler('xyz', [math.pi/2., -math.pi/2., math.pi/2.], degrees=False)
        r3 = r_prefix * r_cam
        orientation = r3.as_quat()
        output_pose_stamped.pose.orientation = Quaternion(x=orientation[0], y=orientation[1],
                                        z=orientation[2], w=orientation[3])

        return True, "", [output_pose_stamped]
