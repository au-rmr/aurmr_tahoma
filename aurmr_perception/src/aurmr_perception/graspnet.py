import argparse
import cv2

import rospy
import ros_numpy
import message_filters
import threading as th

from collections import defaultdict
from sensor_msgs.msg import Image, CameraInfo, PointField, PointCloud2
from sensor_msgs.point_cloud2 import read_points
from std_msgs.msg import Header
from aurmr_perception.srv import *
from geometry_msgs.msg import PoseStamped, Quaternion, Pose, Point
from tf_conversions import transformations

import numpy as np
import sys
import os
os.chdir('/home/aurmr/workspaces/vatsa_ws/graspnet/pytorch_6dof-graspnet') # Location of graspnet
sys.path.insert(1, '/home/aurmr/workspaces/vatsa_ws/graspnet/pytorch_6dof-graspnet')

# os.chdir('/media/enigma/Amplitude/Study/UW/RSE_Lab/pytorch_6dof-graspnet') # Location of graspnet
# sys.path.insert(1, '/media/enigma/Amplitude/Study/UW/RSE_Lab/pytorch_6dof-graspnet')
import grasp_estimator

import glob
import mayavi.mlab as mlab
from utils.visualization_utils import *
from utils import utils
from data import DataLoader
import open3d as o3d

def make_graspnet_parser():
    parser = argparse.ArgumentParser(
    description='6-DoF GraspNet Demo',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--grasp_sampler_folder',
                        type=str,
                        default='checkpoints/gan_pretrained/')
    parser.add_argument('--grasp_evaluator_folder',
                        type=str,
                        default='checkpoints/evaluator_pretrained/')
    parser.add_argument('--refinement_method',
                        choices={"gradient", "sampling"},
                        default='sampling')
    parser.add_argument('--refine_steps', type=int, default=25)

    parser.add_argument('--npy_folder', type=str, default='demo/data/')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.8,
        help=
        "When choose_fn is something else than all, all grasps with a score given by the evaluator notwork less than the threshold are removed"
    )
    parser.add_argument(
        '--choose_fn',
        choices={
            "all", "better_than_threshold", "better_than_threshold_in_sequence"
        },
        default='better_than_threshold',
        help=
        "If all, no grasps are removed. If better than threshold, only the last refined grasps are considered while better_than_threshold_in_sequence consideres all refined grasps"
    )

    parser.add_argument('--target_pc_size', type=int, default=1024)
    parser.add_argument('--num_grasp_samples', type=int, default=200)
    parser.add_argument(
        '--generate_dense_grasps',
        action='store_true',
        help=
        "If enabled, it will create a [num_grasp_samples x num_grasp_samples] dense grid of latent space values and generate grasps from these."
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=30,
        help=
        "Set the batch size of the number of grasps we want to process and can fit into the GPU memory at each forward pass. The batch_size can be increased for a GPU with more memory."
    )
    parser.add_argument('--train_data', action='store_true')
    opts, _ = parser.parse_known_args()
    if opts.train_data:
        parser.add_argument('--dataset_root_folder',
                            required=True,
                            type=str,
                            help='path to root directory of the dataset.')
    return parser


def make_graspnet_estimator(args):
    grasp_sampler_args = utils.read_checkpoint_args(args.grasp_sampler_folder)
    grasp_sampler_args.is_train = False
    grasp_evaluator_args = utils.read_checkpoint_args(
        args.grasp_evaluator_folder)
    grasp_evaluator_args.continue_train = True
    estimator = grasp_estimator.GraspEstimator(grasp_sampler_args,
                                               grasp_evaluator_args, args)
    return estimator


class GraspNetDetector:
    def __init__(self, estimator):
        align_to_bin_orientation = transformations.quaternion_from_euler(1.57, 0, 1.57)
        self.align_to_bin_quat = Quaternion(x=align_to_bin_orientation[0], y=align_to_bin_orientation[1],
                                       z=align_to_bin_orientation[2], w=align_to_bin_orientation[3])
        self.normal_vector = np.array((-1,0,0))
        self.rgb_image = None
        self.depth_image = None
        self.camera_info = None
        self.object_pointcloud = None
        self.estimator = estimator
        self.sorted_generated_grasps = None
        self.sorted_generated_scores = None

    def display_inlier_outlier(self, cloud, ind):
        inlier_cloud = cloud.select_by_index(ind)
        outlier_cloud = cloud.select_by_index(ind, invert=True)

        print("Showing outliers (red) and inliers (gray): ")
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    def filter_pointcloud(self):
        satisfied = 'n'
        nb_points=350
        radius=0.015
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.object_pointcloud)

        while satisfied == 'n':
            cl, ind = pcd.remove_radius_outlier(nb_points, radius)
            self.display_inlier_outlier(pcd, ind)
            satisfied = input('Satisfied with the pointcloud filtering? (y/n)')
            if satisfied == 'n':
                nb_points = int(input('Enter new value of nb_points (default value = 450): '))
                radius = float(input('Enter new value of radius (default value = 0.015): '))

        self.object_pointcloud = np.asarray(cl.points)

    def filter_grasps(self, grasps, grasp_scores):
        filtered_grasps = []
        filtered_scores = []
        for grasp, grasp_score in zip(grasps, grasp_scores):
            euler_grasp = transformations.euler_from_matrix(grasp[0:3, 0:3])
            norm_euler_grasp = np.linalg.norm(euler_grasp[0:2])  # Taking only y and z angles and ignoring rot along x axis
            if norm_euler_grasp < 0.3:
                # print(grasp)
                filtered_grasps.append(grasp)
                filtered_scores.append(grasp_score)
            # print(norm_euler_grasp)
        return filtered_grasps, filtered_scores

    def backproject(self, depth_cv,
                intrinsic_matrix,
                return_finite_depth=True,
                return_selection=False):

        depth = depth_cv.astype(np.float32, copy=True)

        # get intrinsic matrix
        K = intrinsic_matrix
        Kinv = np.linalg.inv(K)

        # compute the 3D points
        width = depth.shape[1]
        height = depth.shape[0]

        # construct the 2D points matrix
        y, z = np.meshgrid(np.arange(width), np.arange(height))
        ones = np.ones((height, width), dtype=np.float32)
        x2d = np.stack((ones, y, z), axis=2).reshape(width * height, 3)

        # backprojection
        R = np.dot(Kinv, x2d.transpose())

        # compute the 3D points
        X = np.multiply(np.tile(depth.reshape(1, width * height), (3, 1)), R)
        X = np.array(X).transpose()
        if return_finite_depth:
            selection = np.isfinite(X[:, 0])
            X = X[selection, :]

        if return_selection:
            return X, selection

        return X

    def visualize_graspnet(self):
        # self.depth_image[self.depth_image == 0 or self.depth_image > 1] = np.nan

        np.nan_to_num(self.depth_image, copy=False)
        mask = np.where(np.logical_or(self.depth_image == 0, self.depth_image > 1.5))
        print('mask', mask)
        self.depth_image[mask] = np.nan

        pc, selection = self.backproject(self.depth_image,
                                    self.camera_info,
                                    return_finite_depth=True,
                                    return_selection=True)
        pc_colors = self.rgb_image.copy()
        pc_colors = np.reshape(pc_colors, [-1, 3])
        pc_colors = pc_colors[selection, :]

        print('Opening mlab figure')

        mlab.figure(bgcolor=(1, 1, 1))

        # print(sorted_generated_grasps, sorted_generated_grasps[0].shape)
        # print(sorted_generated_scores)

        print('Drawing grasp scene')
        draw_scene(
            pc,
            pc_color=pc_colors,
            grasps=self.sorted_generated_grasps,
            grasp_scores=self.sorted_generated_scores,
        )

    def graspnet(self):
        # Depending on your numpy version you may need to change allow_pickle
        # from True to False.
        # input('Extracting depth')
        depth_scale_factor = 0.001
        self.depth_image = self.depth_image * depth_scale_factor
        self.object_pointcloud = np.float64(self.object_pointcloud)

        # 90 degree yaw and 90 degree pitch
        transformation_matrix = np.array([[0, -1, 0, 0],
                                        [0, 0, -1, 0],
                                        [1, 0, 0, 0],
                                        [0, 0, 0, 1]])

        object_pc = np.vstack([self.object_pointcloud.T, np.ones(self.object_pointcloud.shape[0])])
        object_pc = np.matmul(transformation_matrix, object_pc)[0:3,:].T

        generated_grasps, generated_scores = self.estimator.generate_and_refine_grasps(object_pc)

        filtered_grasps, filtered_scores = self.filter_grasps(generated_grasps, generated_scores)

        sorted_grasp_index = sorted(range(len(filtered_scores)), key=filtered_scores.__getitem__)
        self.sorted_generated_grasps = [filtered_grasps[i] for i in reversed(sorted_grasp_index)]
        self.sorted_generated_scores = [filtered_scores[i] for i in reversed(sorted_grasp_index)]

        transformation_matrix_inv = np.array([[0, 0, 1, 0],
                                        [-1, 0, 0, 0],
                                        [0, -1, 0, 0],
                                        [0, 0, 0, 1]])

        transformation_matrix_rot = np.array([[0, -1, 0, 0],
                                        [0, 0, -1, 0],
                                        [1, 0, 0, 0],
                                        [0, 0, 0, 1]])

        return_pose = transformation_matrix_inv @ self.sorted_generated_grasps[0]

        return return_pose

    def key_capture_thread(a_list):
        input('Press enter to continue...')
        a_list.append(True)

    def pc_publish(self, pts):
        pub_opc = rospy.Publisher('ObjectPC', PointCloud2, queue_size=10)
        rate = rospy.Rate(1)
        a_list = []
        th.Thread(target=self.key_capture_thread, args=(), name='key_capture_thread', daemon=True).start()
        while not a_list:
            pub_opc.publish(pts)
            rate.sleep()

    def pose_publish(self, pose_stamped, pts):
        pub = rospy.Publisher('GraspPose', PoseStamped, queue_size=10, latch=True)
        pub_opc = rospy.Publisher('ObjectPC', PointCloud2, queue_size=10, latch=True)

        print('Pose is being published')
        pub.publish(pose_stamped)
        pub_opc.publish(pts)

    def save_points(self):
        np.savez('/home/enigma/catkin_ws/trial.npz', image=self.rgb_image,
                            depth=self.depth_image,
                            intrinsics_matrix=self.camera_info,
                            smoothed_object_pc = self.object_pointcloud)
        print('Points Saved')

    def grasping_callback(self, request):
        # get the average of the pointclouds

        # self.pc_publish(request.points)

        self.object_pointcloud = ros_numpy.numpify(request.points)
        self.object_pointcloud = np.stack([self.object_pointcloud['x'],
                                            self.object_pointcloud['y'],
                                            self.object_pointcloud['z']], axis=1)

        self.filter_pointcloud()

        ros_depth_image = rospy.wait_for_message('/camera_lower_left/aligned_depth_to_color/image_raw', Image, timeout=1)
        ros_rgb_image = rospy.wait_for_message('/camera_lower_left/color/image_raw', Image, timeout=1)
        ros_camera_info = rospy.wait_for_message('/camera_lower_left/color/camera_info', CameraInfo, timeout=1)

        self.rgb_image = ros_numpy.numpify(ros_rgb_image)
        self.depth_image = ros_numpy.numpify(ros_depth_image)
        self.camera_info = np.array(ros_camera_info.K).reshape((3,3))

        # self.save_points()

        print('Running-Graspnet')
        grasp_pose_se3 = self.graspnet()
        print('Graspnet ran successfully.')

        grasp_point = Point(x=grasp_pose_se3[0,3], y=grasp_pose_se3[1,3], z=grasp_pose_se3[2,3])

        grasp_orientation_quat = transformations.quaternion_from_matrix(grasp_pose_se3)
        grasp_orientation = Quaternion(x=grasp_orientation_quat[0], y=grasp_orientation_quat[1],
                                       z=grasp_orientation_quat[2], w=grasp_orientation_quat[3])

        grasp_pose = Pose(position=grasp_point,
                                orientation=grasp_orientation)

        grasp_pose_stamped = PoseStamped(header=request.points.header,
                                pose=grasp_pose)

        self.pose_publish(grasp_pose_stamped, request.points)

        return GraspPoseResponse(success=True, message=f"Grasping Pose has been set", # The function name will be different
                                pose = grasp_pose_stamped,
                                grasp = 0.02)      #gripper_dist
