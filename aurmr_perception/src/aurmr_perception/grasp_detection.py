import math

import numpy as np
import ros_numpy
import rospy
import tf2_ros

from std_msgs.msg import ColorRGBA
from aurmr_perception.srv import DetectGraspPoses
from geometry_msgs.msg import PoseStamped, Quaternion, Pose, Point, Vector3
from tf_conversions import transformations
from aurmr_perception.visualization import create_gripper_pose_markers
from sensor_msgs.msg import PointCloud2 

# See moveit_grasps for inspiration on improved grasp ranking and filtering:
# https://ros-planning.github.io/moveit_tutorials/doc/moveit_grasps/moveit_grasps_tutorial.html
from visualization_msgs.msg import MarkerArray, Marker


class HeuristicGraspDetector:
    def __init__(self, grasp_offset, bin_normal):
        print(grasp_offset)
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

        POD_OFFSET = -0.02
        transform= self.tf_buffer.lookup_transform('base_link', 'pod_base_link', rospy.Time())
        center[0] = transform.transform.translation.x- POD_OFFSET

        # NOTE(nickswalker,4-29-22): Hack to compensate for the chunk of points that don't get observed
        # due to the lip of the bin
        #center[2] -= 0.02
        print(points.shape, center.shape)
        position = self.grasp_offset * self.bin_normal + center
        align_to_bin_orientation = transformations.quaternion_from_euler(math.pi / 2., -math.pi / 2., math.pi / 2.)

        poses_stamped = [(position, align_to_bin_orientation)]
        # print(poses_stamped)
        return poses_stamped


class GraspDetectionROS:
    def __init__(self, detector):
        self.detector = detector
        self.detect_grasps = rospy.Service('~detect_grasps', DetectGraspPoses, self.detect_grasps_cb)
        self.dections_viz_pub = rospy.Publisher("~detected_grasps", MarkerArray, latch=True, queue_size=1)
        self.points_viz_pub = rospy.Publisher("~detected_pts", PointCloud2, latch=True, queue_size=5)

    def visualize_grasps(self, poses_stamped):
        color = ColorRGBA(r = 1, g = 0, b = 0, a = 1)
        scale=Vector3(x=.05, y=.05, z=.05)
        # print(poses_stamped)
        markers = Marker(header = poses_stamped[0].header, pose=poses_stamped[0].pose, type=1, scale=scale, color=color)
        self.dections_viz_pub.publish(MarkerArray(markers=[markers]))

    def detect_grasps_cb(self, request):
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