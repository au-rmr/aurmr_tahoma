import math

import numpy as np
import ros_numpy
import rospy

from aurmr_perception.srv import DetectGraspPoses
from geometry_msgs.msg import PoseStamped, Quaternion, Pose, Point
from tf_conversions import transformations
from aurmr_perception.visualization import create_gripper_pose_markers

# See moveit_grasps for inspiration on improved grasp ranking and filtering:
# https://ros-planning.github.io/moveit_tutorials/doc/moveit_grasps/moveit_grasps_tutorial.html
from visualization_msgs.msg import MarkerArray, Marker


class HeuristicGraspDetector:
    def __init__(self, dist_threshold, bin_normal):
        self.dist_threshold = dist_threshold
        self.bin_normal = np.array(bin_normal)

    def detect(self, points):
        """
        Hacked together for the first grasp
        :param points:
        :return:
        """
        # compute the average point of the pointcloud
        center = np.mean(points, axis=0)
        # NOTE(nickswalker,4-29-22): Hack to compensate for the chunk of points that don't get observed
        # due to the lip of the bin
        center[2] -= 0.02

        position = self.dist_threshold * self.bin_normal + center  # center and extention_dir should be under the same coordiante!
        align_to_bin_orientation = transformations.quaternion_from_euler(0, math.pi / 2., 0)

        poses_stamped = [(position, align_to_bin_orientation)]

        return poses_stamped


class GraspDetectionROS:
    def __init__(self, detector):
        self.detector = detector
        self.detect_grasps = rospy.Service('~detect_grasps', DetectGraspPoses, self.detect_grasps_cb)
        self.dections_viz_pub = rospy.Publisher("~detected_grasps", MarkerArray, latch=True, queue_size=1)

    def visualize_grasps(self, poses_stamped):
        markers = create_gripper_pose_markers(poses_stamped, (1,0,1,1))
        self.dections_viz_pub.publish(MarkerArray(markers=markers))

    def detect_grasps_cb(self, request):
        pts = ros_numpy.numpify(request.points)
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