import numpy as np
import ros_numpy
import rospy

from aurmr_perception.srv import *
from geometry_msgs.msg import PoseStamped, Quaternion, Pose, Point
from tf_conversions import transformations


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
        align_to_bin_orientation = transformations.quaternion_from_euler(1.57, 0, 1.57)
        align_to_bin_quat = Quaternion(x=align_to_bin_orientation[0], y=align_to_bin_orientation[1],
                                            z=align_to_bin_orientation[2], w=align_to_bin_orientation[3])

        poses_stamped = [PoseStamped(header=points.header, pose=Pose(position=Point(x=position[0], y=position[1], z=position[2]),orientation=align_to_bin_quat))]

        return poses_stamped


class GraspDetectionROS:
    def __init__(self, detector):
        self.detector = detector
        self.detect_grasps = rospy.Service('~detect_grasps', GraspPose, self.detect_grasps_cb)

    def detect_grasps_cb(self, request):
        pts = ros_numpy.numpify(request.points)
        pts = np.stack([pts['x'],
                        pts['y'],
                        pts['z']], axis=1)
        detections = self.detector.detect(pts)
        return True, "", detections[0], 0.0