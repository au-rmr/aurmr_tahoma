import numpy as np

from aurmr_perception.srv import *
from geometry_msgs.msg import PoseStamped, Quaternion, Pose, Point
from tf_conversions import transformations


def generate_heuristic_grasp_poses(points, dist_threshold, bin_normal):
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

    position = dist_threshold * bin_normal + center  # center and extention_dir should be under the same coordiante!
    align_to_bin_orientation = transformations.quaternion_from_euler(1.57, 0, 1.57)
    align_to_bin_quat = Quaternion(x=align_to_bin_orientation[0], y=align_to_bin_orientation[1],
                                        z=align_to_bin_orientation[2], w=align_to_bin_orientation[3])

    poses_stamped = [PoseStamped(header=points.header, pose=Pose(position=Point(x=position[0], y=position[1], z=position[2]),orientation=align_to_bin_quat))]

    return poses_stamped
