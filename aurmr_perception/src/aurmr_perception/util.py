from geometry_msgs.msg import Quaternion
from tf import transformations
import numpy as np
import ros_numpy

I_QUAT = Quaternion(x=0, y=0, z=0, w=1)
ROT_90_Z_QUAT = Quaternion(x=0, y=0, z=.707, w=.707)

def qv_mult(q1, v1):
    assert len(q1) == 4
    assert len(v1) == 3
    q2 = list(v1)
    q2.append(0.0)
    return transformations.quaternion_multiply(
        transformations.quaternion_multiply(q1, q2),
        transformations.quaternion_conjugate(q1)
    )[:3]


def quat_msg_to_vec(msg):
    return [msg.x, msg.y, msg.z, msg.w]


def vec_to_quat_msg(vec):
    return Quaternion(x=vec[0],y=vec[1],z=vec[2],w=vec[3])


def compute_xyz(depth_img, intrinsic):
    fx = intrinsic[0][0]
    fy = intrinsic[1][1]
    px = intrinsic[0][2]
    py = intrinsic[1][2]
    height, width = depth_img.shape

    indices = np.indices((height, width), dtype=np.float32).transpose(1,2,0)
    z_e = depth_img
    x_e = (indices[..., 1] - px) * z_e / fx
    y_e = (indices[..., 0] - py) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # Shape: [H x W x 3]
    return xyz_img


def numpify_pointcloud(points, im_shape):
    final_shape = list(im_shape[0:2])
    # final_shape.append(3)
    points = ros_numpy.numpify(points)

    points = np.reshape(points, final_shape)
    # points = np.vstack((points['x'],points['y'],points['z']))
    points = np.stack((points['x'],points['y'],points['z']), axis=2)

    return points


def mask_pointcloud(points, mask):
    points = ros_numpy.numpify(points)
    points_seg = points[mask.flatten() > 0]
    points_seg = np.vstack((points_seg['x'],points_seg['y'],points_seg['z']))
    points_seg = points_seg[:, np.invert(np.isnan(points_seg[2, :]))]
    return points_seg