from geometry_msgs.msg import Quaternion
from tf import transformations

I_QUAT = Quaternion(x=0, y=0, z=0, w=1)


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
