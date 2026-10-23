import numpy as np
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2

import open3d as o3d

LISTENER = '/grasp_detection/detected_pts'
TOF_PCD_LISTENER = '/tof_pcd'

perception_pcd = []
tof_pcd = []

def listener():
    rospy.Subscriber(LISTENER, PointCloud2, callback, queue_size=1)
    rospy.Subscriber(LISTENER, PointCloud2, tof_pcd_callback, queue_size=1)

def callback(msg):
    global perception_pcd
    # print(msg)
    # import pdb; pdb.set_trace()
    perception_pcd = ros_numpy.numpify(msg)

def tof_pcd_callback(msg):
    global tof_pcd
    tof_pcd = ros_numpy.numpify(msg)

if __name__ == '__main__':

    rospy.init_node('target_point_cloud', anonymous=True)
    rate = rospy.Rate(10) # 10h
    while not rospy.is_shutdown():
        listener()
        # print(perception_pcd)
        print(tof_pcd)
        pcd_0 = o3d.geometry.PointCloud()
        pcd_0.points = o3d.utility.Vector3dVector(tof_pcd)
        print(type(pcd_0))
