from copy import deepcopy

import actionlib
import aurmr_perception
import rospy
import numpy as np
import ros_numpy
import cv2
import math
import pickle

from smach import State
from cv_bridge import CvBridge
from tf_conversions import transformations
from aurmr_perception.util import qv_mult, quat_msg_to_vec

from aurmr_hri.msg import RetryGraspAction, RetryGraspGoal
from sensor_msgs.msg import CompressedImage, Image, PointCloud2, CameraInfo
from geometry_msgs.msg import PoseStamped, Quaternion, Pose, Point
from visualization_msgs.msg import Marker
import image_geometry

from aurmr_tasks.util import add_offset


class UserPromptForRetry(State):
    def __init__(self, bin_bounds, tf_buffer, frame_id='base_link', timeout_connection_secs = 10.0, \
                 timeout_response_secs = 120.0, camera_name = 'camera_lower_right', use_depth=False):
        State.__init__(
            self,
            input_keys=['target_bin_id', 'target_object_id', 'target_object_asin', 'grasp_pose'],
            output_keys=['human_grasp_pose', "human_pre_grasp_pose"],
            outcomes=['retry', 'continue']
        )
        self.bin_bounds = bin_bounds
        self.timeout_connection_secs = timeout_connection_secs
        self.timeout_response_secs = timeout_response_secs
        self.camera_name = camera_name
        self.frame_id = frame_id
        self.tf_buffer = tf_buffer
        self.points_sub = rospy.Subscriber(f'/{camera_name}/points2', PointCloud2, self.points_cb)
        self.rgb_image_sub = rospy.Subscriber(f'/{camera_name}/rgb/image_raw', Image, self.rgb_image_cb)
        self.camera_model = None
        self.camera_info_sub = rospy.Subscriber(f'/{camera_name}/rgb/camera_info', CameraInfo, self.camera_info_cb,  queue_size = 1)
        self.depth_image_sub = rospy.Subscriber(f"/{camera_name}/depth/image_raw", Image, self.depth_image_cb)
        self.marker_publisher = rospy.Publisher("visualization_marker", Marker, queue_size=1)
        self.ros_pointcloud = None
        self.ros_rgb_image = None
        self.use_depth = use_depth # Either depth or point cloud will be used
        self.bridge = CvBridge()

    def points_cb(self, ros_pointcloud):
        self.ros_pointcloud = ros_pointcloud

    def rgb_image_cb(self, ros_rgb_image):
        self.ros_rgb_image = ros_rgb_image

    def depth_image_cb(self, ros_depth_image):
        self.ros_depth_image = ros_depth_image

    def camera_info_cb(self, msg):
        self.camera_model = image_geometry.PinholeCameraModel()
        self.camera_model.fromCameraInfo(msg)
        self.camera_info_sub.unregister() #Only subscribe once


    def visualize_point_in_image(self, rgb_image, x, y):
        rgb_image = cv2.circle(rgb_image,(x, y), 20, (0,255,0), -1)

        scale_percent = 20 # percent of original size
        width = int(rgb_image.shape[1] * scale_percent / 100)
        height = int(rgb_image.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(rgb_image, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow('rgb_image', resized)
        cv2.waitKey(-1)

    def visualize_ray(self, ray, frame_id, length=2):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.rostime.Time.now()
        marker.ns = "basic_shapes"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        p1 = Point()
        p1.x = 0
        p1.y = 0
        p1.z = 0

        p2 = Point()
        p2.x = ray[0] * length
        p2.y = ray[1] * length
        p2.z = ray[2] * length

        marker.points = [p1, p2]

        # Set the scale of the marker -- 1x1x1 here means 1m on a side
        marker.scale.x = 0.01
        marker.scale.y = 0.01
        marker.scale.z = 0.05

        # Set the color -- be sure to set alpha to something non-zero!
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.lifetime = rospy.rostime.Duration()

        self.marker_publisher.publish(marker)

    def visualize_point_marker(self, point, frame_id):
        marker2 = Marker()
        # marker2.header.frame_id = self.camera_model.tfFrame()
        marker2.header.frame_id = frame_id
        marker2.header.stamp = rospy.rostime.Time.now()
        marker2.ns = "basic_shapes"
        marker2.id = 1
        marker2.type = Marker.SPHERE
        marker2.action = Marker.ADD

        marker2.pose.position.x = point[0]
        marker2.pose.position.y = point[1]
        marker2.pose.position.z = point[2]
        marker2.pose.orientation.x = 0.0;
        marker2.pose.orientation.y = 0.0;
        marker2.pose.orientation.z = 0.0;
        marker2.pose.orientation.w = 1.0;

        # marker2.points = [p1, p2]

        # Set the scale of the marker -- 1x1x1 here means 1m on a side
        marker2.scale.x = 0.05
        marker2.scale.y = 0.05
        marker2.scale.z = 0.05

        # Set the color -- be sure to set alpha to something non-zero!
        marker2.color.r = 0.0
        marker2.color.g = 1.0
        marker2.color.b = 0.0
        marker2.color.a = 1.0

        marker2.lifetime = rospy.rostime.Duration()

        self.marker_publisher.publish(marker2)

    def get_xyz(self, u, v):
        ray = self.camera_model.projectPixelTo3dRay((u, v))
        ray_z = [el/ray[2] for el in ray]

        self.visualize_ray(ray, self.camera_model.tfFrame())

        if self.use_depth:
            # Get xyz using depth
            depth_image = ros_numpy.numpify(self.ros_depth_image).astype(np.float32)
            depth_z = depth_image[int(v), int(u)]
            camera_point = np.array(ray_z) * (depth_z/1000)
            return camera_point
        else:
            # Get xyz using pointcloud
            # pc_baselink = self.tf_buffer.transform(self.ros_pointcloud, 'base_link')
            pc = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(self.ros_pointcloud, remove_nans=True)
            p1 = np.array([0,0,0])
            p0 = np.array(ray_z) * 2
            closest_point = pc[np.argmin(np.linalg.norm(np.cross(p1-p0, p0-pc, axisb=1), axis=1)/np.linalg.norm(p1-p0))]
            return closest_point

    def execute(self, userdata):
        if self.ros_pointcloud is None or self.ros_rgb_image is None or self.camera_model is None:
            rospy.loginfo("Pointcloud and RGB image not ready for HRI regrasp")
            return "continue"

        if userdata['target_bin_id'] not in self.bin_bounds:
            rospy.loginfo(f"No bin configuration found for {userdata['target_bin_id']}")
            return "continue"

        client = actionlib.SimpleActionClient('/aurmr/hri/retry_grasp', RetryGraspAction)
        if not client.wait_for_server(rospy.Duration.from_sec(self.timeout_connection_secs)):
            rospy.loginfo("UserPromptForRetry timed out connecting to server")
            return "continue"

        rgb_image = self.bridge.imgmsg_to_cv2(self.ros_rgb_image)

        bounds = self.bin_bounds[userdata['target_bin_id']]
        cropped_rgb_image = rgb_image[bounds[0]:bounds[1], bounds[2]:bounds[3], 0:3]

        image_msg = CompressedImage()
        image_msg.header.stamp = rospy.Time.now()
        image_msg.format = "jpeg"
        image_msg.data = np.array(cv2.imencode('.jpg', cropped_rgb_image)[1]).tostring()

        goal = RetryGraspGoal(camera_image=image_msg, object_asin=userdata['target_object_asin'])
        client.send_goal(goal)

        if not client.wait_for_result(rospy.Duration.from_sec(self.timeout_response_secs)):
            rospy.loginfo("UserPromptForRetry timed out waiting for response")
            client.cancel_goal()
            return "continue"

        result = client.get_result()
        if result == None or not result.retry:
            rospy.loginfo("User declined help request")
            return "continue"

        u = round(result.x + bounds[2])
        v = round(result.y + bounds[0])

        # self.visualize_point_in_image(rgb_image, u, v)
        xyz = self.get_xyz(u, v)

        # Start building pose
        grasp_point = Point(x=xyz[0], y=xyz[1], z=xyz[2])

        orientation = transformations.quaternion_from_euler(math.pi / 2., -math.pi / 2., math.pi / 2.)

        quaternion = Quaternion(x=orientation[0], y=orientation[1], z=orientation[2], w=orientation[3])

        grasp_pose = PoseStamped(pose=Pose(position=grasp_point, orientation=quaternion))
        grasp_pose.header.frame_id = self.camera_model.tfFrame()

        # grasp_pose = self.tf_buffer.transform(grasp_pose, "base_link")

        grasp_pose = self.tf_buffer.transform(grasp_pose, "base_link")
        grasp_pose.pose.orientation = quaternion

        transform= self.tf_buffer.lookup_transform('base_link', 'pod_base_link', rospy.Time())
        POD_OFFSET = 0.1
        RGB_TO_DEPTH_FRAME_OFFSET = transform.transform.translation.y-0.47
        DEPTH_TILT = -transform.transform.translation.z-0.02

        grasp_pose.pose.position.z += DEPTH_TILT
        grasp_pose.pose.position.y -= RGB_TO_DEPTH_FRAME_OFFSET
        grasp_pose.pose.position.x = transform.transform.translation.x - POD_OFFSET

        def visualize(pose, frame='base_link'):
            self.visualize_point_marker([
                pose.position.x,
                pose.position.y,
                pose.position.z
            ], frame)

        # visualize(grasp_pose.pose)
        # import pdb; pdb.set_trace()

        # visualize()

        # import pdb; pdb.set_trace()


        # As the arm_tool0 is 20cm in length w.r.t tip of suction cup thus adding 0.2m offset
        grasp_pose = add_offset(-0.20, grasp_pose)

        userdata['human_grasp_pose'] = grasp_pose

        # adding 0.12m offset for pre grasp pose to prepare it for grasp pose which is use to pick the object
        pregrasp_pose = add_offset(-self.pre_grasp_offset, grasp_pose)

        userdata['human_pre_grasp_pose'] = pregrasp_pose

        return "retry"



def prompt_for_confirmation(prompt):
    valid_signal = False
    while not rospy.is_shutdown() and not valid_signal:
        user_input = input(f"{prompt}\nProceed [y/n]?")
        if user_input == "y":
            return True
        elif user_input == "n":
            return False


class AskForHumanAction(State):
    def __init__(self, default_prompt=None):
        State.__init__(self, outcomes=['succeeded', 'aborted'], input_keys=["prompt"])
        self.prompt = default_prompt

    def execute(self, userdata):
        prompt = self.prompt
        if not self.prompt:
            prompt = userdata["prompt"]
        confirmed = prompt_for_confirmation(prompt)
        if confirmed:
            return "succeeded"
        else:
            return "aborted"


class WaitForKeyPress(State):
    def __init__(self):
        State.__init__(self, outcomes=['signalled', 'not_signalled', 'aborted'])

    def execute(self, userdata):
        try:
            user_input = input("Press any key to start\n")
            return 'signalled'
        except KeyboardInterrupt:
            return 'aborted'
