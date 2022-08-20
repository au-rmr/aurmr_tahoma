from copy import deepcopy

import aurmr_perception
import geometry_msgs
import rospy

from smach import State
from aurmr_perception.srv import (
    DetectGraspPoses,
    ResetBin,
    ResetBinRequest,
    GetObjectPoints,
    GetObjectPointsRequest,
    CaptureObjectRequest,
)
from std_srvs.srv import Trigger

from aurmr_perception.util import qv_mult, quat_msg_to_vec, vec_to_quat_msg


class CaptureEmptyBin(State):
    def __init__(self):
        State.__init__(self, input_keys=['target_bin_id'], outcomes=['succeeded', 'preempted', 'aborted'])
        self.reset_bin = rospy.ServiceProxy('/aurmr_perception/reset_bin', ResetBin)

    def execute(self, userdata):
        reset_bin_req = ResetBinRequest(bin_id=userdata['target_bin_id'])
        reset_response = self.reset_bin(reset_bin_req)

        if reset_response.success:
            return "succeeded"
        else:
            return "aborted"

class CaptureEmptyPod(State):
    def __init__(self):
        State.__init__(self, input_keys=[], outcomes=['succeeded'])
        self.capture_empty = rospy.ServiceProxy('/aurmr_perception/capture_empty_pod', Trigger)

    def execute(self, userdata):
        res = self.capture_empty()

        if res.success:
            return "succeeded"
        else:
            return "aborted"

class CaptureObject(State):
    def __init__(self):
        State.__init__(
            self,
            input_keys=['target_bin_id', 'target_object_id'],
            outcomes=['succeeded', 'preempted', 'aborted']
        )
        self.capture_object = rospy.ServiceProxy('/aurmr_perception/capture_object', aurmr_perception.srv.CaptureObject)
        self.capture_object.wait_for_service(timeout=rospy.Duration(5))

    def execute(self, userdata):
        capture_obj_req = CaptureObjectRequest(
            bin_id=userdata['target_bin_id'],
            object_id=userdata['target_object_id'],
        )
        rospy.loginfo("in CAPTUREOBJECT" + userdata['target_bin_id'])
        capture_response = self.capture_object(capture_obj_req)

        if capture_response.success:
            return "succeeded"
        else:
            return "aborted"


class StowObject(State):
    def __init__(self):
        State.__init__(
            self,
            input_keys=['target_bin_id', 'target_object_id'],
            outcomes=['succeeded', 'preempted', 'aborted']
        )
        self.capture_object = rospy.ServiceProxy('/aurmr_perception/stow_object', aurmr_perception.srv.CaptureObject)
        self.capture_object.wait_for_service(timeout=rospy.Duration(5))

    def execute(self, userdata):
        capture_obj_req = CaptureObjectRequest(
            bin_id=userdata['target_bin_id'],
            object_id=userdata['target_object_id'],
        )
        rospy.loginfo("in STOWOBJECT" + userdata['target_bin_id'])
        capture_response = self.capture_object(capture_obj_req)

        if capture_response.success:
            return "succeeded"
        else:
            return "aborted"

class PickObject(State):
    def __init__(self):
        State.__init__(
            self,
            input_keys=['target_bin_id', 'target_object_id'],
            outcomes=['succeeded', 'preempted', 'aborted']
        )
        self.capture_object = rospy.ServiceProxy('/aurmr_perception/pick_object', aurmr_perception.srv.CaptureObject)
        self.capture_object.wait_for_service(timeout=rospy.Duration(5))

    def execute(self, userdata):
        capture_obj_req = CaptureObjectRequest(
            bin_id=userdata['target_bin_id'],
            object_id=userdata['target_object_id'],
        )
        rospy.loginfo("in CAPTUREOBJECT" + userdata['target_bin_id'])
        capture_response = self.capture_object(capture_obj_req)

        if capture_response.success:
            return "succeeded"
        else:
            return "aborted"

class GetGraspPose(State):
    def __init__(self, tf_buffer, frame_id='base_link', pre_grasp_offset=.12):
        State.__init__(
            self,
            input_keys=['target_bin_id', 'target_object_id'],
            output_keys=['grasp_pose', 'pre_grasp_pose'],
            outcomes=['succeeded', 'preempted', 'aborted']
        )
        self.get_points = rospy.ServiceProxy('/aurmr_perception/get_object_points', GetObjectPoints)
        self.get_grasp = rospy.ServiceProxy('/grasp_detection/detect_grasps', DetectGraspPoses)
        # Crash during initialization if these aren't running so see the problem early
        self.get_points.wait_for_service(timeout=5)
        self.get_grasp.wait_for_service(timeout=5)
        self.frame_id = frame_id
        self.pre_grasp_offset = pre_grasp_offset
        self.pose_viz = rospy.Publisher("selected_grasp_pose", geometry_msgs.msg.PoseStamped,
                                                      queue_size=1, latch=True)
        self.pre_grasp_viz = rospy.Publisher("selected_pre_grasp_pose", geometry_msgs.msg.PoseStamped,
                                        queue_size=1, latch=True)
        self.grasp_to_arm_tool0 = tf_buffer.lookup_transform("arm_tool0", "gripper_equilibrium_grasp", rospy.Time(0),
                                               rospy.Duration(1)).transform

    def execute(self, userdata):
        get_points_req = GetObjectPointsRequest(
            bin_id=userdata['target_bin_id'],
            object_id=userdata['target_object_id'],
            frame_id=self.frame_id
        )
        print(get_points_req, get_points_req)
        points_response = self.get_points(get_points_req)

        if not points_response.success:
            return "aborted"

        grasp_response = self.get_grasp(points=points_response.points,
                                        mask=points_response.mask,
                                        dist_threshold=self.pre_grasp_offset)

        if not grasp_response.success:
            return "aborted"

        # NOTE: No extra filtering or ranking on our part. Just take the first one
        grasp_pose = grasp_response.poses[0]
        userdata['grasp_pose'] = grasp_pose

        # Apply additional offset for pregrasp distance
        v = qv_mult(
            quat_msg_to_vec(grasp_pose.pose.orientation), (0, 0, -self.pre_grasp_offset))
        pregrasp_pose = deepcopy(grasp_pose)
        pregrasp_pose.pose.position.x += v[0]
        pregrasp_pose.pose.position.y += v[1]
        pregrasp_pose.pose.position.z += v[2]

        userdata['pre_grasp_pose'] = pregrasp_pose

        self.pose_viz.publish(grasp_pose)
        self.pre_grasp_viz.publish(pregrasp_pose)

        return "succeeded"
