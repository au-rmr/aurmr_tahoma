from copy import deepcopy

import aurmr_perception
import geometry_msgs
import std_msgs
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
from aurmr_tasks.util import add_offset

class CaptureEmptyBin(State):
    def __init__(self):
        State.__init__(self, input_keys=['target_bin_id'], outcomes=['succeeded', 'preempted', 'aborted'])
        self.reset_bin = rospy.ServiceProxy('/aurmr_perception/reset_bin', ResetBin)

    def execute(self, userdata):
        try:
            reset_bin_req = ResetBinRequest(bin_id=userdata['target_bin_id'])
            reset_response = self.reset_bin(reset_bin_req)
            if reset_response.success:
                return "succeeded"
        except Exception as e:
            rospy.logerr(e)



        return "aborted"

class CaptureEmptyPod(State):
    def __init__(self):
        State.__init__(self, input_keys=[], outcomes=['succeeded', 'aborted'])
        self.capture_empty = rospy.ServiceProxy('/aurmr_perception/capture_empty_pod', Trigger)

    def execute(self, userdata):
        try:
            res = self.capture_empty()
            if res.success:
                return "succeeded"
        except Exception as e:
            rospy.logerr(e)

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
        try:
            capture_obj_req = CaptureObjectRequest(
                bin_id=userdata['target_bin_id'],
                object_id=userdata['target_object_id'],
            )
            rospy.loginfo("in CAPTUREOBJECT" + userdata['target_bin_id'])
            capture_response = self.capture_object(capture_obj_req)

            if capture_response.success:
                return "succeeded"
        except Exception as e:
            rospy.logerr(e)

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
        try:
            capture_obj_req = CaptureObjectRequest(
                bin_id=userdata['target_bin_id'],
                object_id=userdata['target_object_id'],
            )
            rospy.loginfo("in CAPTUREOBJECT" + userdata['target_bin_id'])
            capture_response = self.capture_object(capture_obj_req)
            if capture_response.success:
                return "succeeded"
        except Exception as e:
            rospy.logerr(e)
        return "aborted"


class UpdateBin(State):
    def __init__(self):
        State.__init__(
            self,
            input_keys=['target_bin_id'],
            outcomes=['succeeded', 'preempted', 'aborted']
        )
        self.capture_object = rospy.ServiceProxy('/aurmr_perception/update_bin', aurmr_perception.srv.CaptureObject)
        self.capture_object.wait_for_service(timeout=rospy.Duration(5))

    def execute(self, userdata):
        try:
            capture_obj_req = CaptureObjectRequest(
                bin_id=userdata['target_bin_id'],
                object_id=None,
            )
            rospy.loginfo("in UPDATEBIN" + userdata['target_bin_id'])
            capture_response = self.capture_object(capture_obj_req)
            if capture_response.success:
                return "succeeded"
        except Exception as e:
            rospy.logerr(e)

        return "aborted"


class GetPoints(State):
    def __init__(self, frame_id='base_link'):
        State.__init__(
            self,
            input_keys=['target_bin_id', 'target_object_id'],
            output_keys=['points_response', 'status'],
            outcomes=['succeeded', 'aborted']
        )
        self.get_points = rospy.ServiceProxy('/aurmr_perception/get_object_points', GetObjectPoints)
        self.get_points.wait_for_service(timeout=5)
        self.frame_id = frame_id

    def execute(self, userdata):
        try:
            get_points_req = GetObjectPointsRequest(
                bin_id=userdata['target_bin_id'],
                object_id=userdata['target_object_id'],
                frame_id=self.frame_id
            )
            print(get_points_req, get_points_req)
            points_response = self.get_points(get_points_req)

            if not points_response.success:
                userdata["status"] = "pass"
                return "aborted"

            userdata['points_response'] = points_response
            return "succeeded"
        except:
            return "aborted"

class GetGraspPose(State):
    def __init__(self, tf_buffer, frame_id='base_link', pre_grasp_offset=.12):
        State.__init__(
            self,
            input_keys=['target_bin_id', 'target_object_id', 'points_response'],
            output_keys=['grasp_pose', 'pre_grasp_pose'],
            outcomes=['succeeded', 'preempted', 'aborted']
        )
        self.get_grasp = rospy.ServiceProxy('/grasp_detection/detect_grasps', DetectGraspPoses)
        # Crash during initialization if these aren't running so see the problem early
        self.get_grasp.wait_for_service(timeout=5)
        self.frame_id = frame_id
        self.pre_grasp_offset = pre_grasp_offset
        self.pose_viz = rospy.Publisher("~selected_grasp_pose", geometry_msgs.msg.PoseStamped,
                                                      queue_size=1, latch=True)
        self.pre_grasp_viz = rospy.Publisher("~selected_pre_grasp_pose", geometry_msgs.msg.PoseStamped,
                                        queue_size=1, latch=True)
        self.grasp_to_arm_tool0 = tf_buffer.lookup_transform("arm_tool0", "gripper_equilibrium_grasp", rospy.Time(0),
                                               rospy.Duration(1)).transform


    def execute(self, userdata):

        rospy.loginfo("Using perception system to get grasp pose")
        get_points_req = GetObjectPointsRequest(
            bin_id=userdata['target_bin_id'],
            object_id=userdata['target_object_id'],
            frame_id=self.frame_id
        )

        try:
            points_response = userdata["points_response"]

            if not points_response.success:
                return "aborted"

            grasp_response = self.get_grasp(points=points_response.points,
                                            mask=points_response.mask,
                                            dist_threshold=self.pre_grasp_offset, bin_id=userdata['target_bin_id'])

            if not grasp_response.success:
                return "aborted"

            # NOTE: No extra filtering or ranking on our part. Just take the first one
            # As the arm_tool0 is 26cm in length w.r.t tip of suction cup thus adding 0.26m offset
            # TODO(henrifung): Pull this number from xacro, perhaps from userdata
            grasp_pose = grasp_response.poses[0]

            # grasp_pose = add_offset(-0.26, grasp_pose)
            grasp_pose = add_offset(-0.19, grasp_pose)

            userdata['grasp_pose'] = grasp_pose

            # adding 0.12m offset for pre grasp pose to prepare it for grasp pose which is use to pick the object
            pregrasp_pose = add_offset(-self.pre_grasp_offset, grasp_pose)

            userdata['pre_grasp_pose'] = pregrasp_pose

            self.pose_viz.publish(grasp_pose)
            self.pre_grasp_viz.publish(pregrasp_pose)

            return "succeeded"
        except:
            return "aborted"

class GetHapticSearchPose(State):
    """Using capacitive haptic search, get and the next pose to userdata"""
    def __init__(self, timeout: float = 5):
        State.__init__(
            self,
            input_keys=[],
            output_keys=['pose_delta'],
            outcomes=['succeeded', 'preempted', 'aborted']
        )
        self._desired_pose = geometry_msgs.msg.PoseStamped()
        self._get_grasp = rospy.Subscriber("/haptic_search/proposed_pose", geometry_msgs.msg.PoseStamped, self._pose_cb)
        self._timeout = timeout
        self._last_pose = None
        self._clearance = 0.005

    def _pose_cb(self, msg: geometry_msgs.msg.PoseStamped):
        self._desired_pose.header.stamp = rospy.Time.now()
        self._desired_pose.header.frame_id = msg.header.frame_id
        self._desired_pose.pose.position.x = msg.pose.position.x
        self._desired_pose.pose.position.y = msg.pose.position.y
        self._desired_pose.pose.position.z = msg.pose.position.z + self._clearance
        self._desired_pose.pose.orientation.x = msg.pose.orientation.x
        self._desired_pose.pose.orientation.y = msg.pose.orientation.y
        self._desired_pose.pose.orientation.z = msg.pose.orientation.z
        self._desired_pose.pose.orientation.w = msg.pose.orientation.w


    def execute(self, userdata):
        start_time = rospy.Time.now()
        rate = rospy.Rate(50)
        try:
            while not rospy.is_shutdown():
                if self._desired_pose is not None:
                    userdata["pose_delta"] = self._desired_pose
                    rospy.loginfo(f"the pose\n{self._desired_pose}")

                    return "succeeded"

                if self._timeout is not None:
                    if (rospy.Time.now() - start_time).to_sec() > self._timeout:
                        rospy.logwarn("No pose received from haptic search")
                        return "aborted"
            rate.sleep()
        except:
            return "aborted"

class CheckCapAlignment(State):
    def __init__(self, timeout=20):
        State.__init__(self,
            outcomes = ["aligned", "misaligned", "aborted", "preempted"],
            input_keys = ["search_start"]
        )
        self._alignment_sub = rospy.Subscriber("/haptic_search/alignment", std_msgs.msg.Bool, self._alignment_cb)
        self._aligned = False
        self._timeout = timeout

    def _alignment_cb(self, msg):
        self._aligned = msg.data

    def execute(self, userdata):
        if rospy.get_time() - userdata["search_start"] > self._timeout:
            return "aborted"

        if self._aligned:
            return "aligned"
        else:
            return "misaligned"
