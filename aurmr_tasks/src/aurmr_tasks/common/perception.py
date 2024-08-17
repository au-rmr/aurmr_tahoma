from copy import deepcopy

import aurmr_perception
import geometry_msgs
import rospy

from smach import State
from aurmr_perception.srv import (
    DetectGraspPoses,
    DetectGraspPosesTable,
    ResetBin,
    ResetBinRequest,
    GetObjectPoints,
    GetObjectMasks,
    GetObjectPointsRequest,
    GetObjectMasksRequest,
    CaptureObjectRequest,
    CaptureTableRequest
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

class CaptureTable(State):
    def __init__(self):
        State.__init__(
            self,
            input_keys=[],
            outcomes=['succeeded', 'aborted']
        )
        self.capture_table = rospy.ServiceProxy('/aurmr_perception/capture_table', aurmr_perception.srv.CaptureTable)
        self.capture_table.wait_for_service(timeout=rospy.Duration(5))

    def execute(self, userdata):
        try:
            capture_table_req = CaptureTableRequest()
            rospy.loginfo("in CAPTURETABLE")
            capture_response = self.capture_table(capture_table_req)

            if capture_response.success:
                return "succeeded"
            else:
                return "aborted"
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
            else:
                return "aborted"
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
            else:
                return "aborted"
        except Exception as e:
            rospy.logerr(e)

        return "aborted"


class GetPoints(State):
    def __init__(self, frame_id='base_link'):
        State.__init__(
            self,
            input_keys=['target_bin_id', 'target_object_id'],
            output_keys=['points_response'],
            outcomes=['succeeded', 'aborted']
        )
        self.get_points = rospy.ServiceProxy('/aurmr_perception/get_object_points', GetObjectPoints)
        self.get_points.wait_for_service(timeout=5)
        self.frame_id = frame_id

    def execute(self, userdata):
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

class GetGraspPose(State):
    def __init__(self, tf_buffer, frame_id='base_link', pre_grasp_offset=.12):
        State.__init__(
            self,
            input_keys=['target_bin_id', 'target_object_id', 'points_response', 'human_grasp_pose'],
            output_keys=['grasp_pose', 'pre_grasp_pose', 'status', 'human_grasp_pose'],
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

    def add_offset(self, offset, grasp_pose):
        v = qv_mult(
            quat_msg_to_vec(grasp_pose.pose.orientation), (0, 0, offset))
        offset_pose = deepcopy(grasp_pose)
        offset_pose.pose.position.x += v[0]
        offset_pose.pose.position.y += v[1]
        offset_pose.pose.position.z += v[2]
        return offset_pose

    def execute(self, userdata):
        try:
            if 'human_grasp_pose' in userdata and userdata['human_grasp_pose'] != None:
                rospy.loginfo("Using human provided grasp pose")
                grasp_pose = userdata['human_grasp_pose']
                userdata["human_grasp_pose"] = None

            else:
                rospy.loginfo("Using perception system to get grasp pose")

                points_response = userdata['points_response']
                grasp_response = self.get_grasp(points=points_response.points,
                                                mask=points_response.mask,
                                                dist_threshold=self.pre_grasp_offset, bin_id=userdata['target_bin_id'])

                if not grasp_response.success:
                    userdata["status"] = "pass"
                    return "aborted"

                # NOTE: No extra filtering or ranking on our part. Just take the first one
                # As the arm_tool0 is 35cm in length w.r.t tip of suction cup thus adding 0.35m offset
                grasp_pose = grasp_response.poses[0]

            grasp_pose = self.add_offset(-0.35, grasp_pose)

            userdata['grasp_pose'] = grasp_pose

            # adding 0.12m offset for pre grasp pose to prepare it for grasp pose which is use to pick the object
            pregrasp_pose = self.add_offset(-self.pre_grasp_offset, grasp_pose)

            userdata['pre_grasp_pose'] = pregrasp_pose

            self.pose_viz.publish(grasp_pose)
            self.pre_grasp_viz.publish(pregrasp_pose)

            userdata["status"] = "picking"
            return "succeeded"
        except Exception as e:
            rospy.logerr(e)
        return "aborted"

class GetTableGraspPose(State):
    def __init__(self, frame_id='base_link', pre_grasp_offset=.12):
        State.__init__(
            self,
            output_keys=['grasp_pose', 'pre_grasp_pose', 'status'],
            outcomes=['succeeded', 'preempted', 'aborted']
        )
        self.get_masks = rospy.ServiceProxy('/aurmr_perception/get_masks_table', GetObjectMasks)
        self.get_grasp = rospy.ServiceProxy('/grasp_detection/detect_grasps_table', DetectGraspPosesTable)
        # Crash during initialization if these aren't running so see the problem early
        self.get_masks.wait_for_service(timeout=5)
        self.get_grasp.wait_for_service(timeout=5)
        self.frame_id = frame_id
        self.pre_grasp_offset = pre_grasp_offset
        self.pose_viz = rospy.Publisher("~selected_grasp_pose", geometry_msgs.msg.PoseStamped,
                                                      queue_size=1, latch=True)
        self.pre_grasp_viz = rospy.Publisher("~selected_pre_grasp_pose", geometry_msgs.msg.PoseStamped,
                                        queue_size=1, latch=True)

    def add_offset(self, offset, grasp_pose):
        v = qv_mult(
            quat_msg_to_vec(grasp_pose.pose.orientation), (0, 0, offset))
        offset_pose = deepcopy(grasp_pose)
        offset_pose.pose.position.x += v[0]
        offset_pose.pose.position.y += v[1]
        offset_pose.pose.position.z += v[2]
        return offset_pose

    def execute(self, userdata):
        try:
            rospy.loginfo("Using perception system to get grasp pose")
            get_mask_req = GetObjectMasksRequest()
            masks_response = self.get_masks(get_mask_req)

            if not masks_response.success:
                userdata["status"] = "pass"
                return "aborted"

            grasp_response = self.get_grasp(mask=masks_response.mask,
                                            dist_threshold=self.pre_grasp_offset)

            if not grasp_response.success:
                userdata["status"] = "pass"
                return "aborted"

            # NOTE: No extra filtering or ranking on our part. Just take the first one
            # As the arm_tool0 is 20cm in length w.r.t tip of suction cup thus adding 0.2m offset
            grasp_pose = grasp_response.poses[0]

            grasp_pose = self.add_offset(-0.35, grasp_pose)

            userdata['grasp_pose'] = grasp_pose

            # adding 0.12m offset for pre grasp pose to prepare it for grasp pose which is use to pick the object
            pregrasp_pose = self.add_offset(-self.pre_grasp_offset, grasp_pose)

            userdata['pre_grasp_pose'] = pregrasp_pose
            self.pose_viz.publish(grasp_pose)
            self.pre_grasp_viz.publish(pregrasp_pose)

            userdata["status"] = "picking"
            return "succeeded"
        except Exception as e:
            rospy.logerr(e)
        return "aborted"
