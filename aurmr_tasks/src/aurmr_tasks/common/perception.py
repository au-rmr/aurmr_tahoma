import aurmr_perception
import rospy

from smach import State
from aurmr_perception.srv import (
    ResetBin,
    ResetBinRequest,
    GetObjectPoints,
    GraspPose,
    GetObjectPointsRequest,
    GraspPoseRequest,
    CaptureObjectRequest,
)


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


class CaptureObject(State):
    def __init__(self):
        State.__init__(
            self,
            input_keys=['target_bin_id', 'target_object_id'],
            outcomes=['succeeded', 'preempted', 'aborted']
        )
        self.capture_object = rospy.ServiceProxy('/aurmr_perception/capture_object', aurmr_perception.srv.CaptureObject)

    def execute(self, userdata):
        capture_obj_req = CaptureObjectRequest(
            bin_id=userdata['target_bin_id'],
            object_id=userdata['target_object_id'],
        )
        capture_response = self.capture_object(capture_obj_req)

        if capture_response.success:
            return "succeeded"
        else:
            return "aborted"


class GetGraspPose(State):
    def __init__(self, frame_id='base_link', distance_threshold=.25):
        State.__init__(
            self,
            input_keys=['target_bin_id', 'target_object_id'],
            output_keys=['grasp_pose'],
            outcomes=['succeeded', 'preempted', 'aborted']
        )
        self.get_points = rospy.ServiceProxy('/aurmr_perception/get_object_points', GetObjectPoints)
        self.get_grasp = rospy.ServiceProxy('/aurmr_perception/init_grasp', GraspPose)
        # Crash during initialization if these aren't running so see the problem early
        self.get_points.wait_for_service(timeout=5)
        self.get_grasp.wait_for_service(timeout=5)
        self.frame_id = frame_id
        self.distance_threshold = distance_threshold

    def execute(self, userdata):
        get_points_req = GetObjectPointsRequest(
            bin_id=userdata['target_bin_id'],
            object_id=userdata['target_object_id'],
            frame_id=self.frame_id
        )
        points_response = self.get_points(get_points_req)

        if not points_response.success:
            return "aborted"

        get_grasp_req = GraspPoseRequest(
            points=points_response.points,
            dist_th=self.distance_threshold,
            pose_id=0,
            grasp_id=0,
        )
        grasp_response = self.get_grasp(get_grasp_req)

        if not grasp_response.success:
            return "aborted"
        
        userdata['grasp_pose'] = grasp_response.pose

        return "succeeded"
