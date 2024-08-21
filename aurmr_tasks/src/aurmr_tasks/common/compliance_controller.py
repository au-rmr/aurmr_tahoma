from aurmr_tasks.util import apply_offset_to_pose, all_close
import rospy
from smach import State
from geometry_msgs.msg import PoseStamped
import time

import numpy as np
from aurmr_tasks.util import pose_dist


from aurmr_tasks.common.visualization_utils import EEFVisualization
class MoveToOffset(State):
    def __init__(self, robot, offset, frame_id, detect_object, idle_timeout=3):
        super().__init__(outcomes=['succeeded', 'aborted'],  input_keys=['offset_moved_in'], output_keys=['offset_moved_out'])
        self.robot = robot
        self.offset = offset
        self.frame_id = frame_id
        self.detect_object = detect_object
        self.pub = rospy.Publisher('/ur_cartesian_compliance_controller/target_frame', PoseStamped, queue_size=10)
        self.target_frame = 'arm_base_link'
        self.vis = EEFVisualization()
        self.vis_out = EEFVisualization("/visualization_eef_target", (0, 1, 0))
        self.idle_timeout = idle_timeout

    def stop(self):
        start_pose = self.robot.move_group.get_current_pose()
        start_pose.header.stamp = rospy.Time(0)
        start_pose = self.robot.tf2_buffer.transform(start_pose, self.target_frame, rospy.Duration(1))
        self.pub.publish(start_pose)


    def execute(self, userdata):
        self.close_robot_gripper()
        start_pose = self.current_pose_in_target_frame()

        if self.offset is None:
            offset = userdata.offset_moved_in
        else:
            offset = self.offset

        target_pose = self.calculate_target_pose(start_pose, offset)
        self.robot.force_values_init = True

        self.vis.visualize_eef(target_pose)
        self.publish_target_pose(target_pose)

        result = self.wait_until_stopped(0.001, self.idle_timeout)

        current_pose = self.current_pose_in_target_frame()
        current_pose = current_pose.pose
        start_pose = start_pose.pose
        offset_moved = np.array([start_pose.position.x, start_pose.position.y, start_pose.position.z])
        offset_moved = offset_moved - np.array([current_pose.position.x, current_pose.position.y, current_pose.position.z])

        userdata.offset_moved_out = offset_moved

        return result

    def close_robot_gripper(self):
        self.robot.close_gripper(return_before_done=True)

    def calculate_target_pose(self, start_pose, offset):
        target_pose = apply_offset_to_pose(start_pose, offset, self.frame_id, self.robot.tf2_buffer)
        return self.robot.tf2_buffer.transform(target_pose, self.target_frame, rospy.Duration(1))

    def publish_target_pose(self, target_pose):
        self.pub.publish(target_pose)


    def wait_until_stopped(self, min_delta, idle_timeout, timeout=20):
        """
        Return if the end-effector is moving only marginally for a certain amount of time or if an object has been detected
        """

        start_time = time.time()
        previous_pose = self.current_pose_in_target_frame()
        last_moved = time.time()

        while ((time.time() - start_time) < timeout) and not rospy.is_shutdown():
            current_pose = self.current_pose_in_target_frame()
            translation_delta, cos_phi_half = pose_dist(current_pose, previous_pose)
            previous_pose = current_pose

            if translation_delta > min_delta:
                last_moved = time.time()

            if ((time.time() - last_moved) >= idle_timeout):
                self.stop()
                return 'succeeded'

            if self.check_object_condition():
                self.stop()
                return 'succeeded'

            rospy.sleep(0.05)

        rospy.logwarn("Global timeout.")

        return 'succeeded'



    def current_pose_in_target_frame(self):
        current_pose = self.robot.move_group.get_current_pose()
        current_pose.header.stamp = rospy.Time(0)
        current_pose = self.robot.tf2_buffer.transform(current_pose, self.target_frame, rospy.Duration(1))

        return current_pose


    def check_object_condition(self):
        return (self.detect_object and self.robot.check_gripper_item())




from std_srvs.srv import Trigger

class ZeroFT(State):
    def __init__(self):
        State.__init__(self, outcomes=["succeeded", "aborted"])

    def execute(self, _userdata):
        service_name = "/ur_hardware_interface/zero_ftsensor"

        rospy.wait_for_service(service_name)
        try:
            zero_ftsensor_service = rospy.ServiceProxy(service_name, Trigger)
            response = zero_ftsensor_service()
            if response.success:
                return 'succeeded'
            else:
                return 'aborted'

        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
            return 'aborted'
