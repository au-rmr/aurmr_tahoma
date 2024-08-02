from aurmr_tasks.util import apply_offset_to_pose, all_close
import rospy
from smach import State
from geometry_msgs.msg import PoseStamped
import time

from aurmr_tasks.util import pose_dist


from aurmr_tasks.common.visualization_utils import EEFVisualization

class MoveToOffset(State):
    def __init__(self, robot, offset, frame_id, detect_object):
        super().__init__(outcomes=['succeeded', 'aborted'],  output_keys=['starting_grasp_pose_out'], input_keys=['target_pose_in'])
        self.robot = robot
        self.offset = offset
        self.frame_id = frame_id
        self.detect_object = detect_object
        self.pub = rospy.Publisher('/ur_cartesian_compliance_controller/target_frame', PoseStamped, queue_size=10)
        self.target_frame = 'arm_base_link'
        self.vis = EEFVisualization()
        self.vis_out = EEFVisualization("/visualization_eef_target", (0, 1, 0))

    def stop(self):
        start_pose = self.robot.move_group.get_current_pose()
        start_pose.header.stamp = rospy.Time(0)
        start_pose = self.robot.tf2_buffer.transform(start_pose, self.target_frame, rospy.Duration(1))
        self.pub.publish(start_pose)


    def execute(self, userdata):
        self.close_robot_gripper()
        start_pose = self.robot.move_group.get_current_pose()
        start_pose.header.stamp = rospy.Time(0)
        start_pose = self.robot.tf2_buffer.transform(start_pose, self.target_frame, rospy.Duration(1))

        userdata.starting_grasp_pose_out = start_pose
        rospy.logdebug("setting starting pose to ", start_pose)
        self.vis_out.visualize_eef(start_pose)

        if userdata.target_pose_in:
            target_pose = userdata.target_pose_in
        else:
            target_pose = self.calculate_target_pose(start_pose)
            self.robot.force_values_init = True

        self.vis.visualize_eef(target_pose)
        self.publish_target_pose(target_pose)

        return self.wait_until_stopped(0.01)

        #return self.wait_for_target_pose_or_timeout(target_pose)

    def close_robot_gripper(self):
        self.robot.close_gripper(return_before_done=True)

    def calculate_target_pose(self, start_pose):
        target_pose = apply_offset_to_pose(start_pose, self.offset, self.frame_id, self.robot.tf2_buffer)
        return self.robot.tf2_buffer.transform(target_pose, self.target_frame, rospy.Duration(1))

    def publish_target_pose(self, target_pose):
        self.pub.publish(target_pose)

    def wait_for_target_pose_or_timeout(self, target_pose):
        timeout = 20.0
        start_time = time.time()
        
        while ((time.time() - start_time) < timeout) and not rospy.is_shutdown():
            if self.has_reached_target_pose(target_pose):
                self.stop()
                return 'succeeded'
            rospy.sleep(0.005)
        return 'succeeded'
    

    def wait_until_stopped(self, min_delta, idle_timeout=1, timeout=20):
        """
        Return if the end-effector is moving only marginally for a certain amount of time or if an object has been detected
        """

        start_time = time.time()
        previous_pose = self.current_pose_in_target_frame()
        last_moved = time.time()

        while ((time.time() - start_time) < timeout) and not rospy.is_shutdown():
            translation_delta, cos_phi_half = pose_dist(self.current_pose_in_target_frame(), previous_pose)

            print(translation_delta)

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


    def has_reached_target_pose(self, target_pose):
        current_pose = self.robot.move_group.get_current_pose()
        current_pose.header.stamp = rospy.Time(0)
        current_pose = self.robot.tf2_buffer.transform(current_pose, self.target_frame, rospy.Duration(1))
        return (self.detect_object and self.robot.check_gripper_item()) or all_close(target_pose, current_pose, 0.03)




from std_srvs.srv import Trigger

class ZeroFT(State):
    def __init__(self):
        State.__init__(self, outcomes=["succeeded", "aborted"])

    def execute(self, _userdata):
        service_name = "/ur_hardware_interface/zero_ftsensor"

        rospy.wait_for_service(service_name)
        try:
            # Create a service proxy
            zero_ftsensor_service = rospy.ServiceProxy(service_name, Trigger)

            # Call the service with the provided arguments
            response = zero_ftsensor_service()
            print(f"Service call response: {response}")
            if response.success:
                return 'succeeded'
            else:
                return 'aborted'

        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
            return 'aborted'