from aurmr_tasks.util import apply_offset_to_pose, all_close
import rospy
from smach import State
from geometry_msgs.msg import PoseStamped
import time

class MoveToOffset(State):
    def __init__(self, robot, offset, frame_id, detect_object):
        super().__init__(outcomes=['succeeded', 'aborted'])
        self.robot = robot
        self.offset = offset
        self.frame_id = frame_id
        self.detect_object = detect_object
        self.pub = rospy.Publisher('/ur_cartesian_compliance_controller/target_frame', PoseStamped, queue_size=10)
        self.target_frame = 'arm_base_link'

    def execute(self, userdata):
        self.close_robot_gripper()
        target_pose = self.calculate_target_pose()
        self.publish_target_pose(target_pose)
        return self.wait_for_target_pose_or_timeout(target_pose)

    def close_robot_gripper(self):
        self.robot.close_gripper(return_before_done=True)

    def calculate_target_pose(self):
        current = self.robot.move_group.get_current_pose()
        target_pose = apply_offset_to_pose(current, self.offset, self.frame_id, self.robot.tf2_buffer)
        return self.robot.tf2_buffer.transform(target_pose, self.target_frame, rospy.Duration(1))

    def publish_target_pose(self, target_pose):
        self.pub.publish(target_pose)

    def wait_for_target_pose_or_timeout(self, target_pose):
        timeout = 5.0
        start_time = time.time()
        while ((time.time() - start_time) < timeout) and not rospy.is_shutdown():
            if self.has_reached_target_pose(target_pose):
                return 'succeeded'
            rospy.sleep(0.005)
        return 'succeeded'

    def has_reached_target_pose(self, target_pose):
        current_pose = self.robot.move_group.get_current_pose()
        current_pose = self.robot.tf2_buffer.transform(current_pose, self.target_frame, rospy.Duration(1))
        return (self.detect_object and self.robot.object_detected) or all_close(target_pose, current_pose, 0.03)
