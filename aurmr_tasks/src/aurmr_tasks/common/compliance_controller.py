from aurmr_tasks.util import apply_offset_to_pose
import rospy
from smach import State
from controller_manager_msgs.srv import SwitchController
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped, Point, Quaternion

# Define the state
class SwitchControllers(State):
    def __init__(self, start_controllers, stop_controllers):
        State.__init__(self, outcomes=['succeeded', 'aborted'])
        self.start_controllers = start_controllers
        self.stop_controllers = stop_controllers

    def execute(self, userdata):
        rospy.wait_for_service('/controller_manager/switch_controller')
        try:
            # Create a service proxy
            switch_controller_service = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)

            # Call the service with the provided arguments
            response = switch_controller_service(self.start_controllers, self.stop_controllers, 1, False, 0.0)
            print(f"Service call response: {response}")
            
            return 'succeeded'
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
            return 'aborted'


class MoveToOffset(State):
    def __init__(self, robot, offset, frame_id):
        State.__init__(self, outcomes=['succeeded', 'aborted'])
        self.robot = robot
        self.offset = offset
        self.frame_id = frame_id

    def execute(self, userdata):
        rospy.sleep(1)

        target_frame = 'arm_base_link'

        print(f'frame id: {self.frame_id}')
        current = self.robot.move_group.get_current_pose()
        target_pose = apply_offset_to_pose(current, self.offset, self.frame_id, self.robot.tf2_buffer)

        target_pose = self.robot.tf2_buffer.transform(target_pose, target_frame, rospy.Duration(1))

        print(target_pose)
        input('moving to offset')

        # Publish the target pose
        pub = rospy.Publisher('/ur_cartesian_compliance_controller/target_frame', PoseStamped, queue_size=10)
        pub.publish(target_pose)

        return 'succeeded'