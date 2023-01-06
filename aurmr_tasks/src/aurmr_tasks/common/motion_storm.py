import copy
import geometry_msgs.msg
import rospy
import math
import actionlib
from sensor_msgs.msg import JointState
from smach import State, StateMachine
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from smach import State
from std_msgs.msg import Header
from aurmr_tasks.interaction import prompt_for_confirmation
from aurmr_tasks.util import apply_offset_to_pose
from aurmr_tasks import interaction
from aurmr_perception.util import I_QUAT, ROT_90_Z_QUAT
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, Bool
from geometry_msgs.msg import PoseStamped, WrenchStamped
from robotiq_2f_gripper_control.msg import vacuum_gripper_input as VacuumGripperStatus
from control_msgs.msg import FollowJointTrajectoryAction, GripperCommandAction, GripperCommandGoal
import tf2_ros

class MoveEndEffectorToPoseStorm(State):
    def __init__(self, default_pose=None):
        State.__init__(self, input_keys=['pose'], outcomes=['succeeded', 'aborted'])
        self.default_pose = default_pose
        # self.target_pose_visualizer = rospy.Publisher("end_effector_target", geometry_msgs.msg.PoseStamped, queue_size=1, latch=True)
        self.goal_finished = False

        self.GOAL_POSE = '/goal_pose'
        self.STROM_RESULT = '/storm_info/result'
        self.ACTIVATE_CONTROL = '/activate_control'
        rospy.Subscriber(self.STROM_RESULT, Bool, self.callback)
        self.goal_pub = rospy.Publisher(self.GOAL_POSE, geometry_msgs.msg.PoseStamped, queue_size=1)
        self.AC_pub = rospy.Publisher(self.ACTIVATE_CONTROL, Bool, queue_size=1)
        
    
    def callback(self, msg):
        self.goal_finished = msg.data
        #print("Get STROM_RESULT msg", self.goal_finished )

    def execute(self, userdata): 
        if self.default_pose:
            pose = self.default_pose
        else:
            pose = userdata["pose"]
        # print("Start executing storm")
        # self.target_pose_visualizer.publish(pose)
        self.goal_finished = False
        
        time_out, steps = 50, 0.0
        while not(self.goal_finished):
            # print("The robot is still moving to: ", pose)
            self.goal_pub.publish(pose)
            self.AC_pub.publish(Bool(data=True))
            # print("Goal has not reached yet")
            rospy.sleep(0.1)
            steps = steps + 0.1
            if steps>time_out:
                # self.AC_pub.publish(Bool(data=False))
                rospy.loginfo("Time_out in normal movement")
                # early_stop = True
                break
        
        self.AC_pub.publish(Bool(data=False))
        # rospy.loginfo("Finish a waypoint", pose)
        success = self.goal_finished
               
        if success:
            return "succeeded"
        else:
            return "succeeded"

class MoveEndEffectorInLineStorm(State):
    def __init__(self, start_pose=None, goal_pose=None, use_force=False, use_gripper=False, use_curr_pose=False):
        State.__init__(self,outcomes=['succeeded', 'aborted'])
        self.use_curr_pose = use_curr_pose
        if self.use_curr_pose:
            self.start_pose = None
        else:
            self.start_pose = start_pose
        self.goal_pose = goal_pose
        self.use_force = use_force
        self.use_gripper = use_gripper
        self.goal_finished = False
        self.object_detected = False
        self.force_msg = 0
        self.torque_msg = 0
        
        self.GOAL_POSE = '/goal_pose'
        self.STROM_RESULT = '/storm_info/result'
        self.ACTIVATE_CONTROL = '/activate_control'
        self.GRIPPER_ACTION_SERVER = '/gripper_controller/gripper_cmd'
        self.CURRENT_POSE = '/goal_pose/storm'

        self._gripper_client = actionlib.SimpleActionClient(self.GRIPPER_ACTION_SERVER, GripperCommandAction)
        self.goal_pub = rospy.Publisher(self.GOAL_POSE, PoseStamped, queue_size=1)
        self.AC_pub = rospy.Publisher(self.ACTIVATE_CONTROL, Bool, queue_size=1)
        self.goal_listener = rospy.Subscriber(self.STROM_RESULT, Bool, self.goal_finish_cb)
        self.wrench_listener = rospy.Subscriber("/wrench", WrenchStamped, self.wrench_cb)
        self.gripper_status_listener = rospy.Subscriber("/gripper_control/status", VacuumGripperStatus, self.gripper_status_cb)
        self.curr_pose_listener = rospy.Subscriber(self.CURRENT_POSE, PoseStamped, self.curr_pose_cb)
        
    def curr_pose_cb(self, msg: PoseStamped):
        if self.use_curr_pose:
            self.start_pose = msg

    def close_gripper(self, return_before_done=False):
        goal = GripperCommandGoal()
        goal.command.position = 0.83
        goal.command.max_effort = 1
        self._gripper_client.send_goal(goal)
        # rospy.loginfo("Waiting for gripper" + str(return_before_done))
        if not return_before_done:
            rospy.loginfo("Waiting for gripper. \n")
            self._gripper_client.wait_for_result()

    def open_gripper(self, return_before_done=False):
        goal = GripperCommandGoal()
        goal.command.position = 0
        goal.command.max_effort = 1
        self._gripper_client.send_goal(goal)
        if not return_before_done:
            self._gripper_client.wait_for_result()

    def wrench_cb(self, msg: WrenchStamped):
        self.force_msg = math.sqrt(msg.wrench.force.x**2 + msg.wrench.force.y**2+ msg.wrench.force.z**2)
        self.torque_msg = math.sqrt(msg.wrench.torque.x**2 + msg.wrench.torque.y**2+ msg.wrench.torque.z**2)

    def gripper_status_cb(self, msg: VacuumGripperStatus):
        self.object_detected = (msg.gPO < 95)

    def goal_finish_cb(self, msg):
        self.goal_finished = msg.data
        #print("Get STROM_RESULT msg", self.goal_finished )

    def execute(self, userdata):
        if self.start_pose is None:
            print('Waiting for start_pose')
            rospy.sleep(0.5)
        poses = []
        diff_x = self.goal_pose.pose.position.x - self.start_pose.pose.position.x
        diff_y = self.goal_pose.pose.position.y - self.start_pose.pose.position.y
        diff_z = self.goal_pose.pose.position.z - self.start_pose.pose.position.z
        segment_num = int(((diff_x**2 + diff_y**2 + diff_z**2)**(1/2))/0.02)
        # print("number of dividen:", segment_num)
        for i in range(segment_num):
            pose = copy.deepcopy(self.start_pose)
            pose.pose.position.x += (i/segment_num)*diff_x
            pose.pose.position.y += (i/segment_num)*diff_y
            pose.pose.position.z += (i/segment_num)*diff_z
            poses.append(pose)
        poses.append(self.goal_pose)
        
        time_out = 5
        force_limit = 50
        if self.use_gripper:
            self.close_gripper(return_before_done=True)   
        for pose in poses:
            self.goal_finished = False
            steps = 0.0
            while not(self.goal_finished):
                self.goal_pub.publish(pose)
                self.AC_pub.publish(Bool(data=True))
                if self.use_force and self.force_msg > force_limit:
                    self.AC_pub.publish(Bool(data=False))
                    rospy.loginfo("Stopping movement due to force feedback")
                    return "succeeded"
                elif self.use_gripper and self.object_detected:
                    self.AC_pub.publish(Bool(data=False))
                    rospy.loginfo("Object detected")
                    return "succeeded"
                rospy.sleep(0.05)
                steps = steps + 0.05
                if steps>time_out:
                    # self.AC_pub.publish(Bool(data=False))
                    rospy.loginfo("Time_out in cartesian movement")
                    # early_stop = True
                    break

            self.AC_pub.publish(Bool(data=False))
            # print("Finish waypoint: ", pose)
        # if self.use_gripper:
        #     self.open_gripper(return_before_done=True)

        if self.goal_finished:
            return "succeeded"
        else:
            return "succeeded"


class MoveEndEffectorInLineByDistStorm(State):
    def __init__(self, dist=[0, 0, 0], quat=[0, 0.7071068, 0, 0.7071068 ], use_force=False, use_gripper=False):
        State.__init__(self,outcomes=['succeeded', 'aborted'])
        self.dist = dist
        self.quat = quat
        self.start_pose = None
        self.use_force = use_force
        self.use_gripper = use_gripper
        self.goal_finished = False
        self.object_detected = False
        self.force_msg = 0
        self.torque_msg = 0
        
        self.GOAL_POSE = '/goal_pose'
        self.STROM_RESULT = '/storm_info/result'
        self.ACTIVATE_CONTROL = '/activate_control'
        self.GRIPPER_ACTION_SERVER = '/gripper_controller/gripper_cmd'
        self.CURRENT_POSE = '/goal_pose/storm'

        self._gripper_client = actionlib.SimpleActionClient(self.GRIPPER_ACTION_SERVER, GripperCommandAction)
        self.goal_pub = rospy.Publisher(self.GOAL_POSE, PoseStamped, queue_size=1)
        self.AC_pub = rospy.Publisher(self.ACTIVATE_CONTROL, Bool, queue_size=1)
        self.goal_listener = rospy.Subscriber(self.STROM_RESULT, Bool, self.goal_finish_cb)
        self.wrench_listener = rospy.Subscriber("/wrench", WrenchStamped, self.wrench_cb)
        self.gripper_status_listener = rospy.Subscriber("/gripper_control/status", VacuumGripperStatus, self.gripper_status_cb)
        self.curr_pose_listener = rospy.Subscriber(self.CURRENT_POSE, PoseStamped, self.curr_pose_cb)
        
    def curr_pose_cb(self, msg: PoseStamped):

        self.start_pose = msg

    def close_gripper(self, return_before_done=False):
        goal = GripperCommandGoal()
        goal.command.position = 0.83
        goal.command.max_effort = 1
        self._gripper_client.send_goal(goal)
        # rospy.loginfo("Waiting for gripper" + str(return_before_done))
        if not return_before_done:
            rospy.loginfo("Waiting for gripper. \n")
            self._gripper_client.wait_for_result()

    def open_gripper(self, return_before_done=False):
        goal = GripperCommandGoal()
        goal.command.position = 0
        goal.command.max_effort = 1
        self._gripper_client.send_goal(goal)
        if not return_before_done:
            self._gripper_client.wait_for_result()

    def wrench_cb(self, msg: WrenchStamped):
        self.force_msg = math.sqrt(msg.wrench.force.x**2 + msg.wrench.force.y**2+ msg.wrench.force.z**2)
        self.torque_msg = math.sqrt(msg.wrench.torque.x**2 + msg.wrench.torque.y**2+ msg.wrench.torque.z**2)

    def gripper_status_cb(self, msg: VacuumGripperStatus):
        self.object_detected = (msg.gPO < 95)

    def goal_finish_cb(self, msg):
        self.goal_finished = msg.data
        #print("Get STROM_RESULT msg", self.goal_finished )

    def execute(self, userdata):
        if self.start_pose is None:
            print('Waiting for start_pose')
            rospy.sleep(0.5)
        poses = []
        diff_x = self.dist[0]
        diff_y = self.dist[1]
        diff_z = self.dist[2]
        segment_num = int(((diff_x**2 + diff_y**2 + diff_z**2)**(1/2))/0.02)
        # print("number of dividen:", segment_num)
        for i in range(segment_num):
            pose = copy.deepcopy(self.start_pose)
            pose.pose.orientation = Quaternion(self.quat[0], self.quat[1], self.quat[2], self.quat[3]) # vetical to pod face
            pose.pose.position.x += (i/segment_num)*diff_x
            pose.pose.position.y += (i/segment_num)*diff_y
            pose.pose.position.z += (i/segment_num)*diff_z
            poses.append(pose)
 
        
        time_out = 5
        force_limit = 30
        if self.use_gripper:
            self.close_gripper(return_before_done=True)   
        for pose in poses:
            self.goal_finished = False
            steps = 0.0
            while not(self.goal_finished):
                self.goal_pub.publish(pose)
                self.AC_pub.publish(Bool(data=True))
                if self.use_force and self.force_msg > force_limit:
                    self.AC_pub.publish(Bool(data=False))
                    rospy.loginfo("Stopping movement due to force feedback")
                    return "succeeded"
                elif self.use_gripper and self.object_detected:
                    self.AC_pub.publish(Bool(data=False))
                    rospy.loginfo("Object detected")
                    return "succeeded"
                rospy.sleep(0.05)
                steps = steps + 0.05
                if steps>time_out:
                    # self.AC_pub.publish(Bool(data=False))
                    rospy.loginfo("Time_out in cartesian movement")
                    # early_stop = True
                    break

            self.AC_pub.publish(Bool(data=False))
            # print("Finish waypoint: ", pose)
        # if self.use_gripper:
        #     self.open_gripper(return_before_done=True)

        if self.goal_finished:
            return "succeeded"
        else:
            return "succeeded"



class CloseGripperStorm(State):
    def __init__(self, return_before_done=False):
        State.__init__(self, outcomes=['succeeded', 'preempted', 'aborted'])
        self.return_before_done = return_before_done
        self.GRIPPER_ACTION_SERVER = '/gripper_controller/gripper_cmd'
        self._gripper_client = actionlib.SimpleActionClient(self.GRIPPER_ACTION_SERVER, GripperCommandAction)

    def close_gripper(self, return_before_done=False):
        goal = GripperCommandGoal()
        goal.command.position = 0.83
        goal.command.max_effort = 1
        self._gripper_client.send_goal(goal)
        # rospy.loginfo("Waiting for gripper" + str(return_before_done))
        if not return_before_done:
            rospy.loginfo("Waiting for gripper. \n")
            self._gripper_client.wait_for_result()


    def execute(self, ud):
        self.close_gripper(return_before_done=self.return_before_done)
        return "succeeded"


class OpenGripperStorm(State):
    def __init__(self, return_before_done=False):
        State.__init__(self, outcomes=['succeeded', 'preempted', 'aborted'])
        self.return_before_done = return_before_done
        self.GRIPPER_ACTION_SERVER = '/gripper_controller/gripper_cmd'
        self._gripper_client = actionlib.SimpleActionClient(self.GRIPPER_ACTION_SERVER, GripperCommandAction)

    def open_gripper(self, return_before_done=False):
        goal = GripperCommandGoal()
        goal.command.position = 0
        goal.command.max_effort = 1
        self._gripper_client.send_goal(goal)
        if not return_before_done:
            self._gripper_client.wait_for_result()

    def execute(self, ud):
        self.open_gripper(return_before_done=self.return_before_done)
        return "succeeded"

class ToStormFrame(State):
    def __init__(self):
        State.__init__(self, input_keys=['pose'],output_keys=['pose'], outcomes=['succeeded'])
        self.tmp_ros_frame = 'arm_base_link'
        self.tf2_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer)
        self.target_pose_visualizer = rospy.Publisher("storm_target", geometry_msgs.msg.PoseStamped, queue_size=1, latch=True)


    def execute(self, userdata):
        pose = userdata["pose"]
        goal_in_tmp_frame = self.tf2_buffer.transform(pose, self.tmp_ros_frame, rospy.Duration(1))
        goal_in_planning_frame = copy.deepcopy(goal_in_tmp_frame)
        goal_in_planning_frame.pose.position.x = goal_in_tmp_frame.pose.position.y
        goal_in_planning_frame.pose.position.y = -goal_in_tmp_frame.pose.position.x
        goal_in_planning_frame.pose.position.z = goal_in_tmp_frame.pose.position.z
        self.target_pose_visualizer.publish(goal_in_tmp_frame)
        goal_in_planning_frame.pose.orientation = Quaternion(0, 0.7071068, 0, 0.7071068) # vetical to pod face
        userdata["pose"] = goal_in_planning_frame
        return "succeeded"

