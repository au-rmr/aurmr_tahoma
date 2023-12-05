import rospy
import math
import actionlib
from sensor_msgs.msg import JointState
from smach import State, StateMachine
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from smach import State
from std_msgs.msg import Header
from aurmr_perception.util import I_QUAT, ROT_90_Z_QUAT
from geometry_msgs.msg import PoseStamped, WrenchStamped


class ClearCollisionGeometry(State):
    def __init__(self, robot):
        State.__init__(self, outcomes=["succeeded", "aborted"])
        self.robot = robot

    def execute(self, ud):
        self.robot.scene.clear()
        start = rospy.get_time()
        seconds = rospy.get_time()
        timeout = 5.0
        while (seconds - start < timeout) and not rospy.is_shutdown():
            objects = self.robot.scene.get_objects()

            # Test if we are in the expected state
            if len(objects) == 0:
                return "succeeded"

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        return "aborted"


class AddInHandCollisionGeometry(State):
    def __init__(self, robot):
        State.__init__(self, outcomes=["succeeded", "aborted"])
        self.robot = robot

    def execute(self, ud):
        self.robot.scene.add_box("item", PoseStamped(header=Header(frame_id="gripper_equilibrium_grasp"),
                                                pose=Pose(position=Point(x=0, y=0, z=0.03), orientation=Quaternion(x=0, y=0, z=0, w=1))),
                            (.03, .03, .08))
        self.robot.scene.attach_box("gripper_equilibrium_grasp", "item", touch_links=["gripper_robotiq_arg2f_base_link", "gripper_left_distal_phalanx",
                                                 "gripper_left_proximal_phalanx", "gripper_right_proximal_phalanx",
                                                 "gripper_right_distal_phalanx", "gripper_left_bar", "gripper_right_bar", "gripper_base_link", "epick_end_effector"])
        start = rospy.get_time()
        seconds = rospy.get_time()

        timeout = 5.0
        while (seconds - start < timeout) and not rospy.is_shutdown():
            # Test if the box is in attached objects
            attached_objects = self.robot.scene.get_attached_objects(["item"])
            is_attached = len(attached_objects.keys()) > 0

            # Test if the box is in the scene.
            # Note that attaching the box will remove it from known_objects
            is_known = "item" in self.robot.scene.get_known_object_names()

            # Test if we are in the expected state
            if is_attached:
                return "succeeded"

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        return "aborted"


class AddCalibrationTargetInHandCollision(State):
    def __init__(self, robot):
        State.__init__(self, outcomes=["succeeded", "aborted"])
        self.robot = robot

    def execute(self, ud):
        self.robot.scene.add_box("calibration_target", PoseStamped(header=Header(frame_id="arm_tool0"),
                                                pose=Pose(position=Point(x=0, y=0, z=0.31), orientation=Quaternion(x=0, y=0, z=0, w=1))),
                            (.01, .18, .26))
        self.robot.scene.attach_box("arm_tool0", "calibration_target")
        start = rospy.get_time()
        seconds = rospy.get_time()
        timeout = 5.0
        while (seconds - start < timeout) and not rospy.is_shutdown():
            # Test if the box is in attached objects
            attached_objects = self.robot.scene.get_attached_objects(["calibration_target"])
            is_attached = len(attached_objects.keys()) > 0

            # Test if the box is in the scene.
            # Note that attaching the box will remove it from known_objects
            is_known = "calibration_target" in self.robot.scene.get_known_object_names()

            # Test if we are in the expected state
            if is_attached:
                return "succeeded"

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        return "aborted"


class AddPodCollisionGeometry(State):
    def __init__(self, robot):
        State.__init__(self, outcomes=["succeeded", "aborted"])
        self.robot = robot

    def execute(self, ud):
        # FIXME: We should probably read this in from transforms or something
        POD_SIZE = .9398
        HALF_POD_SIZE = POD_SIZE / 2
        WALL_WIDTH = 0.003
        # self.robot.scene.add_box("pod_top", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                            pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=2.03),
        #                                                      orientation=I_QUAT)), (POD_SIZE, POD_SIZE, 1.5))
        self.robot.scene.add_box("col_01", PoseStamped(header=Header(frame_id="pod_base_link"),
                                                      pose=Pose(position=Point(x=0.045, y=HALF_POD_SIZE, z=1.45),
                                                                orientation=I_QUAT)), (WALL_WIDTH, POD_SIZE, 2.3))
        # self.robot.scene.add_box("col_02", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=0.5*HALF_POD_SIZE, y=HALF_POD_SIZE, z=1.45),
        #                                                         orientation=I_QUAT)), (WALL_WIDTH, POD_SIZE, 2.3))
        # self.robot.scene.add_box("col_03", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=1.45),
        #                                                         orientation=I_QUAT)), (WALL_WIDTH, POD_SIZE, 2.3))
        # self.robot.scene.add_box("col_04", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=1.5*HALF_POD_SIZE, y=HALF_POD_SIZE, z=1.45),
        #                                                         orientation=I_QUAT)), (WALL_WIDTH, POD_SIZE, 2.3))
        self.robot.scene.add_box("col_05", PoseStamped(header=Header(frame_id="pod_base_link"),
                                                      pose=Pose(position=Point(x=POD_SIZE-0.045, y=HALF_POD_SIZE, z=1.45),
                                                                orientation=I_QUAT)), (WALL_WIDTH, POD_SIZE, 2.3))
        # self.robot.scene.add_box("row_01", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=0.34),
        #                                                         orientation=I_QUAT)), (POD_SIZE, POD_SIZE, WALL_WIDTH))
        # self.robot.scene.add_box("row_02", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=0.56),
        #                                                         orientation=I_QUAT)), (POD_SIZE, POD_SIZE, WALL_WIDTH))
        # self.robot.scene.add_box("row_03", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=0.70),
        #                                                         orientation=I_QUAT)), (POD_SIZE, POD_SIZE, WALL_WIDTH))
        # self.robot.scene.add_box("row_04", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=0.89),
        #                                                         orientation=I_QUAT)), (POD_SIZE, POD_SIZE, WALL_WIDTH))
        # self.robot.scene.add_box("row_05", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=1.04),
        #                                                         orientation=I_QUAT)), (POD_SIZE, POD_SIZE, WALL_WIDTH))
        # self.robot.scene.add_box("row_06", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=1.16),
        #                                                         orientation=I_QUAT)), (POD_SIZE, POD_SIZE, WALL_WIDTH))
        # self.robot.scene.add_box("row_07", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=1.38),
        #                                                         orientation=I_QUAT)), (POD_SIZE, POD_SIZE, WALL_WIDTH))
        # self.robot.scene.add_box("row_08", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=1.51),
        #                                                         orientation=I_QUAT)), (POD_SIZE, POD_SIZE, WALL_WIDTH))
        # self.robot.scene.add_box("row_09", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=1.66),
        #                                                         orientation=I_QUAT)), (POD_SIZE, POD_SIZE, WALL_WIDTH))
        # self.robot.scene.add_box("row_10", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=1.79),
        #                                                         orientation=I_QUAT)), (POD_SIZE, POD_SIZE, WALL_WIDTH))
        # self.robot.scene.add_box("row_11", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=2.01),
        #                                                         orientation=I_QUAT)), (POD_SIZE, POD_SIZE, WALL_WIDTH))
        # self.robot.scene.add_box("row_12", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=2.14),
        #                                                         orientation=I_QUAT)), (POD_SIZE, POD_SIZE, WALL_WIDTH))
        # self.robot.scene.add_box("row_13", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=2.29),
        #                                                         orientation=I_QUAT)), (POD_SIZE, POD_SIZE, WALL_WIDTH))
        # self.robot.scene.add_box("row_14", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=2.57),
        #                                                         orientation=I_QUAT)), (POD_SIZE, POD_SIZE, WALL_WIDTH))
        self.robot.scene.add_box("back_frame", PoseStamped(header=Header(frame_id="base_link"),
                                                    pose=Pose(position=Point(x=-0.28, y=0, z=1.27),
                                                              orientation=ROT_90_Z_QUAT)), (1.45, .2, 1.5))


        start = rospy.get_time()
        seconds = rospy.get_time()
        timeout = 50.0
        while (seconds - start < timeout) and not rospy.is_shutdown():
            objects = self.robot.scene.get_objects()

            # Test if we are in the expected state
            if len(objects) == 3:
                return "succeeded"

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        return "aborted"


class AddFullPodCollisionGeometryDropHide(State):
    def __init__(self, robot):
        State.__init__(self, outcomes=['succeeded', 'aborted'])
        self.robot = robot

    def execute(self, ud):
        # FIXME: We should probably read this in from transforms or something
        POD_SIZE = .9148
        HALF_POD_SIZE = POD_SIZE / 2
        WALL_WIDTH = 0.003
        SIDE_WALL_WIDTH = 0.033
        
        self.robot.scene.add_box("front_frame", PoseStamped(header=Header(frame_id="pod_base_link"),
                                                    pose=Pose(position=Point(x=POD_SIZE/2, y=0., z=1.34),
                                                              orientation=I_QUAT)), (1.8, .05, 3.0))
        
        self.robot.scene.add_box("left_side_frame", PoseStamped(header=Header(frame_id="base_link"),
                                                    pose=Pose(position=Point(x=0.25, y=1.00, z=1.34),
                                                              orientation=I_QUAT)), (2.00, .05, 3.0))
        
        self.robot.scene.add_box("right_side_frame", PoseStamped(header=Header(frame_id="base_link"),
                                                    pose=Pose(position=Point(x=0.25, y=-1.00, z=1.34),
                                                              orientation=I_QUAT)), (2.00, .05, 3.0))

        
        number_collision_box = 3

        start = rospy.get_time()
        seconds = rospy.get_time()
        timeout = 5.0
        while (seconds - start < timeout) and not rospy.is_shutdown():
            objects = self.robot.scene.get_objects()

            # Test if we are in the expected state
            if len(objects) == number_collision_box:
                return "succeeded"

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        return "aborted"


class AddFullPodCollisionGeometry(State):
    def __init__(self, robot,):
        State.__init__(self, input_keys=["target_bin_id"], outcomes=['succeeded', 'aborted'])
        self.robot = robot

    def execute(self, ud):
        # FIXME: We should probably read this in from transforms or something
        POD_SIZE = .9148
        HALF_POD_SIZE = POD_SIZE / 2
        WALL_WIDTH = 0.003
        SIDE_WALL_WIDTH = 0.033
        
        self.robot.scene.add_box("front_frame", PoseStamped(header=Header(frame_id="pod_base_link"),
                                                    pose=Pose(position=Point(x=POD_SIZE/2, y=0., z=1.34),
                                                              orientation=I_QUAT)), (1.8, .05, 3.0))
        
        self.robot.scene.add_box("left_side_frame", PoseStamped(header=Header(frame_id="base_link"),
                                                    pose=Pose(position=Point(x=0.25, y=1.00, z=1.34),
                                                              orientation=I_QUAT)), (2.00, .05, 3.0))
        
        self.robot.scene.add_box("right_side_frame", PoseStamped(header=Header(frame_id="base_link"),
                                                    pose=Pose(position=Point(x=0.25, y=-1.00, z=1.34),
                                                              orientation=I_QUAT)), (2.00, .05, 3.0))

        
        number_collision_box = 3
        print("target bin id", ud['target_bin_id'])
        try:
            import tf
            listener = tf.TransformListener()
            (trans, rot) = listener.lookupTransform('base_link', 'pod_bin_4h', rospy.Time())
        except:
            try:
                bin_id = ud['target_bin_id']
                self.bin_2A = ['C', 'D', 'E']
                if(bin_id[1] in self.bin_2A):
                    self.bin_2A_heights = {'C':1.15, 'D':1.4, 'E':1.65}
                    self.bin_2A_widths = {'1':0.28, '2':-0.03, '3':-0.34}
                    number_collision_box = 6
                    z_coordinate = self.bin_2A_heights[bin_id[1]]
                    y_coordinate = self.bin_2A_widths[bin_id[0]]
                    # self.robot.scene.add_box("horizontal_plane", PoseStamped(header=Header(frame_id="base_link"),
                    #                                             pose=Pose(position=Point(x=0.75, y=0.0, z=z_coordinate),
                    #                                                     orientation=I_QUAT)), (0.5, 1.5, 0.02))
                    
                    # self.robot.scene.add_box("vertical_plane_1", PoseStamped(header=Header(frame_id="base_link"),
                    #                                             pose=Pose(position=Point(x=0.72, y=y_coordinate+0.18, z=1.2),
                    #                                                     orientation=I_QUAT)), (0.5, 0.01, 3.0))
                    
                    # self.robot.scene.add_box("vertical_plane_2", PoseStamped(header=Header(frame_id="base_link"),
                    #                                             pose=Pose(position=Point(x=0.72, y=y_coordinate-0.18, z=1.2),
                    #                                                     orientation=I_QUAT)), (0.5, 0.01, 3.0))
            except Exception as e:
                print(e)

        start = rospy.get_time()
        seconds = rospy.get_time()
        timeout = 5.0
        while (seconds - start < timeout) and not rospy.is_shutdown():
            objects = self.robot.scene.get_objects()

            # Test if we are in the expected state
            if len(objects) == number_collision_box:
                return "succeeded"

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        return "aborted"


class AddPartialPodCollisionGeometry(State):
    def __init__(self, robot):
        State.__init__(self, outcomes=["succeeded", "aborted"])
        self.robot = robot

    def execute(self, ud):
        # FIXME: We should probably read this in from transforms or something
        POD_SIZE = .9398
        HALF_POD_SIZE = POD_SIZE / 2
        WALL_WIDTH = 0.003
        SIDE_WALL_WIDTH = 0.033
        # self.robot.scene.add_box("pod_top", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                            pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE, z=2.03),
        #                                                      orientation=I_QUAT)), (POD_SIZE, POD_SIZE, 1.5))
        self.robot.scene.add_box("col_01", PoseStamped(header=Header(frame_id="pod_base_link"),
                                                      pose=Pose(position=Point(x=0.0, y=HALF_POD_SIZE, z=1.45),
                                                                orientation=I_QUAT)), (SIDE_WALL_WIDTH, POD_SIZE, 2.3))

        self.robot.scene.add_box("col_05", PoseStamped(header=Header(frame_id="pod_base_link"),
                                                      pose=Pose(position=Point(x=POD_SIZE, y=HALF_POD_SIZE, z=1.45),
                                                                orientation=I_QUAT)), (SIDE_WALL_WIDTH, POD_SIZE, 2.3))

        self.robot.scene.add_box("back_frame", PoseStamped(header=Header(frame_id="base_link"),
                                                    pose=Pose(position=Point(x=-0.28, y=0, z=1.27),
                                                              orientation=ROT_90_Z_QUAT)), (1.45, .2, 1.5))

        # self.robot.scene.add_box("upper_pods_outer", PoseStamped(header=Header(frame_id="pod_base_link"),
        #                                               pose=Pose(position=Point(x=HALF_POD_SIZE, y=HALF_POD_SIZE-0.1, z=1.66+0.5),
        #                                                         orientation=I_QUAT)), (POD_SIZE, POD_SIZE, 1))

        start = rospy.get_time()
        seconds = rospy.get_time()
        timeout = 50.0
        while (seconds - start < timeout) and not rospy.is_shutdown():
            objects = self.robot.scene.get_objects()

            # Test if we are in the expected state
            if len(objects) == 3:
                return "succeeded"

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        return "aborted"
