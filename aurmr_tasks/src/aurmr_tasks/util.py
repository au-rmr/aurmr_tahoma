from copy import deepcopy

import geometry_msgs.msg
from math import pi, tau, dist, fabs, cos

import rospy
from geometry_msgs.msg import PoseStamped
from moveit_commander.conversions import pose_to_list
from smach import State, StateMachine
from controller_manager_msgs.srv import SwitchController


def apply_offset_to_pose(pose, offset, offset_frame=None, tf_buffer=None):
    if not isinstance(pose, PoseStamped):
        raise RuntimeError("Can't offset without knowing pose frame")
    offset_pose = deepcopy(pose)
    if offset_frame is None:
        offset_frame = offset_pose.header.frame_id

    offset_pose = tf_buffer.transform(offset_pose, offset_frame, rospy.Duration(1))
    offset_pose.pose.position.x += offset[0]
    offset_pose.pose.position.y += offset[1]
    offset_pose.pose.position.z += offset[2]
    # It would be better to grab the transform once and apply it to the offset vector
    offset_pose = tf_buffer.transform(offset_pose, pose.header.frame_id, rospy.Duration(1))
    return offset_pose


class Formulator(State):
    def __init__(self, template, input_keys, output_key):
        State.__init__(self, outcomes=["succeeded", "aborted"], input_keys=input_keys,
                       output_keys=[output_key])
        self.template = template

    def execute(self, ud):
        try:
            args = [ud[input_key] for input_key in self._input_keys]
            ud[list(self._output_keys)[0]] = self.template.format(*args)
            return "succeeded"
        except Exception as e:
            # Catch any fumbled templates
            rospy.logerr("Bad formulate_ud_str: {}, {}, {}".format(self.template, self._input_keys, self._output_keys))
            return "aborted"


def formulate_ud_str_auto(name, template, input_keys, output_key, transitions=None):
    if transitions is None:
        transitions = {}
    StateMachine.add_auto(name, Formulator(template, input_keys, output_key), ["succeeded"],transitions=transitions)


def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if the values in two lists are within a tolerance of each other.
    For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
    between the identical orientations q and -q is calculated correctly).
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        assert (goal.header.frame_id == actual.header.frame_id)
        #print("This is posestamped")
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))
        print(d)
        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        # return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)
        return d <= tolerance

    return True


def pose_dist(goal, actual):
    if type(goal) is geometry_msgs.msg.PoseStamped:
        return pose_dist(goal.pose, actual.pose)

    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))
        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return (d, cos_phi_half)
    else:
        return None
    

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
