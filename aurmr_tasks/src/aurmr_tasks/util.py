import geometry_msgs.msg
from math import pi, tau, dist, fabs, cos

import rospy
from moveit_commander.conversions import pose_to_list
from smach import State, StateMachine


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
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))
        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

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
