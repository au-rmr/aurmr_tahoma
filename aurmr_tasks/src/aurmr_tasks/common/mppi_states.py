"""
SMACH states for MPPI-based motion via the mppi_bridge_node.

These are thin wrappers around MPPIClient, following the same pattern
as the MoveIt-based states in motion.py.
"""

import rospy
from smach import State


class MPPIMoveToGoal(State):
    """Move end-effector to a PoseStamped goal using MPPI."""

    def __init__(self, mppi_client, default_pose=None, timeout=15.0, tolerance=0.01):
        State.__init__(self, input_keys=['pose'],
                       outcomes=['succeeded', 'aborted'])
        self.mppi = mppi_client
        self.default_pose = default_pose
        self.timeout = timeout
        self.tolerance = tolerance

    def execute(self, ud):
        pose = self.default_pose if self.default_pose else ud['pose']
        rospy.loginfo("MPPI moving to pose in frame '%s'", pose.header.frame_id)
        success = self.mppi.move_to_pose(pose, self.timeout, self.tolerance)
        return 'succeeded' if success else 'aborted'


class MPPIMoveToPosition(State):
    """Move end-effector to a [x, y, z] position (arm_base_link) using MPPI."""

    def __init__(self, mppi_client, default_position=None, timeout=15.0, tolerance=0.01):
        State.__init__(self, input_keys=['position'],
                       outcomes=['succeeded', 'aborted'])
        self.mppi = mppi_client
        self.default_position = default_position
        self.timeout = timeout
        self.tolerance = tolerance

    def execute(self, ud):
        position = self.default_position if self.default_position else ud['position']
        rospy.loginfo("MPPI moving to position [%.3f, %.3f, %.3f]",
                      position[0], position[1], position[2])
        success = self.mppi.move_to_position(position, self.timeout, self.tolerance)
        return 'succeeded' if success else 'aborted'
