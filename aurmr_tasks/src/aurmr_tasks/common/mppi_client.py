"""
Client-side helper for SMACH states to command the MPPI bridge node.

Communicates via ROS topics and services — no direct zerorpc dependency.
"""

import numpy as np
import rospy

from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import Bool, Float64
from std_srvs.srv import SetBool

import tf2_ros
import tf2_geometry_msgs  # noqa: F401  registers PoseStamped transforms


class MPPIClient:
    def __init__(self):
        # Publishers
        self.goal_pub = rospy.Publisher(
            '/mppi_bridge/goal', Point, queue_size=1, latch=True)

        # Service proxies
        self.enable_srv = rospy.ServiceProxy('/mppi_bridge/enable', SetBool)

        # Feedback subscribers
        self.goal_distance = float('inf')
        self._converged = False
        rospy.Subscriber('/mppi_bridge/goal_distance', Float64, self._dist_cb)
        rospy.Subscriber('/mppi_bridge/converged', Bool, self._converged_cb)

        # TF for frame transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def _dist_cb(self, msg):
        self.goal_distance = msg.data

    def _converged_cb(self, msg):
        self._converged = msg.data

    def set_goal_position(self, position):
        """Set goal from a [x, y, z] list/array in arm_base_link frame."""
        msg = Point(x=position[0], y=position[1], z=position[2])
        self.goal_pub.publish(msg)

    def set_goal_pose(self, pose_stamped):
        """Set goal from a PoseStamped, transforming to arm_base_link if needed."""
        if pose_stamped.header.frame_id != 'arm_base_link':
            try:
                pose_stamped = self.tf_buffer.transform(
                    pose_stamped, 'arm_base_link', rospy.Duration(1.0))
            except Exception as e:
                rospy.logerr("Failed to transform goal to arm_base_link: %s", e)
                return False
        p = pose_stamped.pose.position
        self.set_goal_position([p.x, p.y, p.z])
        return True

    def enable(self):
        """Enable the MPPI control loop (activates velocity controller)."""
        try:
            resp = self.enable_srv(True)
            if not resp.success:
                rospy.logerr("MPPI enable failed: %s", resp.message)
            return resp.success
        except rospy.ServiceException as e:
            rospy.logerr("MPPI enable service call failed: %s", e)
            return False

    def disable(self):
        """Disable the MPPI control loop (restores trajectory controller)."""
        try:
            resp = self.enable_srv(False)
            return resp.success
        except rospy.ServiceException as e:
            rospy.logerr("MPPI disable service call failed: %s", e)
            return False

    def wait_for_convergence(self, timeout=15.0, tolerance=0.01):
        """
        Block until the EE is within tolerance of the goal, or timeout.

        Returns True if converged, False if timed out.
        """
        start = rospy.Time.now()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            elapsed = (rospy.Time.now() - start).to_sec()
            if elapsed > timeout:
                rospy.logwarn(
                    "MPPI convergence timed out after %.1f s (distance=%.4f)",
                    timeout, self.goal_distance)
                return False
            if self.goal_distance < tolerance:
                rospy.loginfo(
                    "MPPI goal reached (distance=%.4f)", self.goal_distance)
                return True
            rate.sleep()
        return False

    def move_to_position(self, position, timeout=15.0, tolerance=0.01):
        """
        Full motion sequence: set goal, enable, wait, disable.

        Args:
            position: [x, y, z] in arm_base_link frame.
            timeout: seconds to wait for convergence.
            tolerance: distance threshold in meters.

        Returns:
            True if converged, False otherwise.
        """
        self.set_goal_position(position)
        rospy.sleep(0.05)  # let goal propagate to planner

        if not self.enable():
            return False

        converged = self.wait_for_convergence(timeout, tolerance)
        self.disable()
        return converged

    def move_to_pose(self, pose_stamped, timeout=15.0, tolerance=0.01):
        """
        Full motion sequence from a PoseStamped.
        Extracts position (orientation is handled by MPPI cost function).
        """
        if not self.set_goal_pose(pose_stamped):
            return False
        rospy.sleep(0.05)

        if not self.enable():
            return False

        converged = self.wait_for_convergence(timeout, tolerance)
        self.disable()
        return converged
