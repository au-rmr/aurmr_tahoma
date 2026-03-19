#!/usr/bin/env python
"""
MPPI Bridge Node — connects the genesismpc MPPI planner (zerorpc) to
the real UR robot via ROS velocity control.

Runs a 100 Hz control loop:
  1. Reads /joint_states
  2. Sends state to the MPPI planner via zerorpc
  3. Publishes velocity commands to joint_group_vel_controller
  4. Publishes convergence feedback for SMACH states

Usage:
  rosrun aurmr_tasks mppi_bridge_node.py _planner_address:=tcp://127.0.0.1:4242
"""

import math
import threading

import numpy as np
import rospy
import tf2_ros
import torch
import zerorpc

from controller_manager_msgs.srv import ListControllers, SwitchController
from geometry_msgs.msg import Point, WrenchStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, Float64MultiArray
from std_srvs.srv import SetBool, SetBoolResponse

from genesis_mpc.utils.transport import torch_to_bytes, bytes_to_torch

# Joint ordering must match the velocity controller config
JOINT_NAMES = [
    'arm_shoulder_pan_joint',
    'arm_shoulder_lift_joint',
    'arm_elbow_joint',
    'arm_wrist_1_joint',
    'arm_wrist_2_joint',
    'arm_wrist_3_joint',
]
NUM_JOINTS = len(JOINT_NAMES)

# UR joint limits (radians) — conservative, within ±2pi
JOINT_LIMITS_LOWER = np.array([-2 * math.pi] * NUM_JOINTS)
JOINT_LIMITS_UPPER = np.array([2 * math.pi] * NUM_JOINTS)
LIMIT_MARGIN = 0.05  # rad

# Controllers
VEL_CONTROLLER = 'joint_group_vel_controller'
TRAJ_CONTROLLER = 'scaled_pos_joint_traj_controller'
TRAJ_CONTROLLER_SIM = 'pos_joint_traj_controller'
POS_GROUP_CONTROLLER = 'joint_group_pos_controller'


class MPPIBridgeNode:
    def __init__(self):
        rospy.init_node('mppi_bridge_node')

        # Parameters
        self.planner_address = rospy.get_param('~planner_address', 'tcp://127.0.0.1:4242')
        self.control_rate = rospy.get_param('~control_rate', 100)
        self.max_velocity = rospy.get_param('~max_velocity', 1.0)
        self.force_limit = rospy.get_param('~force_limit', 50.0)
        self.goal_tolerance = rospy.get_param('~goal_tolerance', 0.01)
        self.in_sim = rospy.get_param('~in_sim', False)

        # State
        self.enabled = False
        self.current_q = None
        self.current_dq = None
        self.current_goal = None  # (3,) numpy, in arm_base_link frame
        self.goal_distance = float('inf')
        self.force_mag = 0.0
        self.last_planner_response_time = None
        self.state_lock = threading.Lock()

        # Zerorpc planner client
        self.planner = zerorpc.Client(timeout=50)
        self.planner.connect(self.planner_address)
        rospy.loginfo("Connected to MPPI planner at %s", self.planner_address)

        # Verify planner is reachable
        try:
            self.planner.get_goal()
            rospy.loginfo("MPPI planner is responding")
        except Exception as e:
            rospy.logwarn("MPPI planner not yet reachable: %s", e)

        # Controller manager services
        rospy.loginfo("Waiting for controller_manager services...")
        rospy.wait_for_service('/controller_manager/list_controllers', timeout=10.0)
        self._controller_lister = rospy.ServiceProxy(
            '/controller_manager/list_controllers', ListControllers)
        self._controller_switcher = rospy.ServiceProxy(
            '/controller_manager/switch_controller', SwitchController)

        # TF for EE pose lookup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Subscribers
        rospy.Subscriber('/joint_states', JointState, self._joint_state_cb)
        rospy.Subscriber('/wrench', WrenchStamped, self._wrench_cb)
        rospy.Subscriber('/mppi_bridge/goal', Point, self._goal_cb)

        # Publishers
        self.vel_pub = rospy.Publisher(
            '/joint_group_vel_controller/command',
            Float64MultiArray, queue_size=1)
        self.distance_pub = rospy.Publisher(
            '/mppi_bridge/goal_distance', Float64, queue_size=1)
        self.converged_pub = rospy.Publisher(
            '/mppi_bridge/converged', Bool, queue_size=1)

        # Services
        rospy.Service('/mppi_bridge/enable', SetBool, self._enable_cb)

        # Control loop timer
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.control_rate), self._control_loop)

        rospy.on_shutdown(self._shutdown)
        rospy.loginfo("MPPI bridge node ready (rate=%d Hz, max_vel=%.2f rad/s)",
                      self.control_rate, self.max_velocity)

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #

    def _joint_state_cb(self, msg):
        """Extract q and dq in the correct joint order."""
        try:
            indices = [msg.name.index(name) for name in JOINT_NAMES]
        except ValueError:
            return  # message doesn't contain our joints

        q = np.array([msg.position[i] for i in indices])
        dq = np.array([msg.velocity[i] for i in indices]) if msg.velocity else np.zeros(NUM_JOINTS)

        with self.state_lock:
            self.current_q = q
            self.current_dq = dq

    def _wrench_cb(self, msg):
        f = msg.wrench.force
        self.force_mag = math.sqrt(f.x ** 2 + f.y ** 2 + f.z ** 2)

    def _goal_cb(self, msg):
        """Receive a goal position in arm_base_link frame."""
        self.current_goal = np.array([msg.x, msg.y, msg.z])
        # Forward to the MPPI planner
        goal_tensor = torch.tensor(
            [msg.x, msg.y, msg.z], dtype=torch.float32).unsqueeze(0)
        try:
            self.planner.set_goal(torch_to_bytes(goal_tensor))
            rospy.loginfo("MPPI goal set to [%.3f, %.3f, %.3f]",
                          msg.x, msg.y, msg.z)
        except Exception as e:
            rospy.logerr("Failed to set MPPI goal: %s", e)

    def _enable_cb(self, req):
        """Enable or disable the MPPI control loop."""
        if req.data:
            if self.current_goal is None:
                return SetBoolResponse(
                    success=False, message="No goal set. Publish to /mppi_bridge/goal first.")
            ok = self._activate_vel_controller()
            if ok:
                self.enabled = True
                return SetBoolResponse(success=True, message="MPPI control enabled")
            else:
                return SetBoolResponse(success=False, message="Failed to activate velocity controller")
        else:
            self.enabled = False
            self._publish_zero_velocity()
            self._deactivate_vel_controller()
            return SetBoolResponse(success=True, message="MPPI control disabled")

    # ------------------------------------------------------------------ #
    # Control loop
    # ------------------------------------------------------------------ #

    def _control_loop(self, event):
        if not self.enabled:
            return

        # 1. Snapshot joint state
        with self.state_lock:
            if self.current_q is None:
                rospy.logwarn_throttle(1.0, "No joint state received yet")
                return
            q = self.current_q.copy()
            dq = self.current_dq.copy()

        # 2. Build dof_state tensor (robot only, no objects in phase 1)
        q_t = torch.tensor(q, dtype=torch.float32)
        dq_t = torch.tensor(dq, dtype=torch.float32)
        dof_state = torch.cat([q_t, dq_t])

        root_state = torch.zeros((1, 13), dtype=torch.float32)
        root_state[0, 6] = 1.0  # identity quaternion w=1

        # 3. Query planner
        try:
            action_bytes = self.planner.compute_action_tensor(
                torch_to_bytes(dof_state),
                torch_to_bytes(root_state),
            )
            action = bytes_to_torch(action_bytes)
            self.last_planner_response_time = rospy.Time.now()
        except Exception as e:
            rospy.logerr_throttle(1.0, "Planner unreachable: %s", e)
            self._publish_zero_velocity()
            return

        # 4. Safety checks
        action_np = action.cpu().numpy().flatten()[:NUM_JOINTS]

        # Clamp velocities
        action_np = np.clip(action_np, -self.max_velocity, self.max_velocity)

        # Joint limit enforcement
        for i in range(NUM_JOINTS):
            if q[i] <= JOINT_LIMITS_LOWER[i] + LIMIT_MARGIN and action_np[i] < 0:
                action_np[i] = 0.0
            if q[i] >= JOINT_LIMITS_UPPER[i] - LIMIT_MARGIN and action_np[i] > 0:
                action_np[i] = 0.0

        # Force safety
        if self.force_mag > self.force_limit:
            rospy.logwarn_throttle(1.0,
                "Force limit exceeded (%.1f N > %.1f N), zeroing velocity",
                self.force_mag, self.force_limit)
            action_np[:] = 0.0

        # Watchdog: zero if planner hasn't responded recently
        if self.last_planner_response_time is not None:
            dt = (rospy.Time.now() - self.last_planner_response_time).to_sec()
            if dt > 0.1:
                rospy.logwarn_throttle(1.0,
                    "Planner response stale (%.3f s), zeroing velocity", dt)
                action_np[:] = 0.0

        # 5. Publish velocity command
        msg = Float64MultiArray()
        msg.data = action_np.tolist()
        self.vel_pub.publish(msg)

        # 6. Convergence feedback
        self._publish_convergence()

    # ------------------------------------------------------------------ #
    # Convergence
    # ------------------------------------------------------------------ #

    def _publish_convergence(self):
        if self.current_goal is None:
            return
        try:
            tf = self.tf_buffer.lookup_transform(
                'arm_base_link', 'arm_tool0', rospy.Time(0), rospy.Duration(0.05))
            t = tf.transform.translation
            ee_pos = np.array([t.x, t.y, t.z])
            self.goal_distance = np.linalg.norm(ee_pos - self.current_goal)
        except Exception:
            return

        self.distance_pub.publish(Float64(data=self.goal_distance))
        self.converged_pub.publish(Bool(data=self.goal_distance < self.goal_tolerance))

    # ------------------------------------------------------------------ #
    # Controller switching
    # ------------------------------------------------------------------ #

    def _activate_vel_controller(self):
        to_stop = []
        for name in [TRAJ_CONTROLLER, TRAJ_CONTROLLER_SIM, POS_GROUP_CONTROLLER]:
            if self._is_controller_running(name):
                to_stop.append(name)
        rospy.loginfo("Activating %s, stopping %s", VEL_CONTROLLER, to_stop)
        try:
            ok = self._controller_switcher.call(
                [VEL_CONTROLLER], to_stop, 1, False, 2.0)
            if not ok:
                rospy.logerr("Controller switch failed")
            return ok
        except Exception as e:
            rospy.logerr("Controller switch error: %s", e)
            return False

    def _deactivate_vel_controller(self):
        traj = TRAJ_CONTROLLER_SIM if self.in_sim else TRAJ_CONTROLLER
        rospy.loginfo("Deactivating %s, restoring %s", VEL_CONTROLLER, traj)
        try:
            self._controller_switcher.call(
                [traj], [VEL_CONTROLLER], 1, False, 2.0)
        except Exception as e:
            rospy.logerr("Controller switch error on deactivate: %s", e)

    def _is_controller_running(self, name):
        try:
            controllers = self._controller_lister().controller
            return any(c.name == name and c.state == 'running' for c in controllers)
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    def _publish_zero_velocity(self):
        msg = Float64MultiArray()
        msg.data = [0.0] * NUM_JOINTS
        self.vel_pub.publish(msg)

    def _shutdown(self):
        rospy.loginfo("MPPI bridge shutting down")
        self.enabled = False
        self._publish_zero_velocity()
        rospy.sleep(0.1)
        self._deactivate_vel_controller()


if __name__ == '__main__':
    try:
        node = MPPIBridgeNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
