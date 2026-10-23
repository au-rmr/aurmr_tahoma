#!/usr/bin/env python
"""
MPPI Bridge Node — connects the genesismpc MPPI planner (zerorpc) to
the real UR robot via ROS velocity control.

Runs a 100 Hz control loop in a dedicated thread (required because
zerorpc uses gevent, which needs its own event loop per thread).

Usage:
  rosrun aurmr_tasks mppi_bridge_node.py _planner_address:=tcp://127.0.0.1:4242
"""

import math
import threading
import time

import numpy as np
import rospy
import tf2_ros
import torch
import io

from controller_manager_msgs.srv import ListControllers, SwitchController
from geometry_msgs.msg import Point, WrenchStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, Float64MultiArray
from std_srvs.srv import SetBool, SetBoolResponse
from aurmr_tasks.common.object_tracker import ObjectTracker
from aurmr_tasks.common.foundationpose_tracker import FoundationPoseTracker

def torch_to_bytes(t: torch.Tensor) -> bytes:
    buff = io.BytesIO()
    torch.save(t, buff)
    buff.seek(0)
    return buff.read()


def bytes_to_torch(b: bytes) -> torch.Tensor:
    buff = io.BytesIO(b)
    return torch.load(buff)

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
        self.force_limit = rospy.get_param('~force_limit', 60.0)
        self.goal_tolerance = rospy.get_param('~goal_tolerance', 0.01)
        self.in_sim = rospy.get_param('~in_sim', False)
        self.smoothing_alpha = rospy.get_param('~smoothing_alpha', 0.2)

        # How much an object must move (meters) before pushing the update to the planner
        self.object_update_threshold = rospy.get_param('~object_update_threshold', 0.02)
        # Minimum seconds between object state pushes regardless of movement
        self.object_update_min_period = rospy.get_param('~object_update_min_period', 0.5)
        # Rate at which velocity commands are published to the robot
        self.publish_rate = rospy.get_param('~publish_rate', 500)
        # Seconds before a stale planner command triggers a hold-position
        self.command_timeout = rospy.get_param('~command_timeout', 0.15)

        # Rescale smoothing_alpha from control_rate to publish_rate so the
        # effective time constant τ = -dt/ln(1-α) stays the same.
        dt_ctrl = 1.0 / self.control_rate
        dt_pub  = 1.0 / self.publish_rate
        self._publish_alpha = 1.0 - (1.0 - self.smoothing_alpha) ** (dt_pub / dt_ctrl)

        self._last_object_push_time = 0.0
        self._last_pushed_object_positions = None  # (N, 3) numpy

        # Latest planner command: (timestamp: float, action: np.ndarray) or None
        self._latest_command = None
        self._command_lock = threading.Lock()

        # Object tracker (optional — only active when object_tracking param is set).
        # tracker_type: 'foundationpose' (default) or 'apriltag'
        self.object_tracker = None
        if rospy.has_param('~object_tracking'):
            tracker_type = rospy.get_param('~object_tracking/tracker_type', 'foundationpose')
            if tracker_type == 'apriltag':
                self.object_tracker = ObjectTracker(param_prefix='~object_tracking')
            else:
                self.object_tracker = FoundationPoseTracker(param_prefix='~object_tracking')
            rospy.loginfo("Object tracking enabled: %s (%d objects)",
                          tracker_type, self.object_tracker.num_cubes())
        else:
            rospy.loginfo("Object tracking disabled (no ~object_tracking param)")

        # State (shared between ROS callbacks and control thread)
        self.enabled = False
        self.current_q = None
        self.current_dq = None
        self.current_goal = None  # (3,) numpy, in arm_base_link frame
        self.goal_to_send = None  # pending goal to forward to planner
        self.goal_distance = float('inf')
        self.force_mag = 0.0
        self.last_action = np.zeros(NUM_JOINTS)
        self.state_lock = threading.Lock()

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

        # Planner thread: owns the zerorpc client, runs at the planner's
        # natural rate (~25 Hz limited by GPU compute time).
        self._control_thread = threading.Thread(
            target=self._control_thread_main, daemon=True)
        self._control_thread.start()

        # Publisher thread: reads the latest planner command and publishes
        # velocity commands at publish_rate Hz with smoothing + safety checks.
        self._publisher_thread = threading.Thread(
            target=self._publisher_thread_main, daemon=True)
        self._publisher_thread.start()

        rospy.on_shutdown(self._shutdown)
        rospy.loginfo("MPPI bridge node ready (planner~%d Hz, publish=%d Hz, max_vel=%.2f rad/s)",
                      self.control_rate, self.publish_rate, self.max_velocity)

    # ------------------------------------------------------------------ #
    # Control thread (owns the zerorpc client)
    # ------------------------------------------------------------------ #

    def _control_thread_main(self):
        """Planner thread — owns the zerorpc client.

        Calls compute_action_tensor as fast as the planner allows (~25 Hz),
        stores the result with a timestamp for the publisher thread to consume.
        No sleep: the GPU compute time is the natural rate limiter.
        """
        import zerorpc

        planner = zerorpc.Client(timeout=30)
        planner.connect(self.planner_address)
        rospy.loginfo("Control thread connected to MPPI planner at %s",
                      self.planner_address)

        try:
            planner.get_goal()
            rospy.loginfo("MPPI planner is responding")
        except Exception as e:
            rospy.logwarn("MPPI planner not yet reachable: %s", e)

        while not rospy.is_shutdown():
            self._planner_step(planner)

    def _planner_step(self, planner):
        """Query the planner once and store the result in _latest_command."""
        with self.state_lock:
            if self.current_q is None:
                time.sleep(0.01)
                return
            q  = self.current_q.copy()
            dq = self.current_dq.copy()
            pending_goal = self.goal_to_send
            self.goal_to_send = None

        if pending_goal is not None:
            try:
                goal_tensor = torch.tensor(
                    pending_goal, dtype=torch.float32).unsqueeze(0)
                planner.set_goal(torch_to_bytes(goal_tensor))
                rospy.loginfo("MPPI goal set to [%.3f, %.3f, %.3f]",
                              pending_goal[0], pending_goal[1], pending_goal[2])
            except Exception as e:
                rospy.logerr("Failed to set MPPI goal: %s", e)

        dof_state = torch.cat([torch.tensor(q,  dtype=torch.float32),
                               torch.tensor(dq, dtype=torch.float32)])

        if self.object_tracker is not None:
            self._maybe_push_object_states(planner)

        root_state = torch.zeros((1, 13), dtype=torch.float32)
        root_state[0, 6] = 1.0

        try:
            action_bytes = planner.compute_action_tensor(
                torch_to_bytes(dof_state),
                torch_to_bytes(root_state),
            )
        except Exception as e:
            rospy.logerr_throttle(1.0, "Planner unreachable: %s", e)
            return

        action_np = bytes_to_torch(action_bytes).cpu().numpy().flatten()[:NUM_JOINTS]
        action_np = np.clip(action_np, -self.max_velocity, self.max_velocity)

        # Joint limit enforcement needs current q — do it here before storing
        for i in range(NUM_JOINTS):
            if q[i] <= JOINT_LIMITS_LOWER[i] + LIMIT_MARGIN and action_np[i] < 0:
                action_np[i] = 0.0
            if q[i] >= JOINT_LIMITS_UPPER[i] - LIMIT_MARGIN and action_np[i] > 0:
                action_np[i] = 0.0

        with self._command_lock:
            self._latest_command = (time.time(), action_np)

    # ------------------------------------------------------------------ #
    # Publisher thread (500 Hz) — safety checks, smoothing, publish
    # ------------------------------------------------------------------ #

    def _publisher_thread_main(self):
        """High-rate publisher thread.

        Reads the latest planner command, enforces safety constraints,
        applies exponential smoothing (rescaled to publish_rate), and
        publishes velocity commands at publish_rate Hz.
        Holds position (zero velocity) when the command is stale.
        """
        rate = rospy.Rate(self.publish_rate)
        rospy.loginfo("Publisher thread started at %d Hz (alpha=%.4f)",
                      self.publish_rate, self._publish_alpha)
        while not rospy.is_shutdown():
            self._publisher_step()
            rate.sleep()

    def _publisher_step(self):
        with self._command_lock:
            cmd = self._latest_command

        if cmd is None:
            if self.enabled:
                self._publish_zero_velocity()
            return

        ts, action_np = cmd
        age = time.time() - ts

        if age > self.command_timeout:
            if self.enabled:
                rospy.logwarn_throttle(1.0,
                    "Planner command stale (%.3f s > %.3f s), holding position",
                    age, self.command_timeout)
                self._publish_zero_velocity()
            return

        if not self.enabled:
            return

        action_np = action_np.copy()

        # Force safety (checked at publish_rate for fast response)
        if self.force_mag > self.force_limit:
            rospy.logwarn_throttle(1.0,
                "Force limit exceeded (%.1f N > %.1f N), zeroing velocity",
                self.force_mag, self.force_limit)
            action_np[:] = 0.0
            self.last_action[:] = 0.0  # flush filter memory

        # Stop when converged
        if self.goal_distance < self.goal_tolerance:
            action_np[:] = 0.0
            self.last_action[:] = 0.0  # flush filter memory

        # Exponential smoothing at publish_rate
        action_np = (self._publish_alpha * action_np
                     + (1.0 - self._publish_alpha) * self.last_action)
        self.last_action = action_np.copy()

        # # Deadband: suppress tiny residual commands that cause near-goal oscillation
        # if np.max(np.abs(action_np)) < 0.1: #0.005:
        #     action_np[:] = 0.0

        msg = Float64MultiArray()
        msg.data = action_np.tolist()
        self.vel_pub.publish(msg)

        self._publish_convergence()

    # Table bounds from conf/actors/table.yaml:
    #   center=[0.65, 0.0, 0.71], size=[1.40, 2.50, 0.10]
    _TABLE_X = (0.65 - 0.70, 0.65 + 0.70)   # (−0.05,  1.35)
    _TABLE_Y = (0.0  - 1.25, 0.0  + 1.25)   # (−1.25,  1.25)
    _TABLE_TOP_Z = 0.775 + 0.035               # 0.76 m

    def _object_on_table(self, pos, half_size):
        """Return True if the object at `pos` overlaps with the table volume."""
        x, y, z = pos
        in_xy = (self._TABLE_X[0] - half_size < x < self._TABLE_X[1] + half_size and
                 self._TABLE_Y[0] - half_size < y < self._TABLE_Y[1] + half_size)
        in_z  = z < self._TABLE_TOP_Z + half_size
        return in_xy and in_z


    def _maybe_push_object_states(self, planner):
        """Push object states to the planner only when objects have moved
        significantly or the minimum period has elapsed."""
        obj_states = self.object_tracker.get_states()
        if not any(s is not None for s in obj_states):
            return

        now = time.time()
        half_cube = float(rospy.get_param('~object_tracking/cube_size', 0.10)) / 2.0
        q_blocks, dq_blocks = [], []
        current_positions = []
        for i, state in enumerate(obj_states):
            if state is not None:
                q_obj, dq_obj = state
                # print(f'object {i}: {q_obj}')
                # if self._object_on_table(q_obj[:3], half_cube):
                #     rospy.logwarn_throttle(2.0,
                #         "Object %d pose [%.3f, %.3f, %.3f] intersects table — skipping update",
                #         i, q_obj[0], q_obj[1], q_obj[2])
                #     # Use last known good position if available
                #     if self._last_pushed_object_positions is not None:
                #         q_obj = q_obj.copy()
                #         q_obj[:3] = self._last_pushed_object_positions[i]
                #     dq_obj = np.zeros(6)
            else:
                q_obj  = np.zeros(6)
                dq_obj = np.zeros(6)
            q_blocks.append(q_obj)
            dq_blocks.append(dq_obj)
            current_positions.append(q_obj[:3])

        current_positions = np.array(current_positions)  # (N, 3)

        # Check if any object has moved beyond threshold
        moved = False
        if self._last_pushed_object_positions is None:
            moved = True
        else:
            max_displacement = np.max(
                np.linalg.norm(current_positions - self._last_pushed_object_positions, axis=1))
            if max_displacement > self.object_update_threshold:
                moved = True

        period_elapsed = (now - self._last_object_push_time) > self.object_update_min_period

        if not (moved or period_elapsed):
            return

        # Assemble and send
        obj_data = torch.tensor(
            np.concatenate(q_blocks + dq_blocks), dtype=torch.float32)
        try:
            planner.set_object_states(torch_to_bytes(obj_data))
            self._last_pushed_object_positions = current_positions.copy()
            self._last_object_push_time = now
        except Exception as e:
            rospy.logerr_throttle(1.0, "Failed to push object states: %s", e)


    # ------------------------------------------------------------------ #
    # ROS Callbacks (run on ROS threads, no zerorpc calls)
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
        with self.state_lock:
            self.current_goal = np.array([msg.x, msg.y, msg.z])
            self.goal_to_send = np.array([msg.x, msg.y, msg.z])

    def _enable_cb(self, req):
        """Enable or disable the MPPI control loop."""
        if req.data:
            ok = self._activate_vel_controller()
            if ok:
                self.enabled = True
                return SetBoolResponse(success=True, message="MPPI control enabled")
            else:
                return SetBoolResponse(success=False, message="Failed to activate velocity controller")
        else:
            self.enabled = False
            self.last_action = np.zeros(NUM_JOINTS)
            self._publish_zero_velocity()
            self._deactivate_vel_controller()
            return SetBoolResponse(success=True, message="MPPI control disabled")

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
