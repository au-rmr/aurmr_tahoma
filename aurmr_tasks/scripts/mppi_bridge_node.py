#!/usr/bin/env python
"""
MPPI Bridge Node — connects the genesismpc MPPI planner (zerorpc) to
the real UR robot via ROS velocity control.

Runs a 100 Hz control loop in a dedicated thread (required because
zerorpc uses gevent, which needs its own event loop per thread).

Usage:
  rosrun aurmr_tasks mppi_bridge_node.py _planner_address:=tcp://127.0.0.1:4242
"""

import json
import math
import os
import threading
import time

import cv2
import numpy as np
import rospy
import tf2_ros
import torch
import io

from controller_manager_msgs.srv import ListControllers, SwitchController
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, WrenchStamped
from sensor_msgs.msg import Image as RosImage, JointState
from std_msgs.msg import Bool, Float64, Float64MultiArray
from std_srvs.srv import SetBool, SetBoolResponse
from aurmr_tasks.common.object_tracker import ObjectTracker
from aurmr_tasks.common.foundationpose_tracker import FoundationPoseTracker

def _euler_xyz_to_quat_wxyz(euler_xyz: np.ndarray) -> np.ndarray:
    """Convert intrinsic XYZ Euler angles (radians) to wxyz quaternion."""
    ex, ey, ez = euler_xyz[0], euler_xyz[1], euler_xyz[2]
    cx, sx = np.cos(ex / 2), np.sin(ex / 2)
    cy, sy = np.cos(ey / 2), np.sin(ey / 2)
    cz, sz = np.cos(ez / 2), np.sin(ez / 2)
    return np.array([
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
    ], dtype=np.float32)


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
        # Rate at which velocity commands are published to the robot
        self.publish_rate = rospy.get_param('~publish_rate', 500)
        # Seconds before a stale planner command triggers a hold-position
        self.command_timeout = rospy.get_param('~command_timeout', 0.15)
        # Maximum joint acceleration (rad/s²) — limits jerk without adding phase lag
        self.max_accel = rospy.get_param('~max_accel', 2.0)

        dt_pub = 1.0 / self.publish_rate
        self._max_delta = self.max_accel * dt_pub  # max velocity change per publisher step

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

        # ------------------------------------------------------------------ #
        # Demo collection
        # ------------------------------------------------------------------ #
        self.collect_demos     = rospy.get_param('~collect_demos',     False)
        self.demo_out_dir      = rospy.get_param('~demo_out_dir',      '/tmp/real_robot_demos')
        self.demo_n_episodes   = rospy.get_param('~demo_n_episodes',   200)
        self.demo_max_steps    = rospy.get_param('~demo_max_steps',    500)
        self.demo_success_dist = rospy.get_param('~demo_success_dist', 0.04)
        self.demo_action_dt    = rospy.get_param('~demo_action_dt',    1.0 / 60.0)
        self.demo_steps_json   = rospy.get_param('~demo_steps_json',   '')

        self._bridge           = CvBridge()
        self._camera_images    = [None, None]   # [base_rgb, wrist_rgb] 224×224 uint8
        self._camera_lock      = threading.Lock()

        self._ep_frames        = []
        self._ep_steps         = 0
        self._n_success        = 0
        self._n_attempts       = 0
        self._collection_steps = []
        self._collecting       = False
        self._pending_start    = False
        self._collection_lock  = threading.Lock()

        if self.collect_demos:
            base_topic  = rospy.get_param('~base_camera_topic',  '/base_camera/color/image_raw')
            wrist_topic = rospy.get_param('~wrist_camera_topic', '/wrist_camera/color/image_raw')
            rospy.Subscriber(base_topic,  RosImage, lambda m: self._camera_cb(0, m))
            rospy.Subscriber(wrist_topic, RosImage, lambda m: self._camera_cb(1, m))
            if self.demo_steps_json:
                with open(self.demo_steps_json) as f:
                    self._collection_steps = json.load(f)['steps']
                rospy.loginfo("[collect] Loaded %d steps from %s",
                              len(self._collection_steps), self.demo_steps_json)
            os.makedirs(self.demo_out_dir, exist_ok=True)
            rospy.Service('/mppi_bridge/start_episode', SetBool, self._start_episode_cb)
            rospy.loginfo("[collect] Demo collection enabled → %s  (target %d episodes)",
                          self.demo_out_dir, self.demo_n_episodes)

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
        # Handle episode start signal from /mppi_bridge/start_episode service.
        if self.collect_demos:
            with self._collection_lock:
                start = self._pending_start
                self._pending_start = False
            if start and not self._collecting:
                self._ep_frames   = []
                self._ep_steps    = 0
                self._n_attempts += 1
                self._collecting  = True
                try:
                    planner.reset_episode(json.dumps(self._collection_steps))
                except Exception as e:
                    rospy.logwarn("[collect] reset_episode failed: %s", e)
                rospy.loginfo("[collect] Episode %d started: %s",
                              self._n_attempts,
                              self._make_task_prompt(self._collection_steps))

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

        parts = [torch.tensor(q, dtype=torch.float32),
                 torch.tensor(dq, dtype=torch.float32)]

        if self.object_tracker is not None:
            obj_states = self.object_tracker.get_states()
            for state in obj_states:
                if state is not None:
                    q_obj, _ = state
                    pos  = torch.tensor(q_obj[:3], dtype=torch.float32)
                    quat = torch.tensor(_euler_xyz_to_quat_wxyz(q_obj[3:6]), dtype=torch.float32)
                    parts.extend([pos, quat])

        dof_state = torch.cat(parts)

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

        # Record frame for demo collection.
        if self.collect_demos and self._collecting and self.enabled:
            with self._camera_lock:
                base_img  = self._camera_images[0]
                wrist_img = self._camera_images[1]
            if base_img is not None and wrist_img is not None:
                action_demo = np.concatenate([
                    (action_np * self.demo_action_dt).astype(np.float32),
                    np.zeros(1, dtype=np.float32),   # gripper placeholder
                ])
                self._ep_frames.append({
                    'base_rgb':  base_img.copy(),
                    'wrist_rgb': wrist_img.copy(),
                    'joints':    q.astype(np.float32),
                    'actions':   action_demo,
                    'task':      self._make_task_prompt(self._collection_steps),
                })
                self._ep_steps += 1

                done    = self._check_success()
                timeout = self._ep_steps >= self.demo_max_steps
                if done or timeout:
                    self._collecting = False
                    if done:
                        self._save_episode()
                        self._n_success += 1
                        rospy.loginfo("[collect] SUCCESS %d/%d  (%d frames)",
                                      self._n_success, self.demo_n_episodes,
                                      self._ep_steps)
                        if self._n_success >= self.demo_n_episodes:
                            rospy.loginfo("[collect] All %d episodes saved → %s",
                                          self._n_success, self.demo_out_dir)
                    else:
                        rospy.loginfo("[collect] Attempt %d timed out — discarding",
                                      self._n_attempts)

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
        rospy.loginfo("Publisher thread started at %d Hz (max_accel=%.2f rad/s²)",
                      self.publish_rate, self.max_accel)
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

        # Rate limiter: cap per-step velocity change to avoid jerk, without phase lag
        action_np = np.clip(action_np,
                            self.last_action - self._max_delta,
                            self.last_action + self._max_delta)
        self.last_action = action_np.copy()

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


    # ------------------------------------------------------------------ #
    # Demo collection helpers
    # ------------------------------------------------------------------ #

    def _camera_cb(self, idx, msg):
        """Cache the latest image from camera idx, resized to 224×224 RGB."""
        try:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
            with self._camera_lock:
                self._camera_images[idx] = img
        except Exception as e:
            rospy.logwarn_throttle(5.0, "[collect] camera %d error: %s", idx, e)

    def _start_episode_cb(self, req):
        """ROS service: queue the start of a new collection episode."""
        if not self.collect_demos:
            return SetBoolResponse(success=False,
                                   message="collect_demos param is False")
        if self._n_success >= self.demo_n_episodes:
            return SetBoolResponse(success=False,
                                   message="Collection already complete")
        with self._collection_lock:
            self._pending_start = True
        return SetBoolResponse(success=True, message="Episode start queued")

    def _check_success(self) -> bool:
        """Return True when every step's block is within demo_success_dist of its goal (2-D)."""
        if self.object_tracker is None or not self._collection_steps:
            return False
        try:
            obj_states = self.object_tracker.get_states()
        except Exception:
            return False
        for step in self._collection_steps:
            obj_idx = step['obj_idx']
            if obj_idx >= len(obj_states) or obj_states[obj_idx] is None:
                return False
            q_obj, _ = obj_states[obj_idx]
            dist = np.linalg.norm(
                np.array(q_obj[:2]) - np.array(step['end_pos'][:2]))
            if dist > self.demo_success_dist:
                return False
        return True

    def _save_episode(self):
        """Write the current episode frames to a compressed .npz file."""
        if not self._ep_frames:
            return
        base_rgb  = np.stack([f['base_rgb']  for f in self._ep_frames])
        wrist_rgb = np.stack([f['wrist_rgb'] for f in self._ep_frames])
        joints    = np.stack([f['joints']    for f in self._ep_frames])
        actions   = np.stack([f['actions']   for f in self._ep_frames])
        task      = self._ep_frames[0]['task']
        path = os.path.join(self.demo_out_dir,
                            f'episode_{self._n_success:04d}.npz')
        np.savez_compressed(path,
                            base_rgb=base_rgb, wrist_rgb=wrist_rgb,
                            joints=joints, actions=actions,
                            task=np.array(task))
        rospy.loginfo("[collect] → %s  (%d frames)", path, len(self._ep_frames))

    @staticmethod
    def _make_task_prompt(steps) -> str:
        if not steps:
            return ""
        parts = [f"push the {s['obj_name']} to ({s['end_pos'][0]:.2f}, {s['end_pos'][1]:.2f})"
                 for s in steps]
        return ", then ".join(parts)


if __name__ == '__main__':
    try:
        node = MPPIBridgeNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
