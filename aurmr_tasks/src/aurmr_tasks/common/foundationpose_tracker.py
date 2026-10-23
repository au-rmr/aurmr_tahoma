"""
FoundationPose tracker for the MPPI bridge.

Drop-in replacement for ObjectTracker. Instead of subscribing to AprilTag
detections, this tracker pulls 4×4 pose matrices (object-in-camera) from a
ZMQ PUSH socket published by run_realsense.py running inside the FoundationPose
Docker container (--network host).

The poses are transformed into the robot world frame using TF2, then stored in
the same CubeState format that mppi_bridge_node.py expects.

Configuration (same ROS param prefix as ObjectTracker, e.g. ~object_tracking):

  camera_frame: camera_color_optical_frame
  world_frame:  base_link
  zmq_address:  tcp://localhost:5555   # address of the ZMQ PUSH socket
  objects:                             # one entry per tracked object
    - id: 0                            # must match the 'id' field in run_realsense.py config
      genesis_index: 0                 # position in sim.objects[]
    - id: 1
      genesis_index: 1
"""

import json
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import rospy
import tf2_ros
import zmq
from scipy.spatial.transform import Rotation


def _tf_to_matrix(tf_stamped) -> np.ndarray:
    t = tf_stamped.transform.translation
    r = tf_stamped.transform.rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
    T[:3, 3]  = [t.x, t.y, t.z]
    return T

def _quat_to_euler_xyz(qxyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion [x,y,z,w] to intrinsic XYZ Euler angles (radians)."""
    return Rotation.from_quat(qxyzw).as_euler('xyz')


# --------------------------------------------------------------------------- #
# Per-cube state
# --------------------------------------------------------------------------- #

class CubeState:
    """Tracks the estimated pose and velocity of one cube."""

    def __init__(self):
        # Latest pose in world frame: [x, y, z, ex, ey, ez]
        self.q = np.zeros(6)
        # Latest velocity estimate: [vx, vy, vz, wx, wy, wz]
        self.dq = np.zeros(6)
        # Pose at previous estimate (for finite-difference velocity)
        self._prev_pos = None
        self._prev_quat = None  # [x,y,z,w]
        self._prev_time = None
        self.valid = False
        self.lock = threading.Lock()

    def update(self, pos: np.ndarray, quat_xyzw: np.ndarray, stamp: rospy.Time):
        """Update pose from a new detection. Finite-differences for velocity."""
        euler = _quat_to_euler_xyz(quat_xyzw)
        now = rospy.Time.now().to_sec()

        with self.lock:
            if self._prev_pos is not None and self._prev_time is not None:
                dt = now - self._prev_time
                if 1e-4 < dt < 1.0:   # sanity bounds: between 0.1 ms and 1 s
                    vxyz = (pos - self._prev_pos) / dt
                    # Angular velocity from quaternion difference
                    dR = (Rotation.from_quat(quat_xyzw) *
                          Rotation.from_quat(self._prev_quat).inv())
                    wxyz = dR.as_rotvec() / dt
                    self.dq = np.concatenate([vxyz, wxyz])

            self.q = np.array([pos[0], pos[1], pos[2],
                                euler[0], euler[1], euler[2]])
            self._prev_pos = pos.copy()
            self._prev_quat = quat_xyzw.copy()
            self._prev_time = now
            self.valid = True

    def get(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (q, dq) copies."""
        with self.lock:
            return self.q.copy(), self.dq.copy()

class FoundationPoseTracker:
    """
    Pulls object poses from a ZMQ PUSH socket (published by run_realsense.py),
    transforms them into the robot world frame via TF2, and maintains a
    CubeState per tracked object.

    Public API mirrors ObjectTracker:
      get_states() -> List[Optional[Tuple[np.ndarray, np.ndarray]]]
      all_valid()  -> bool
      num_cubes()  -> int
    """

    def __init__(self, param_prefix: str = '~object_tracking'):
        cfg = rospy.get_param(param_prefix)

        self.camera_frame: str = cfg.get('camera_frame', 'camera_color_optical_frame')
        self.world_frame:  str = cfg.get('world_frame',  'base_link')
        zmq_address:       str = cfg.get('zmq_address',  'tcp://localhost:5555')

        objects_cfg: List[dict] = cfg.get('objects', [])
        n_objects = max((o['genesis_index'] for o in objects_cfg), default=-1) + 1

        # FoundationPose object id -> genesis index
        self._id_to_idx: Dict[int, int] = {
            int(o['id']): int(o['genesis_index']) for o in objects_cfg
        }
        self._states: List[CubeState] = [CubeState() for _ in range(n_objects)]

        # TF2
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # ZMQ PULL — connects to the PUSH socket in the Docker container
        self._zmq_ctx  = zmq.Context()
        self._zmq_sock = self._zmq_ctx.socket(zmq.PULL)
        self._zmq_sock.setsockopt(zmq.RCVTIMEO, 100)  # ms; lets the thread exit cleanly
        self._zmq_sock.setsockopt(zmq.RCVHWM, 2)      # drop stale frames
        self._zmq_sock.connect(zmq_address)

        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._zmq_loop, daemon=True,
                                        name='foundationpose_zmq')
        self._thread.start()

        rospy.loginfo(
            "FoundationPoseTracker ready: %d objects, zmq='%s', "
            "camera='%s', world='%s'",
            n_objects, zmq_address, self.camera_frame, self.world_frame)

        rospy.on_shutdown(self.shutdown)

    # ------------------------------------------------------------------ #
    # Background ZMQ receive loop
    # ------------------------------------------------------------------ #

    def _zmq_loop(self):
        rospy.loginfo("FoundationPoseTracker ZMQ thread started")
        try:
            while not self._stop.is_set():
                try:
                    raw = self._zmq_sock.recv_string()
                except zmq.Again:
                    continue
                except zmq.ZMQError as e:
                    if not self._stop.is_set():
                        rospy.logwarn_throttle(2.0, "FoundationPoseTracker ZMQ error: %s", e)
                    continue

                try:
                    msg = json.loads(raw)
                    # rospy.loginfo(f"[FP] msg received, {len(msg.get('objects', []))} objects")
                    self._process_msg(msg)
                except Exception as e:
                    rospy.loginfo(f"[FP] bad message: {e}")
        except Exception as e:
            rospy.logerr("FoundationPoseTracker ZMQ thread crashed: %s", e)

    def _process_msg(self, msg: dict):
        # Look up camera → world transform. rospy.Time(0) uses the latest
        # available transform, avoiding extrapolation errors.
        try:
            tf_cam_world = self.tf_buffer.lookup_transform(
                self.world_frame, self.camera_frame,
                rospy.Time(0), rospy.Duration(0.1))
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            print(f"[FP] TF lookup failed ({self.camera_frame} -> {self.world_frame}): {e}")
            return

        T_cam_world = _tf_to_matrix(tf_cam_world)
        stamp       = rospy.Time.now()

        for obj in msg.get('objects', []):
            rospy.loginfo_throttle(2.0, "FoundationPoseTracker msg: id=%s valid=%s known=%s",
                                   obj.get('id'), obj.get('valid'), obj.get('id') in self._id_to_idx)
            if not obj.get('valid', False):
                continue
            obj_id = int(obj['id'])
            if obj_id not in self._id_to_idx:
                continue

            # pose is 16 floats: row-major 4×4, object-in-camera
            T_obj_cam   = np.array(obj['pose'], dtype=np.float64).reshape(4, 4)
            T_obj_world = T_cam_world @ T_obj_cam

            pos  = T_obj_world[:3, 3]
            quat = Rotation.from_matrix(T_obj_world[:3, :3]).as_quat()  # [x,y,z,w]

            idx = self._id_to_idx[obj_id]
            if idx < len(self._states):
                self._states[idx].update(pos, quat, stamp)

    # ------------------------------------------------------------------ #
    # Public API (matches ObjectTracker)
    # ------------------------------------------------------------------ #

    def get_states(self) -> List[Optional[Tuple[np.ndarray, np.ndarray]]]:
        """Returns (q, dq) per object in genesis_index order; None if not yet seen."""
        return [s.get() if s.valid else None for s in self._states]

    def all_valid(self) -> bool:
        return all(s.valid for s in self._states)

    def num_cubes(self) -> int:
        return len(self._states)

    def shutdown(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._zmq_sock.close()
        self._zmq_ctx.term()
