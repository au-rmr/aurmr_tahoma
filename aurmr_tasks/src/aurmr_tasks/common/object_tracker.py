"""
Object tracker for the MPPI bridge — converts AprilTag detections to
Genesis-format object states (position + Euler angles, linear + angular velocity).

Genesis DOF format for free rigid bodies:
  q_block  = [x, y, z, euler_x, euler_y, euler_z]   (intrinsic XYZ, radians)
  dq_block = [vx, vy, vz, wx, wy, wz]                (linear + angular velocity)

Tag layout convention (apriltag_ros):
  - Tag z-axis points AWAY from the tag surface (toward the camera).
  - Tags are physically centered on each cube face.
  - Each face is named: "+x", "-x", "+y", "-y", "+z", "-z" in the cube body frame.

Configuration (via ROS params under a prefix, e.g. ~object_tracking):

  camera_frame: camera_color_optical_frame
  world_frame:  arm_base_link
  cube_size:    0.10          # meters, edge length of the cube
  stack_margin: 0.003         # gap left above a cube when correcting a stacking overlap
  refined_tag_timeout: 0.5    # seconds before a refined_tag_<id> TF is considered stale
  refined_tag_rate:    30.0   # Hz, how often to poll refined_tag_<id> TF frames
  cubes:
    - isaac_index: 0        # index in sim.objects[]
      tags:
        - {id: 0, face: "+x"}
        - {id: 1, face: "-x"}
        - {id: 2, face: "+y"}
        - {id: 3, face: "-y"}
        - {id: 4, face: "+z"}
        - {id: 5, face: "-z"}
    - isaac_index: 1
      tags:
        - {id: 6, face: "+x"}
        ...

NOTE on tag orientation within the face:
  The "standard" orientation below assumes the tag's y-axis (up direction when
  looking at the tag from outside the cube) is aligned with +Z of the cube when
  on side faces, and -Y when on the +Z face.  If your tags are rotated differently
  on any face, you can override via an optional `rotation_deg` key (rotation of
  the tag around its own z-axis, degrees, default 0).
"""

import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import rospy
import tf2_ros
# import tf2_geometry_msgs  # noqa: registers PoseStamped transform support

from aurmr_tasks.msg import AprilTagDetectionArray
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from scipy.spatial.transform import Rotation


# --------------------------------------------------------------------------- #
# Face transforms: T_tag_in_cube
#   Describes the tag frame expressed in the cube body frame.
#   Each entry is (translation, rotation_matrix).
#   Translation: the tag center position in the cube frame = face_center.
#   Rotation: columns are [x_tag, y_tag, z_tag] expressed in cube frame.
#
#   Convention:
#     z_tag = face outward normal in cube frame
#     y_tag = +z_cube for side faces (tag "up" is cube's +Z)
#             -y_cube for +Z face (tag "up" is cube's -Y)
#             +y_cube for -Z face (tag "up" is cube's +Y)
#     x_tag = y_tag cross z_tag (right-hand rule)
#
#   Rotate the tag face-normal onto the cube face with a yaw of `rotation_deg`
#   around the face-normal if the tag is not upright on that face.
# --------------------------------------------------------------------------- #

def _face_rotation(face: str, rotation_deg: float = 0.0) -> np.ndarray:
    """
    Rotation matrix R such that R @ [0,0,1] = outward face normal,
    with the tag's y-axis aligned as described in the module docstring.
    An additional in-plane rotation of `rotation_deg` around the face normal
    can be applied if the tag is not upright.
    """
    # Base rotations: each maps tag-z to the face outward normal
    base = {
        "+x": Rotation.from_euler('y', 90,  degrees=True),
        "-x": Rotation.from_euler('y', -90, degrees=True),
        "+y": Rotation.from_euler('x', -90, degrees=True),
        "-y": Rotation.from_euler('x', 90,  degrees=True),
        "+z": Rotation.from_euler('x', 0,   degrees=True),   # identity
        "-z": Rotation.from_euler('x', 180, degrees=True),
    }
    r = base[face]
    if rotation_deg:
        # Rotate around the tag z-axis (= face normal) by rotation_deg
        r = r * Rotation.from_euler('z', rotation_deg, degrees=True)
    return r.as_matrix()


def _face_translation(face: str, half_size: float) -> np.ndarray:
    """Tag center position in cube body frame = face center."""
    return {
        "+x": np.array([half_size, 0.0, 0.0]),
        "-x": np.array([-half_size, 0.0, 0.0]),
        "+y": np.array([0.0, half_size, 0.0]),
        "-y": np.array([0.0, -half_size, 0.0]),
        "+z": np.array([0.0, 0.0, half_size]),
        "-z": np.array([0.0, 0.0, -half_size]),
    }[face]


def _make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a 3x3 rotation and 3-vector translation."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _quat_to_euler_xyz(qxyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion [x,y,z,w] to intrinsic XYZ Euler angles (radians)."""
    return Rotation.from_quat(qxyzw).as_euler('xyz')


def _tilt_from_level(quat_xyzw: np.ndarray) -> float:
    """Angle (radians) between the cube's most-vertical local axis and world +z.

    0 means some face of the cube is resting flat on the table, regardless of
    which local axis (x, y, or z) is the one pointing (up or down) vertically.
    """
    R = Rotation.from_quat(quat_xyzw).as_matrix()
    best_alignment = np.max(np.abs(R[2, :]))  # |z-component| of each local axis
    return np.arccos(np.clip(best_alignment, -1.0, 1.0))


def _level_quat(quat_xyzw: np.ndarray) -> np.ndarray:
    """Snap whichever local axis is closest to vertical (see _tilt_from_level)
    to be exactly vertical, via the minimal rotation that does so — i.e. rotation
    about that axis (yaw on the resting face) is left untouched.
    """
    R = Rotation.from_quat(quat_xyzw).as_matrix()
    col = np.argmax(np.abs(R[2, :]))
    v = R[:, col]
    target = np.array([0.0, 0.0, 1.0 if R[2, col] >= 0 else -1.0])

    axis = np.cross(v, target)
    axis_norm = np.linalg.norm(axis)
    cos_angle = np.clip(np.dot(v, target), -1.0, 1.0)

    if axis_norm < 1e-8:
        R_align = np.eye(3) if cos_angle > 0 else Rotation.from_rotvec(
            np.pi * _any_perpendicular(v)).as_matrix()
    else:
        R_align = Rotation.from_rotvec(axis / axis_norm * np.arccos(cos_angle)).as_matrix()

    return Rotation.from_matrix(R_align @ R).as_quat()


def _any_perpendicular(v: np.ndarray) -> np.ndarray:
    """An arbitrary unit vector perpendicular to v (for the 180-degree edge case)."""
    other = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    perp = np.cross(v, other)
    return perp / np.linalg.norm(perp)


def _pose_msg_to_T(pose) -> np.ndarray:
    """Convert geometry_msgs/Pose to a 4x4 homogeneous transform."""
    q = pose.orientation
    t = pose.position
    R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    return _make_T(R, np.array([t.x, t.y, t.z]))


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

    def set_z(self, z: float):
        """Override the z-coordinate only (used to resolve stacked-cube overlaps)."""
        with self.lock:
            self.q[2] = z

    def set_orientation(self, quat_xyzw: np.ndarray):
        """Override orientation only (used to level stacked cubes)."""
        with self.lock:
            self.q[3:6] = _quat_to_euler_xyz(quat_xyzw)


# --------------------------------------------------------------------------- #
# Main tracker class
# --------------------------------------------------------------------------- #

class ObjectTracker:
    """
    Subscribes to /tag_detections and maintains Genesis-format state
    for each tracked cube.

    Usage:
        tracker = ObjectTracker()        # reads config from ROS params
        ...
        states = tracker.get_states()    # -> List of (q, dq) per cube
        dof_state = assemble_dof_state(q_robot, dq_robot, states)
    """

    def __init__(self, param_prefix: str = '~object_tracking'):
        cfg = rospy.get_param(param_prefix)
        self.camera_frame: str = cfg.get('camera_frame', 'camera_color_optical_frame')
        self.world_frame: str  = cfg.get('world_frame',  'base_link')
        self.cube_size: float  = float(cfg.get('cube_size', 0.05))
        # Extra gap left above the lower cube when a vertical-overlap correction is
        # applied, so the corrected pose doesn't sit flush against it.
        self.stack_margin: float = float(cfg.get('stack_margin', 0.003))
        # Max age (seconds) of a refined_tag_<id> TF before it's considered stale —
        # i.e. the gripper camera no longer currently sees that tag.
        self.refined_tag_timeout: float = float(cfg.get('refined_tag_timeout', 5.5))
        refined_tag_rate: float = float(cfg.get('refined_tag_rate', 30.0))
        # Minimum tilt improvement (degrees) a candidate tag needs over the
        # currently-published orientation before we switch to it — otherwise two
        # near-tied visible faces flip-flop the chosen orientation every cycle.
        self.orientation_hysteresis: float = np.deg2rad(
            float(cfg.get('orientation_hysteresis_deg', 3.0)))
        cube_size              = self.cube_size
        half                   = cube_size / 2.0

        # tag_id -> (cube_index, T_tag_in_cube 4x4)
        self._tag_to_cube: Dict[int, Tuple[int, np.ndarray]] = {}

        cubes_cfg: List[dict] = cfg.get('cubes', [])
        n_cubes = max((c['isaac_index'] for c in cubes_cfg), default=-1) + 1
        self._states: List[CubeState] = [CubeState() for _ in range(n_cubes)]

        for cube_cfg in cubes_cfg:
            idx = int(cube_cfg['isaac_index'])
            for tag_cfg in cube_cfg.get('tags', []):
                tag_id  = int(tag_cfg['id'])
                face    = str(tag_cfg['face'])
                rot_deg = float(tag_cfg.get('rotation_deg', 0.0))
                R = _face_rotation(face, rot_deg)
                t = _face_translation(face, half)
                T_tag_in_cube = _make_T(R, t)
                self._tag_to_cube[tag_id] = (idx, T_tag_in_cube)

        if not self._tag_to_cube:
            rospy.logwarn("ObjectTracker: no tags configured — check ROS params")

        # cube_index -> time.time() of its last refined_tag_<id>-sourced update.
        # Used so the fixed-camera path defers to the gripper camera while it's live.
        self._last_refined_update: Dict[int, float] = {}

        # TF
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Subscribe to detections
        self._sub = rospy.Subscriber(
            '/fused_tag_detections', AprilTagDetectionArray, self._detection_cb,
            queue_size=1)

        # Poll the gripper-mounted camera's refined_tag_<id> TF frames independently
        # of the fixed-camera detection stream, since they update on their own schedule.
        self._refined_timer = rospy.Timer(
            rospy.Duration(1.0 / refined_tag_rate), self._refined_tag_timer_cb)

        # Publish object poses in world frame — one Pose per cube, indexed by isaac_index
        self._pose_pub = rospy.Publisher(
            '/mppi_bridge/object_poses', PoseArray, queue_size=1)

    # ------------------------------------------------------------------ #
    # Callback
    # ------------------------------------------------------------------ #

    def _detection_cb(self, msg: AprilTagDetectionArray):
        if not msg.detections:
            return

        # Look up camera -> world once per message
        try:
            tf_cam_world = self.tf_buffer.lookup_transform(
                self.world_frame, self.camera_frame,
                msg.header.stamp, rospy.Duration(0.1))
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(2.0, "ObjectTracker TF lookup failed: %s", e)
            return

        T_cam_world = self._tf_to_matrix(tf_cam_world)

        # Accumulate per-cube estimates (may have multiple tags per cube visible)
        # cube_index -> list of (pos, quat, distance_from_camera)
        estimates: Dict[int, List[Tuple[np.ndarray, np.ndarray, float]]] = {}

        for det in msg.detections:
            # apriltag_ros can bundle multiple IDs per detection (tag bundles)
            # but normally each detection has one id
            tag_ids = det.id if hasattr(det.id, '__iter__') else [det.id]
            for tag_id in tag_ids:
                if tag_id not in self._tag_to_cube:
                    continue
                cube_idx, T_tag_in_cube = self._tag_to_cube[tag_id]

                # Tag pose in camera frame
                T_tag_cam = _pose_msg_to_T(det.pose.pose.pose)

                # Tag pose in world frame
                T_tag_world = T_cam_world @ T_tag_cam

                # Cube pose in world frame
                T_cube_world = T_tag_world @ np.linalg.inv(T_tag_in_cube)

                pos  = T_cube_world[:3, 3]
                quat = Rotation.from_matrix(T_cube_world[:3, :3]).as_quat()

                distance = np.linalg.norm(T_tag_cam[:3, 3])

                estimates.setdefault(cube_idx, []).append((pos, quat, distance))

        # Fuse and update each cube that had at least one visible tag — unless the
        # gripper camera is currently tracking it more accurately (see _is_refined_live).
        for cube_idx, est_list in estimates.items():
            if cube_idx >= len(self._states) or self._is_refined_live(cube_idx):
                continue

            pos, quat = self._fuse_estimates(cube_idx, est_list)
            self._states[cube_idx].update(pos, quat, msg.header.stamp)

        self._resolve_vertical_overlaps()
        self._publish_poses(msg.header.stamp)

    def _fuse_estimates(
            self, cube_idx: int,
            est_list: List[Tuple[np.ndarray, np.ndarray, float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Combine one or more (pos, quat, distance) estimates for a single cube."""
        if len(est_list) == 1:
            pos, quat, _ = est_list[0]
            return pos, self._debounce_orientation(cube_idx, quat)

        # Weighted mean position (weight = 1/distance^2, closer = more accurate)
        weights = np.array([1.0 / (d ** 2 + 1e-6) for _, _, d in est_list])
        weights /= weights.sum()
        pos = sum(w * p for (p, _, _), w in zip(est_list, weights))
        # For orientation: cubes rest flat on some face ~99% of the time, and
        # off-axis viewing angle / image decimation inflates apparent tilt far
        # more than it distorts position, so trust whichever tag reads flattest.
        best = min(est_list, key=lambda e: _tilt_from_level(e[1]))
        return pos, self._debounce_orientation(cube_idx, best[1])

    def _debounce_orientation(self, cube_idx: int, quat: np.ndarray) -> np.ndarray:
        """Keep publishing the cube's current orientation unless the candidate is
        at least orientation_hysteresis flatter — otherwise two near-tied visible
        faces (e.g. viewed near an edge) flip-flop which one "wins" every cycle
        purely from detection noise, instead of only switching on a real change.
        """
        state = self._states[cube_idx]
        if not state.valid:
            return quat
        prev_q, _ = state.get()
        prev_quat = Rotation.from_euler('xyz', prev_q[3:6]).as_quat()
        prev_tilt = _tilt_from_level(prev_quat)
        candidate_tilt = _tilt_from_level(quat)
        if prev_tilt - candidate_tilt < self.orientation_hysteresis:
            return prev_quat
        return quat

    def _is_refined_live(self, cube_idx: int) -> bool:
        """True if the gripper camera has updated this cube more recently than
        refined_tag_timeout — i.e. it's currently tracking it, so the fixed-camera
        path should defer instead of overwriting with a lower-quality estimate."""
        last = self._last_refined_update.get(cube_idx)
        return last is not None and (time.time() - last) < self.refined_tag_timeout

    def _refined_tag_timer_cb(self, _event=None):
        """Poll refined_tag_<id> TF frames from the gripper camera. Unlike the fixed
        cameras (raw PoseStamped in camera_frame, fused externally into one topic),
        these are published directly into the TF tree, so no separate camera->world
        composition is needed — tf2 resolves the whole chain to world_frame itself.
        """
        now = rospy.Time.now()
        estimates: Dict[int, List[Tuple[np.ndarray, np.ndarray, float]]] = {}

        for tag_id, (cube_idx, T_tag_in_cube) in self._tag_to_cube.items():
            try:
                tf_tag_world = self.tf_buffer.lookup_transform(
                    self.world_frame, f"refined_tag_{tag_id}",
                    rospy.Time(0), rospy.Duration(0.05))
            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                continue

            age = (now - tf_tag_world.header.stamp).to_sec()
            if age > self.refined_tag_timeout:
                continue

            T_tag_world = self._tf_to_matrix(tf_tag_world)
            T_cube_world = T_tag_world @ np.linalg.inv(T_tag_in_cube)
            pos  = T_cube_world[:3, 3]
            quat = Rotation.from_matrix(T_cube_world[:3, :3]).as_quat()
            estimates.setdefault(cube_idx, []).append((pos, quat, 1.0))

        if not estimates:
            return

        for cube_idx, est_list in estimates.items():
            if cube_idx >= len(self._states):
                continue
            pos, quat = self._fuse_estimates(cube_idx, est_list)
            self._states[cube_idx].update(pos, quat, now)
            self._last_refined_update[cube_idx] = time.time()

        self._resolve_vertical_overlaps()
        self._publish_poses(now)

    def _resolve_vertical_overlaps(self):
        """Objects can't be inside each other: if two cubes' footprints overlap
        horizontally and their vertical extents interpenetrate, lift the higher
        one so it rests on top of the lower one instead of clipping into it.

        Repeats to convergence (bounded by cube count) rather than a single pass,
        since a 3+-cube stack needs corrections to propagate: e.g. resolving
        (bottom, middle) first, then re-checking (middle, top) against middle's
        just-corrected height instead of its stale raw reading.
        """
        half = self.cube_size / 2.0
        for _ in range(len(self._states)):
            changed = False
            valid = [s for s in self._states if s.valid]
            for a in range(len(valid)):
                si = valid[a]
                qi, _ = si.get()
                for b in range(a + 1, len(valid)):
                    sj = valid[b]
                    qj, _ = sj.get()

                    dxy = np.linalg.norm(qi[:2] - qj[:2])
                    if dxy >= self.cube_size:
                        continue  # footprints don't overlap horizontally

                    bottom_i, top_i = qi[2] - half, qi[2] + half
                    bottom_j, top_j = qj[2] - half, qj[2] + half
                    if bottom_i >= top_j or bottom_j >= top_i:
                        continue  # no vertical overlap

                    changed = True
                    if qi[2] >= qj[2]:
                        self._snap_onto(si, qi, top_j, half)
                    else:
                        self._snap_onto(sj, qj, top_i, half)

            if not changed:
                break

    def _snap_onto(self, state: 'CubeState', q: np.ndarray, lower_top: float, half: float):
        """Place a stacked cube's bottom face on top of the cube below it, and
        level its orientation — a cube resting on another flat cube is flat too.
        """
        q[2] = lower_top + half + self.stack_margin
        leveled = _level_quat(Rotation.from_euler('xyz', q[3:6]).as_quat())
        q[3:6] = _quat_to_euler_xyz(leveled)
        state.set_z(q[2])
        state.set_orientation(leveled)

    def _publish_poses(self, stamp):
        pa = PoseArray()
        pa.header.stamp = stamp
        pa.header.frame_id = self.world_frame
        for state in self._states:
            p = Pose()
            if state.valid:
                q, _ = state.get()
                p.position.x, p.position.y, p.position.z = q[0], q[1], q[2]
                quat = Rotation.from_euler('xyz', q[3:6]).as_quat()  # [x,y,z,w]
                p.orientation.x = quat[0]
                p.orientation.y = quat[1]
                p.orientation.z = quat[2]
                p.orientation.w = quat[3]
            else:
                p.orientation.w = 1.0  # identity, position stays at origin
            pa.poses.append(p)
        self._pose_pub.publish(pa)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_states(self) -> List[Optional[Tuple[np.ndarray, np.ndarray]]]:
        """
        Returns a list of (q, dq) for each cube, in isaac_index order.
        Entry is None if that cube has no valid detection yet.
        """
        result = []
        for state in self._states:
            if state.valid:
                result.append(state.get())
            else:
                result.append(None)
        return result

    def all_valid(self) -> bool:
        """True if every tracked cube has at least one detection."""
        return all(s.valid for s in self._states)

    def num_cubes(self) -> int:
        return len(self._states)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tf_to_matrix(tf_stamped) -> np.ndarray:
        t = tf_stamped.transform.translation
        r = tf_stamped.transform.rotation
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
        T[:3, 3]  = [t.x, t.y, t.z]
        return T