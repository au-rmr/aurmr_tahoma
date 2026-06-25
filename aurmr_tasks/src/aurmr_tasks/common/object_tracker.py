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
  cubes:
    - genesis_index: 0        # index in sim.objects[]
      tags:
        - {id: 0, face: "+x"}
        - {id: 1, face: "-x"}
        - {id: 2, face: "+y"}
        - {id: 3, face: "-y"}
        - {id: 4, face: "+z"}
        - {id: 5, face: "-z"}
    - genesis_index: 1
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import rospy
import tf2_ros
import tf2_geometry_msgs  # noqa: registers PoseStamped transform support

from apriltag_ros.msg import AprilTagDetectionArray
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
        # Use ROS wall clock instead of message stamp to avoid issues with
        # clock skew or bags played back at non-realtime speed.
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
        self.world_frame: str  = cfg.get('world_frame',  'arm_base_link')
        cube_size: float       = float(cfg.get('cube_size', 0.10))
        half                   = cube_size / 2.0

        # tag_id -> (cube_index, T_tag_in_cube 4x4)
        self._tag_to_cube: Dict[int, Tuple[int, np.ndarray]] = {}

        cubes_cfg: List[dict] = cfg.get('cubes', [])
        n_cubes = max((c['genesis_index'] for c in cubes_cfg), default=-1) + 1
        self._states: List[CubeState] = [CubeState() for _ in range(n_cubes)]

        for cube_cfg in cubes_cfg:
            idx = int(cube_cfg['genesis_index'])
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

        # TF
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Subscribe to detections
        self._sub = rospy.Subscriber(
            '/tag_detections', AprilTagDetectionArray, self._detection_cb,
            queue_size=1)

        # Publish object poses in world frame — one Pose per cube, indexed by genesis_index
        self._pose_pub = rospy.Publisher(
            '/mppi_bridge/object_poses', PoseArray, queue_size=1)

        rospy.loginfo("ObjectTracker ready: %d cubes, %d tags, "
                      "camera='%s', world='%s', cube_size=%.3f m",
                      n_cubes, len(self._tag_to_cube),
                      self.camera_frame, self.world_frame, cube_size)

    # ------------------------------------------------------------------ #
    # Callback
    # ------------------------------------------------------------------ #

    def _detection_cb(self, msg: AprilTagDetectionArray):
        if not msg.detections:
            return

        # Look up camera -> world using the latest available transform.
        # rospy.Time(0) avoids extrapolation errors caused by TF/detection
        # timestamp skew, which is acceptable here since the camera and robot
        # base are both static (or slow-moving) relative to each other.
        try:
            tf_cam_world = self.tf_buffer.lookup_transform(
                self.world_frame, self.camera_frame,
                rospy.Time(0), rospy.Duration(0.1))
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

                rospy.loginfo_throttle(1.0,
                    "tag_%d | in_cam: [%.3f %.3f %.3f] | in_world: [%.3f %.3f %.3f] | cube: [%.3f %.3f %.3f]",
                    tag_id,
                    T_tag_cam[0,3], T_tag_cam[1,3], T_tag_cam[2,3],
                    T_tag_world[0,3], T_tag_world[1,3], T_tag_world[2,3],
                    pos[0], pos[1], pos[2])

                estimates.setdefault(cube_idx, []).append((pos, quat, distance))

        # Fuse and update each cube that had at least one visible tag
        for cube_idx, est_list in estimates.items():
            if cube_idx >= len(self._states):
                continue

            if len(est_list) == 1:
                pos, quat, _ = est_list[0]
            else:
                # Weighted mean position (weight = 1/distance^2, closer = more accurate)
                weights = np.array([1.0 / (d ** 2 + 1e-6) for _, _, d in est_list])
                weights /= weights.sum()
                pos = sum(w * p for (p, _, _), w in zip(est_list, weights))
                # For orientation: use the detection closest to the camera
                best = min(est_list, key=lambda e: e[2])
                quat = best[1]

            self._states[cube_idx].update(pos, quat, msg.header.stamp)

        self._publish_poses(msg.header.stamp)

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
        Returns a list of (q, dq) for each cube, in genesis_index order.
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
