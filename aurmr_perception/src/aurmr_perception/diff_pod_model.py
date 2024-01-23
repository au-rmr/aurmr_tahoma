import cv2
import numpy as np
from collections import defaultdict
from aurmr_perception.pod_model import PodModel

class DiffPodModel(PodModel):

    def __init__(self, dataset, diff_threshold) -> None:
        super().__init__(dataset)
        self.diff_threshold = diff_threshold
        self.latest_captures = {}
        self.latest_masks = {}
        self.points_table = {}


    def capture_object(self, bin_id, object_id, rgb_image, depth_image, camera_intrinsics):
        if not bin_id or not object_id:
            return False

        if bin_id not in self.latest_captures:
            return False, f"Bin {bin_id} has not been reset"

        last_rgb = self.latest_captures[bin_id]
        current_rgb = rgb_image

        # Get the difference of the two captured RGB images
        difference = cv2.absdiff(last_rgb, current_rgb)

        # Threshold the difference image to get the initial mask
        mask = difference.sum(axis=2) >= self.diff_threshold

        # Group the masked pixels and leave only the group with the largest area
        (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8, cv2.CV_32S)
        areas = np.array([stats[i, cv2.CC_STAT_AREA] for i in range(len(stats))])
        max_area = np.max(areas[1:])
        max_area_idx = np.where(areas == max_area)[0][0]
        mask[np.where(labels != max_area_idx)] = 0.0

        # Apply mask to the depth image
        masked_depth = depth_image * mask

        # Get intrinsics reported via camera_info (focal point and principal point offsets)
        fp_x = camera_intrinsics[0]
        fp_y = camera_intrinsics[4]
        ppo_x = camera_intrinsics[2]
        ppo_y = camera_intrinsics[5]

        # Convert the masked depth into a point cloud
        height, width = masked_depth.shape
        nx = np.linspace(0, width-1, width)
        ny = np.linspace(0, height-1, height)
        u, v = np.meshgrid(nx, ny)
        x = (u.flatten() - ppo_x) / fp_x
        y = (v.flatten() - ppo_y) / fp_y
        z = masked_depth.flatten() / 1000
        x = np.multiply(x,z)
        y = np.multiply(y,z)

        x = x[np.nonzero(z)]
        y = y[np.nonzero(z)]
        z = z[np.nonzero(z)]

        # Rearrange axes to match ROS axes
        # Camera: X- right, Y- down, Z- forward
        # ROS: X- forward, Y- left, Z- up
        x_ros = z
        y_ros = -x
        z_ros = -y

        if bin_id not in self.points_table:
            self.points_table[bin_id] = {}

        self.points_table[bin_id][object_id] = np.vstack((x_ros,y_ros,z_ros))

        self.latest_captures[bin_id] = current_rgb
        self.latest_masks[bin_id] = mask

        return True, f"Object {object_id} in bin {bin_id} has been captured with {x.shape[0]} points."

    def get_object(self, bin_id, object_id):
        if not object_id:
            return False, "object_id is required"

        if bin_id not in self.points_table:
            return False, f"Bin {bin_id} was not found"

        if object_id not in self.points_table[bin_id]:
            return False, f"Object {object_id} was not found in bin {bin_id}"

        points = self.points_table[bin_id][object_id]
        return (bin_id, points), None

    def remove_object(self, bin_id, object_id, rgb_image):
        if not bin_id or not object_id:
            return False, "bin_id and object_id are required"

        if bin_id not in self.points_table:
            return False, f"Bin {bin_id} was not found"

        if object_id not in self.points_table[bin_id]:
            return False, f"Object {object_id} was not found in bin {bin_id}"

        del self.points_table[bin_id][object_id]
        self.latest_captures[bin_id] = rgb_image

        return True, None

    def reset(self, bin_id, rgb_image):
        if not bin_id:
            return False

        self.latest_captures[bin_id] = rgb_image

        return True
