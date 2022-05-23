# aurmr_perception

ROS package for perception tasks. The current version of this package was designed for the "first pick" milestone and is based on simple heuristics.

## Overview

The package consists of two nodes: `aurmr_perception` and `aurmr_grasping`. The `aurmr_perception` node exposes the following service endpoints:

| Service name                        | Inputs                                                 | Outputs                                                                 |   |   |
|-------------------------------------|--------------------------------------------------------|-------------------------------------------------------------------------|---|---|
| /aurmr_perception/reset_bin         | bin_id (string)                                        | success (bool), message (string)                                        |   |   |
| /aurmr_perception/capture_object    | bin_id (string), object_id (string)                    | success (bool), message (string)                                        |   |   |
| /aurmr_perception/get_object_points | bin_id (string), object_id (string), frame_id (string) | success (bool), message (string), bin_id (string), points (PointCloud2) |   |   |
| /aurmr_perception/remove_object     | bin_id (string), object_id (string)                    | success (bool), message (string)                                        |   |   |

The `aurmr_grasping` node exposes the following service endpoints:
| Service name                 | Inputs                          | Outputs                                              |   |   |
|------------------------------|---------------------------------|------------------------------------------------------|---|---|
| /aurmr_perception/init_grasp | pose_id (int8), grasp_id (int8) | success (bool), message (string), pose (PoseStamped) |   |   |

## Usage

The provided launch script can be used to launch the ROS node:

`roslaunch aurmr_perception aurmr_perception.launch`

Before capturing the points for an object, `reset_bin` must be called to establish a background frame. For example, with rosservice:

`rosservice call /aurmr_perception/reset_bin "BIN_ABC"`

Next, place the object into view of the sensor and call the `capture_object` service endpoint:

`rosservice call /aurmr_perception/capture_object "BIN_ABC" "OBJECT_XYZ"`

After the `capture_object` call, the node will segment the objects pointcloud based on the RGB difference and store those points internally. The points can later be retrieved by calling `get_object_points` with the `bin_id`, `object_id`, and the coordinate `frame_id` to which the points should be transformed:

`rosservice call /aurmr_perception/get_object_points "BIN_ABC" "OBJECT_XYZ" "base_link"`

Finally, after removing an object from the scene `remove_object` can be called to remove the associated pointcloud from memory:

`rosservice call /aurmr_perception/remove_object "BIN_ABC" "OBJECT_XYZ"`


