# tahoma_description

Meshes and URDFs for our UR16e workcell.

## Usage

To view the model

    roslaunch tahoma_description view_models.launch

## Calibration

### Robot Model

Place the end effector in collision with the frame and manually tweak values until the robot model properly reflects the collision.

### Pod position

Manually tweak values in `pod_transform.launch` until the point cloud matches the model.

### Fixed cameras (eye to hand)

Use moveit_calibration. Place the calibration target in the gripper.

* Sensor frame: camera_lower_right_link (root of camera system)
* Object frame: handeye_target
* End-effector frame: arm_tool0
* Robot base frame: camera_lower_right_mount (mount point)

The tool will produce the static transform between the "base frame" and the "sensor frame". 

[Advice for taking good calibration images](https://github.com/ros-planning/moveit_calibration/issues/89#issuecomment-846465865):
>1. Keep the camera close to the target. Partial views are fine, as long as the target is still being detected.
>2. Between each sample, be sure to include a large rotation (30 degrees) of the camera/EEF.
>3. Make subsequent rotations around different axes.

### End-effector camera (eye in hand)

Use moveit_calibration. Place the calibration target statically in the scene.

* Sensor frame: camera_hand_link (root of camera system)
* Object frame: handeye_target
* End-effector frame: arm_tool0
* Robot base frame: camera_hand_mount (mount point)

The tool will produce the static transform between the "base frame" and the "sensor frame". 