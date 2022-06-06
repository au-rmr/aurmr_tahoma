# tahoma_description

Meshes and URDFs for our UR16e workcell.

## Usage

To view the model

    roslaunch tahoma_description view_model.launch


## Development

Make sure you update the SRDF in `tahoma_moveit_config` if you makee changes to the robot's URDF.

### Verification

To check the kinematic chain:

    xacro $(rospack find tahoma_description)/robots/tahoma.xacro > /tmp/tahoma.urdf && check_urdf /tmp/tahoma.urdf

To pretty print the full robot specification (not in XML format):

    xacro $(rospack find tahoma_description)/robots/tahoma.xacro | rosrun urdfdom_py display_urdf - 
