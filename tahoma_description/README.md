# tahoma_description

Meshes and URDFs for our UR16e workcell.

## Usage

To view the model

    roslaunch tahoma_description view_model.launch


## Development

Both the workcell and pod models use uncommon xacro features. Read the xacro docs and inspect the compiled URDFs to understand what's going on. This [wiki tutorial on typical patterns](http://wiki.ros.org/urdf/Tutorials/Using%20Xacro%20to%20Clean%20Up%20a%20URDF%20File) is a good starting place.

Make sure you update the SRDF in `tahoma_moveit_config` if you make changes to the robot's URDF.

### Verification

To check the kinematic chain:

    xacro $(rospack find tahoma_description)/robots/tahoma.xacro > /tmp/tahoma.urdf && check_urdf /tmp/tahoma.urdf

To pretty print the full robot specification (not in XML format):

    xacro $(rospack find tahoma_description)/robots/tahoma.xacro | rosrun urdfdom_py display_urdf - 
