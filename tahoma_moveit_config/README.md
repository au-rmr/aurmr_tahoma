# tahoma_moveit_config

* SRDF for the Tahoma workcell with Robotiq 2F-85 gripper
* MoveIt Servo cartesian controller configuration
* OMPL, OMPL + CHOMP planning pipeline configurations

## Usage


## Development

    rosrun srdfdom display_srdf $(rospack find tahoma_moveit_config)/config/tahoma.srdf

### Updating collision specs

Part of the SRDF specification is a listing of link pairs to disable collision checking for. This can greatly reduce the cost of motion planning. Whenever we update the robot description however, we must also update this collision specification:

    rosrun moveit_setup_assistant collisions_updater --config-pkg tahoma_moveit_config --verbose --trials 100000
