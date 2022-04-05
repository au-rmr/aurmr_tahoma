# aurmr_tahoma

ROS workspace for working with AURMR's UR16e workcell ("Tahoma"). Targets ROS Noetic.

## Usage

To install most of the dependencies, run this from your workspace's `src/` directory:

    rosdep install --ignore-src --from-paths . -y -r

Many of the UR dependencies are in flux, with important changes existing only on obscure branches or forks. We use submodules
to keep track of pointers to the correct versions of these dependencies.

To clone with submodules:

  git clone --recurse-submodules git@github.com:au-rmr/aurmr_tahoma.git
