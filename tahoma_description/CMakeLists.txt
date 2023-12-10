cmake_minimum_required(VERSION 3.0.2)
project(tahoma_description)

find_package(catkin REQUIRED)
catkin_package()

install(DIRECTORY config launch meshes urdf robots
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

catkin_install_python(PROGRAMS scripts/pod_location_calibration
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

# unit tests are enabled selectively
if (CATKIN_ENABLE_TESTING)
  find_package(roslaunch REQUIRED)
  roslaunch_add_file_check(launch)
endif()
