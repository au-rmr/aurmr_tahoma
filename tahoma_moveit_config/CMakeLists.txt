cmake_minimum_required(VERSION 3.0.2)
project(tahoma_moveit_config)

find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(catkin REQUIRED COMPONENTS
        actionlib_msgs
        geometry_msgs
        message_generation
        std_srvs
)

add_action_files(
        DIRECTORY action
        FILES ServoToPose.action
)

generate_messages(
        DEPENDENCIES actionlib_msgs geometry_msgs std_msgs  # Or other packages containing msgs
)

catkin_package(INCLUDE_DIRS include
        CATKIN_DEPENDS
        actionlib_msgs
        geometry_msgs
        message_runtime
        std_srvs)

include_directories(include)
include_directories(SYSTEM
        ${catkin_INCLUDE_DIRS}
        )


if (CATKIN_ENABLE_TESTING)
  find_package(roslaunch REQUIRED)
  roslaunch_add_file_check(tests/moveit_planning_execution.xml)
endif()

install(DIRECTORY config launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

catkin_add_env_hooks(${PROJECT_NAME} SHELLS bash zsh DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/env-hooks)
