cmake_minimum_required(VERSION 3.0.2)
project(aurmr_unseen_object_clustering)

find_package(catkin REQUIRED COMPONENTS
    roslint
    rospy
    std_msgs
    sensor_msgs
    message_generation
)


catkin_python_setup()

add_service_files(
  FILES
  GetSegmentationUOIS.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
)

catkin_install_python(PROGRAMS nodes/uois_service
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

# Lint Python modules for PEP8 compatibility
set(ROSLINT_PYTHON_OPTS "--max-line-length=120")

file(GLOB_RECURSE ${PROJECT_NAME}_PY_SRC
        RELATIVE ${PROJECT_SOURCE_DIR} src/${PROJECT_NAME}/**/*.py)
roslint_python(${${PROJECT_NAME}_PY_SRC})
