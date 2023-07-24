cmake_minimum_required(VERSION 3.0.2)
project(aurmr_hri)

find_package(catkin REQUIRED COMPONENTS
    roslint
    std_msgs
    actionlib_msgs
    sensor_msgs
    message_generation
)

add_action_files(
  FILES
  RetryGrasp.action
)

generate_messages(
  DEPENDENCIES
  std_msgs
  actionlib_msgs
  sensor_msgs
)

catkin_package(
)

catkin_install_python(PROGRAMS scripts/emulate_failed_grasp
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

file(GLOB_RECURSE ${PROJECT_NAME}_PY_SCRIPTS
        RELATIVE ${PROJECT_SOURCE_DIR} scripts/*.py)
roslint_python(${${PROJECT_NAME}_PY_SCRIPTS})