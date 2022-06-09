cmake_minimum_required(VERSION 3.0.2)
project(aurmr_tasks)

find_package(catkin REQUIRED COMPONENTS
    roslint
)

catkin_python_setup()


catkin_package(
)

catkin_install_python(PROGRAMS scripts/analyze_workspace scripts/move_to_calibration_poses scripts/pick scripts/stow_and_detect_grasps
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

# Lint Python modules for PEP8 compatibility
set(ROSLINT_PYTHON_OPTS "--max-line-length=120")

file(GLOB_RECURSE ${PROJECT_NAME}_PY_SRC
        RELATIVE ${PROJECT_SOURCE_DIR} src/${PROJECT_NAME}/**/*.py)
roslint_python(${${PROJECT_NAME}_PY_SRC})

file(GLOB_RECURSE ${PROJECT_NAME}_PY_SCRIPTS
        RELATIVE ${PROJECT_SOURCE_DIR} scripts/*.py)
roslint_python(${${PROJECT_NAME}_PY_SCRIPTS})