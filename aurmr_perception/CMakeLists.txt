cmake_minimum_required(VERSION 3.0.2)
project(aurmr_perception)

find_package(catkin REQUIRED COMPONENTS
    roslint
)


catkin_python_setup()


catkin_package(
)


# Lint Python modules for PEP8 compatibility
set(ROSLINT_PYTHON_OPTS "--max-line-length=120")

file(GLOB_RECURSE ${PROJECT_NAME}_PY_SRC
        RELATIVE ${PROJECT_SOURCE_DIR} src/${PROJECT_NAME}/**/*.py)
roslint_python(${${PROJECT_NAME}_PY_SRC})

file(GLOB_RECURSE ${PROJECT_NAME}_PY_SCRIPTS
        RELATIVE ${PROJECT_SOURCE_DIR} scripts/*.py)
roslint_python(${${PROJECT_NAME}_PY_SCRIPTS})