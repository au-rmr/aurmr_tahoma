cmake_minimum_required(VERSION 3.0.2)
project(vacuum_gripper_control)
find_package(catkin REQUIRED COMPONENTS message_generation roscpp rospy)


add_message_files(
  FILES
  vacuum_gripper_input.msg
  vacuum_gripper_output.msg
)


generate_messages()

catkin_package(
  # INCLUDE_DIRS include
  CATKIN_DEPENDS message_runtime roscpp rospy
  # LIBRARIES ${robotiq_ethercat_LIBRARIES}
)


catkin_install_python(PROGRAMS scripts/blow_off_gripper_control scripts/close_ejector scripts/evacuate_ejector scripts/suction_gripper_test
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
