#!/usr/bin/env bash
switch_group_sim() {
  rosservice call /controller_manager/switch_controller "start_controllers: ['joint_group_pos_controller']
stop_controllers: ['pos_joint_traj_controller']
strictness: 1
start_asap: false
timeout: 0.0"

}

switch_traj_sim() {
  rosservice call /controller_manager/switch_controller "start_controllers: ['pos_joint_traj_controller']
stop_controllers: ['joint_group_pos_controller']
strictness: 1
start_asap: false
timeout: 0.0"
}


switch_group() {
  rosservice call /controller_manager/switch_controller "start_controllers: ['joint_group_pos_controller']
stop_controllers: ['scaled_pos_joint_traj_controller']
strictness: 1
start_asap: false
timeout: 0.0"

}

switch_traj() {
  rosservice call /controller_manager/switch_controller "start_controllers: ['scaled_pos_joint_traj_controller']
stop_controllers: ['joint_group_pos_controller']
strictness: 1
start_asap: false
timeout: 0.0"
}