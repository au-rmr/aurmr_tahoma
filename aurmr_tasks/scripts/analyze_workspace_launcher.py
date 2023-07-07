#!/usr/bin/env python

import rospy
import subprocess

def main():
    pod_x = rospy.get_param('~pod_x', 'default_arg1_value')
    pod_y = rospy.get_param('~pod_y', 'default_arg2_value')

    cmd = ['python', '/path/to/my_script.py', pod_x, pod_y]

    subprocess.call(cmd)

if __name__ == '__main__':
    rospy.init_node('my_script_launcher')
    main()