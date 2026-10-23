#!/usr/bin/env python3
import rospy
from capacitive_cup.msg import CapData


def callback(msg):
    rospy.loginfo(f"Received processed data: {msg.data}")

rospy.init_node("cap_control_listener", anonymous=True)

sub = rospy.Subscriber("/cap_data", CapData, callback)

rospy.spin()
