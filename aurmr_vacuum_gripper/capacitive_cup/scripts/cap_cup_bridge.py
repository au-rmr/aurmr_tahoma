#!/usr/bin/env python3
import rospy
from capacitive_cup.msg import CapData

import socket
import struct

def pipe_data(sock: socket.socket) -> None:
    """ Scans for UDP packets from capcup device and publishes to ROS topic if found.
    """
    rospy.init_node('cap_cup_bridge', anonymous=True)
    pub = rospy.Publisher('/cap_data', CapData, queue_size=10)
    rate = rospy.Rate(50)  # Hz

    while not rospy.is_shutdown():
        try:
            print("Attempting read")
            raw, addr = sock.recvfrom(1024)
            print("unpacking")
            timestamp, *values = struct.unpack("<Q8l", raw)
            print(timestamp, values)
            # timestamp = 0
            # values = [0]*8  # Placeholder for actual data parsing

            msg = CapData()
            msg.timestamp = timestamp
            msg.values = values

            # rospy.loginfo(msg)
            pub.publish(msg)

        except socket.timeout:
            # No packet received
            pass
        except Exception as e:
            rospy.logwarn(f"Error receiving UDP packet: {e}")

        rate.sleep()

def setup_stream() -> socket.socket:
    """Sets up UDP socket to receive data from CapCup device.
     Returns:
        Configured UDP socket.
    """
    UDP_IP = "192.168.10.50" # Capcup IP
    UDP_PORT = 12345 # Port being streamed to

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", UDP_PORT))
    sock.settimeout(0.1)  # Non-blocking with timeout

    return sock

if __name__ == "__main__":
    sock = setup_stream()
    try:
        pipe_data(sock)
    except rospy.ROSInterruptException:
        pass
