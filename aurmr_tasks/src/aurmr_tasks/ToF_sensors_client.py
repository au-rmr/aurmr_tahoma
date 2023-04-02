#!/usr/bin/env python

import json
import requests 
import rospy
from std_msgs.msg import String
import time
def sensor_publisher():
    pub = rospy.Publisher('tof_sensors', String, queue_size=10) ##The name talker will appear as topic
    rospy.init_node('publisher_node', anonymous = True)
    rate = rospy.Rate(1)
    rospy.loginfo("Publisher node started and will publish sensor data")

    while not rospy.is_shutdown():

        init_time = time.time()
        url = 'http://10.42.0.129/'
        r = requests.get(url, stream = True)  
        print('==================>', time.time() - init_time)
        ##json.loads gets the json as a python dict
        data = json.loads(r.text)

         #typecast individual data points to str
        dataSensorOne = str(data["sensor1"])
        dataSensorTwo = str(data["sensor2"])

        ##Publish: typecast the python dict to a str
        pub.publish(str(data))
        
        ##Publish: typecasted sensor data for sensor one and then two
        pub.publish("1: " + dataSensorOne)
        pub.publish("2: " + dataSensorTwo)
        rate.sleep()


if __name__ == '__main__':
    try:
        sensor_publisher()
    except rospy.ROSInterruptException:
        pass