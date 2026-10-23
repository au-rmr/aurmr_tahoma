import serial
import numpy as np
import re

import rospy
from std_msgs.msg import Int16MultiArray

import subprocess



if __name__ == '__main__':

    # try:
    #     subprocess.Popen('chmod a+rw /dev/ttyUSB0', shell=True)

    # except:
    #     print("Permission denied to /dev/ttyUSB0?")

    pub1 = rospy.Publisher('/tofSensor1', Int16MultiArray, queue_size = 1)
    pub2 = rospy.Publisher('/tofSensor2', Int16MultiArray, queue_size = 1)
    rospy.init_node('tof')
    #print('created node')
    ser = serial.Serial('/dev/ttyACM0')
    ser.baudrate = 115200
    #print('opened serial')
    sensor1 = False

    # sensor1Readings = []
    # sensor2Readings = []

    # read junk bytes
    line = ser.readline()

    while True:
        try:
            line = ser.readline()

            #print('after second readline')

            # get readings
            if line.strip() != b'':
                line_str = line.decode().strip()

                # determine if sensor 1 readings
                if line_str[0] == '1':
                    sensor1 = True
                elif line_str[0] == '2':
                    sensor1 = False
                else:
                    import pdb
                    pdb.set_trace()

                # convert string data to numpy array
                numbers = re.findall(r'\d+', line.decode().strip())
                numbers = np.array(numbers).astype(np.int)
                numbers = numbers[1::2]

                msg = Int16MultiArray()
                msg.data = numbers

                #print('before sending data')

                if sensor1:
                    assert line_str[0] == '1', "Sensor 1 parsed wrong"
                    print('sensor1: ' +  str(numbers))
                    pub1.publish(msg)
                else:
                    assert line_str[0] == '2', "Sensor 2 parsed wrong"
                    print('sensor2: ' +  str(numbers))
                    pub2.publish(msg)

                #print('after sending data')
        except KeyboardInterrupt:
            ser.close()
            break

    print('stopped publishing sensor data')