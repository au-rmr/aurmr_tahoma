#!/usr/bin/env python3


# Driver for Schmalz vacuum ejectors. Manual can be found at chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www.schmalz.com/site/binaries/content/assets/media/01_vacuum-technology-automation/SCTSi/en/BAL_10.02.99.10501_en-EN_02.pdf
import os
from pycomm3 import CIPDriver, Services, configure_default_logger, LOG_VERBOSE, exceptions, SINT, UINT, USINT
from logging import INFO
import socket
from robotiq_2f_gripper_control.msg import vacuum_gripper_input, vacuum_gripper_output
import rospy


LOG_FILE_PATH = 'c:/tmp/pycomm3.log'
# Log levels are LOG_VERBOSE or INFO
LOG_LEVEL = INFO
VALVE_ONE = None


DEVICE_STATUS = 10 # UINT8 Length 1 byte ro
EJECTOR_STATUS = 11 # UINT8 Length 16 bytes ro
SUPPLY_PRESSURE = 12 # UINT8 Length 1 rw
EJECTOR_CONTROL = 13 # UINT8 Length 16 rw
SUPPLY_VOLTAGE = 66
SETPOINT_H1 = 100 # UINT16 Length 16 x 2 rw
HYSTERESIS_h1 = 101 # UINT16 Lenth 16 x 2 rw
SETPOINT_H2 = 102 # UINT16 Lenth 16 x 2 rw
HYSTERESIS_h2 = 103 # UINT16 Lenth 16 x 2 rw
LEAKAGE_RATE = 160 # UINT16 Length 16 x 2 ro
FREE_FLOW_VACUUM = 161 # UINT16 Length 16 x 2 ro
MAX_VACUUM_REACHED = 164 # UINT16 Length 16 x 2 ro
SYSTEM_VACUUM = 515 # UINT16 Length 16 x 2 ro



class VacuumGripper:
    def __init__(self, ip = '192.168.250.2'):
        self.ip = ip

        if os.path.exists(LOG_FILE_PATH):
            configure_default_logger(level=LOG_LEVEL, filename=LOG_FILE_PATH)
        else:
            f = open(LOG_FILE_PATH, "w")
            f.write("\n")
            f.close()
            print("Created new log file ", LOG_FILE_PATH)
            configure_default_logger(level=LOG_LEVEL, filename=LOG_FILE_PATH)

    def set_ip(self, ip):
        self.ip = ip

    def open_connection(self):
        print("Opening connection to ", self.ip)
        self.cip_driver = CIPDriver(self.ip)
        print(self.cip_driver)
        assert self.cip_driver.open()
        # assert self.cip_driver.connected
        print("Opened connection to ", self.ip)
        return True

    def close_connection(self):
        self.cip_driver.close()
    
    def is_connected(self):
        return self.cip_driver.connected

    def getStatus(self):
        if self.gripper_type == RobotiqCModelURCap.GripperType.VACUUM:
            message = vacuum_gripper_input()
            #Assign the values to their respective variables
            message.DEVICE_STATUS = self.get_device_status(self)
            message.EJECTOR_STATUS = self.get_ejector_status(self)
            message.SUPPLY_VOLTAGE = self.get_supply_voltage(self)
            message.LEAKAGE_RATE = self.get_leakage_rate(self) # what is the leakage rate supposed to give
            message.FREE_FLOW_VACUUM = self.get_free_flow_vacuum(self) #what is the free flow supposed to give
            message.MAX_VACUUM_REACHED = self.get_max_vacuum_range(self) # what is this max vacuum supposed to give
            message.SYSTEM_VACUUM = self.get_system_vacuum(self)
        else:
            raise RuntimeError(f"Unknown gripper type requested: {self.gripper_type}")
        return message
    
    def vacuum(self, pressure, release):
        var_dict = dict([(self.GTO, 0)]) # GTO is gripper regulation -- similar to ejector control
        self._set_vars(var_dict) 
        while(self._get_var(self.GTO) != 0): # sleep until the gripper regulation is 0, meaning the gripper is off
            rospy.sleep(.005)
        # var_dict = dict([(self.PR, pressure), (self.GTO, 1)])
        print(var_dict) 
        var_dict = dict([(self.MOD, 0b01), (self.PR, pressure), (self.SP, 0), (self.FR, pressure+40), (self.GTO, 1)])
        # MOD - gripper mode  
        #   X
        # PR - max pressure requested when writing / pressure reading when read
        #   similar to SETPOINT_H1 <-- why is this set to 100 above? 
        # SP - timeout period
        #  
        # FR - minimum pressure requested when writing

        # GTO - gripper
        return self._set_vars(var_dict)

    def sendCommand(self):
        #if self.gripper_type == RobotiqCModelURCap.GripperType.VACUUM:
            #self.vacuum(command.rPR, command.rATR)
            #rPR is the register for pressure on the vacuum
            #rATR is the register for auto release on the vacuum
            #don't need gripper mode, max-pressure
        # EJECTOR_CONTROL is an instance so it should not be set to 0, 
        # is there a way to to make this gripper be regulated to 0 - meaning off
        var_dict = {EJECTOR_CONTROL, 0} # this is the instance - I need a varible that does the same thing
        while(EJECTOR_CONTROL != 0) {
            rospy.sleep(0.005);
        }
        print(var_dict)
        var_dict = {
            #self.MOD: 0b01
            #set the max pressure
            #set the timeout
            #in our case this would be the H1 and h1 values?
            #regulate gripper on
        }
        

# request = cip_driver.generic_message(service=Services.set_attribute_single, class_code=0xA2, instance=13, attribute=5, request_data=bytearray([0b00000011] * 16))
# request = cip_driver.generic_message(service=Services.set_attribute_single, class_code=0xA2, instance=13, attribute=5, request_data=bytearray([0x00] * 16))

    def close_valves(self):
        pass

    def close_valve(self, number):
        try:
            self.cip_driver.generic_message(service=Services.set_attribute_single, class_code=0xA2, instance=EJECTOR_CONTROL, attribute=5, request_data=bytearray([0b00000001] + [0b00000001]*15))
        except exceptions.CommError:
            if not self.open_connection():
                raise exceptions.CommError
        except AttributeError:
            print('No such attribute found. You may need to open a connection first.') 

    def open_valve(self, number):
        ejectors = [0] * 16
        ejectors[number - 1] = 1
        SINT[None].encode(ejectors)
        try:
            self.cip_driver.generic_message(service=Services.set_attribute_single, class_code=0xA2, instance=EJECTOR_CONTROL, attribute=5, request_data=bytearray([0b00000000] + [0b00000001]*15))
        except exceptions.CommError:
            if not self.open_connection():
                raise exceptions.CommError
        except AttributeError:
            print('No such attribute found. You may need to open a connection first.') 

    def get_device_status(self):
        msg = self.cip_driver.generic_message(service=Services.get_attribute_single, class_code=0xA2, instance=DEVICE_STATUS, attribute=5)
        return USINT.decode(msg)
    
    #TODO Is a list comprehension correct?
    def get_ejector_status(self):
        msg = self.cip_driver.generic_message(service=Services.get_attribute_single, class_code=0xA2, instance=EJECTOR_STATUS, attribute=5)
        status = [USINT.decode(msg) for ejector in msg]
        return status 

    def get_supply_pressure(self):
        msg = self.cip_driver.generic_message(service=Services.get_attribute_single, class_code=0xA2, instance=SUPPLY_PRESSURE, attribute=5)
        return USINT.decode(msg)
    
    def get_leakage_rate(self):
        pass
    def get_system_vacuum(self):
        pass
    def get_max_vacuum_range():
        pass
    def get_free_flow_vacuum():
        pass
    def get_supply_voltage(self):
        msg = self.cip_driver.generic_message(service=Services.get_attribute_single, class_code=0xA2, instance=SUPPLY_VOLTAGE, attribute=1)
        return UINT.decode(msg[0:3])/10


def mainLoop(ur_address, gripper_type):
  # Gripper is a C-Model that is connected to a UR controller with the Robotiq URCap installed. 
  # Commands are published to port 63352 as ASCII strings.
  gripper = RobotiqCModelURCap(ur_address, gripper_type)

  gripper.activate(True)
  # The Gripper status
  if gripper_type == 'vacuum':
    input_msg = vacuum_gripper_input
    output_msg = vacuum_gripper_output
  else:
    raise RuntimeError(f"Unknown gripper type requested: {gripper_type}")

  pub = rospy.Publisher('~status', input_msg, queue_size=3)
  # The Gripper command
  rospy.Subscriber('~command', output_msg, gripper.sendCommand)
  
  while not rospy.is_shutdown():
    # Get and publish the Gripper status
    status = gripper.getStatus()
    pub.publish(status)
    # Wait a little
    rospy.sleep(0.1)

if __name__ == '__main__':
  rospy.init_node('robotiq_2f_gripper_socket_node')
  ip = rospy.get_param("~robot_ip", "192.168.43.92")
  gripper_type = rospy.get_param("~gripper_type", "finger")
  try:
    mainLoop(ip, gripper_type)
  except rospy.ROSInterruptException: pass


