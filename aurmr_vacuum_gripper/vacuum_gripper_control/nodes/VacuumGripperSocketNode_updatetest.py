#!/usr/bin/env python3
# Driver for Schmalz vacuum ejectors. Manual can be found at chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www.schmalz.com/site/binaries/content/assets/media/01_vacuum-technology-automation/SCTSi/en/BAL_10.02.99.10501_en-EN_02.pdf

import os
from pycomm3 import CIPDriver, Services, configure_default_logger, LOG_VERBOSE, exceptions, SINT, UINT, USINT, BYTE #BYTE is used to return 1 byte as an array of boolean, True = 1, False = 0, length 8 for 8 bits
#from pycomm3 import StringDataType, LOGIX_STRING, STRING
from logging import WARNING
import socket
from vacuum_gripper_control.msg import vacuum_gripper_set, vacuum_gripper_get
import rospy

LOG_FILE_PATH = '/home/aurmr/workspaces/ToF/logs/vacuum_gripper_control/pycomm3.log'

# '/home/aurmr/workspaces/' + os + '/logs/vacuum_gripper_control/pycomm3.log'

# Log levels are LOG_VERBOSE or INFO
LOG = False
LOG_LEVEL = WARNING #INFO
VALVE_ONE = None

DEVICE_STATUS = 10 # UINT8 Length 1 byte ro
EJECTOR_STATUS = 11 # UINT8 Length 16 bytes ro
SUPPLY_PRESSURE = 12 # UINT8 Length 1 rw
EJECTOR_CONTROL = 13 # UINT8 Length 16 rw
SUPPLY_VOLTAGE = 66 # float32
SETPOINT_H1 = 100 # UINT16 Length 16 x 2 rw
HYSTERESIS_h1 = 101 # UINT16 Lenth 16 x 2 rw
SETPOINT_H2 = 102 # UINT16 Lenth 16 x 2 rw
HYSTERESIS_h2 = 103 # UINT16 Lenth 16 x 2 rw
LEAKAGE_RATE = 160 # UINT16 Length 16 x 2 ro
FREE_FLOW_VACUUM = 161 # UINT16 Length 16 x 2 ro
MAX_VACUUM_REACHED = 164 # UINT16 Length 16 x 2 ro
SYSTEM_VACUUM = 515 # UINT16 Length 16 x 2 ro
EJECTOR_ONE_EXTENDED = 11000 # UINT16, referencing the first vacuum
EJECTOR_TWO_EXTENDED = 11001 # UINT16, referencing the first vacuum
EJECTOR_THREE_EXTENDED = 11002 # UINT16, referencing the third vacuum
EJECTOR_FOUR_EXTENDED = 11003 # UINT16, referencing the four vacuum
CU_ERRORS = 130 # UINT

MAX_RANGE_SETPOINT = 999
MIN_RANGE_SETPOINT = 0
TOTAL_POSSIBLE_UNITS = 16

class VacuumGripper:
    def __init__(self, ip = '192.168.137.2', gripper_type = "vacuum"):
        self.ip = ip
        self.gripper_type = gripper_type
        self.cip_driver = CIPDriver(self.ip)
        self.active_control_units = set({}) ### doesn't need to be sets, same commands work with lists.
        temp_message = self.decodingByteMessageToInt(self.get_message(SYSTEM_VACUUM, None))
        for i in range(0, len(temp_message), 2):
            if temp_message[i] > 0:
                self.active_control_units.add(i//2+1) ###CHECK
        self.debugger()
        if LOG:
            if os.path.exists(LOG_FILE_PATH):
                configure_default_logger(level=LOG_LEVEL, filename=LOG_FILE_PATH)
            else:
                f = open(LOG_FILE_PATH, "x")
                f.write("\n")
                f.close()
                print("Created new log file ", LOG_FILE_PATH)
                configure_default_logger(level=LOG_LEVEL, filename=LOG_FILE_PATH)

    #changes the connected cip_driver
    def set_ip(self, ip):
        self.ip = ip
        self.cip_driver = CIPDriver(self.ip)

    # asserts a connection, returns true if open, throws attributeError if false
    def open_connection(self):
        print("Opening connection to ", self.ip)
        assert self.cip_driver.open()
        return True

    # closes connection with CIP driver
    def close_connection(self):
        self.cip_driver.close()

    # method within CIP driver to check connection
    def is_connected(self):
        return self.cip_driver.connected

    # rospy logs warnings about the errors within the vacuum gripper including the following
    # device status, active errors, condition monitoring, and ejector status
    def debugger(self):
        if self.gripper_type == "vacuum":
            msg_get_device_status = self.get_device_status()
            msg_get_ejector_status = self.get_ejector_status()
            msg_control_unit_errors = self.control_unit_errors()
            if msg_get_device_status.strip():
                rospy.logwarn(msg_get_device_status)
            if msg_get_ejector_status.strip():
                rospy.logwarn(msg_get_ejector_status)
            if msg_control_unit_errors.strip():
                rospy.logwarn(msg_control_unit_errors)
            rospy.loginfo(self.get_activeEjectorExtendedValues())
        else:
            raise RuntimeError(f"Unknown gripper type requested: {self.gripper_type}")

    # sets the custom rospy message for file vacuum_gripper_get.msg and changes its values
    # by getting the values from the vacuum unit
    def getStatus(self):
        message = vacuum_gripper_get()
        message.SETPOINT_H1 = self.get_SetPointH1()
        message.HYSTERESIS_h1 = self.get_Hysteresis_h1()
        message.SETPOINT_H2 = self.get_SetPointH2()
        message.HYSTERESIS_h2 = self.get_Hysteresis_h2()

        message.SYSTEM_VACUUM = self.get_system_vacuum()
        message.LEAKAGE_RATE = self.get_leakage_rate()
        message.FREE_FLOW_VACUUM = self.get_free_flow_vacuum()
        message.MAX_VACUUM_REACHED = self.get_max_vacuum_range()

        message.EJECTOR_CONTROL = self.get_ejector_control()

        message.SUPPLY_VOLTAGE = self.get_supply_voltage()
        message.SUPPLY_PRESSURE = self.get_supply_pressure()
        return message

    # This method decodes a hex byte message and returns a python List of the byte message
    # in binary as strings. Primarily used for a true-false status measurement.
    # This method is needed if our data length exceeds the length of a BYTE
    # or else we could request data_type = BYTE in our message
    def decodingMessageBinary(self, message):
        stringOfByte = bin(int.from_bytes(message, byteorder='big'))[2:]
        binaryStep = []
        binary = ''
        counter = 0
        for i in range(len(stringOfByte), 0, -1):
            counter += 1
            binary = stringOfByte[i-1] + binary
            if counter % 8 == 0:
                counter = 0
                binaryStep.append(binary)
                binary = ''
        if len(binary) < 8:
            binaryStep.append('0'*(8-len(binary)) + binary)
        while (len(binaryStep) < len(message)):
                binaryStep.append('00000000')
        binaryStep.reverse()
        return binaryStep

    # this method decodes bytes and returns a python List of the byte messages as decimals
    def decodingByteMessageToInt(self, message):
        return [message[byte] for byte in range(len(message))] #apparently calling a byte in a list converts it automatically

    def decodingBytesMessageToInt(self, message): #started thsis but might not need....
        result = []
        for i in range(len(message)):
            result.append(int.from_bytes(message[i:i+1], byteorder='little'))
        return result

    # converts a single integer value to a binary 8-bit value as a string
    def decodingIntToBinary(self, decimalInt):
        if decimalInt == 0:
            return "00000000"
        binary_result = ""
        while decimalInt > 0:
            binary_result = str(decimalInt % 2) + binary_result
            decimalInt //= 2
        while len(binary_result) < 8:
            binary_result = "0" + binary_result
        return binary_result

    # reading the CIP driver offset instance and returns the data in the representation of the dataType
    # if None then the output is a byte representation
    def get_message(self, offsetReadOnlyInstance, dataType):
        try:
            message = self.cip_driver.generic_message(
                service = Services.get_attribute_single,
                class_code= 0xA2,
                instance = offsetReadOnlyInstance,
                attribute = 5,
                data_type = dataType
            )
        except exceptions.CommError:
            if self.restablishConnection():
                message = self.cip_driver.generic_message(
                    service = Services.set_attribute_single,
                    class_code= 0xA2,
                    instance = offsetReadOnlyInstance,
                    attribute = 5,
                    data_type = dataType
                )
        except AttributeError:
            if self.restablishConnection():
                message = self.cip_driver.generic_message(
                    service = Services.set_attribute_single,
                    class_code= 0xA2,
                    instance = offsetReadOnlyInstance,
                    attribute = 5,
                    data_type = dataType
                )
        return message.value

    # writing the CIP driver offset instance with a specific request data to enter
    # should return if sucessfully sent or not
    def set_message(self, offsetReadOnlyInstance, requestData):
        try:
            message = self.cip_driver.generic_message(
                    service = Services.set_attribute_single,
                    class_code= 0xA2,
                    instance = offsetReadOnlyInstance,
                    attribute = 5,
                    request_data = requestData
                )
        except exceptions.CommError:
            if self.restablishConnection():
                message = self.cip_driver.generic_message(
                    service = Services.set_attribute_single,
                    class_code= 0xA2,
                    instance = offsetReadOnlyInstance,
                    attribute = 5,
                    request_data = requestData
                )
        except AttributeError:
            if self.restablishConnection():
                message = self.cip_driver.generic_message(
                    service = Services.set_attribute_single,
                    class_code= 0xA2,
                    instance = offsetReadOnlyInstance,
                    attribute = 5,
                    request_data = requestData
                )
        return message.error == None

    # This method restablishes the connection between the CIP driver of the vacuum and this computer.
    def restablishConnection(self):
        tries = 0
        while not self.cip_driver.connected and tries < 3:
            try:
                self.cip_driver.close()
                self.cip_driver.open()
            except exceptions.CommError:
                print("Connection broken, Restablishing connection...")
            except AttributeError:
                print("Connection broken, Restablishing connection...")
            tries +=1
        if(tries >= 3 and not self.cip_driver.connected):
            print("Failed to reconnect.")
        else:
            print("Connected successfully")
        return self.cip_driver.connected

    # gets the device status of the entire ejector
    # interpreted from the device's manual
    def get_device_status(self):
        byteArray = self.get_message(DEVICE_STATUS, BYTE)
        if not byteArray[1] and not byteArray[0]:
            return "Device is working optimally \n"
        elif not byteArray[1] and byteArray[0]:
            return "Device is working, maintenance necessary \n"
        else: # 10 or 11 in the last 2 bits in the byte
            return "Device is working, but there are warnings in the control unit \n"

    # gets the ejector status of the each module in the device, returns a string of each ejector's status
    def get_ejector_status(self):
        ejector_status_diagnostics = ""
        intArray = self.decodingByteMessageToInt(self.get_message(EJECTOR_STATUS, None))
        for i in range(len(intArray)):
            if intArray[i] != 0:
                ejector_status_diagnostics += "EJECTOR STATUS FOR EJECTOR " + str(i) + ": \n"
                byte = self.decodingIntToBinary(intArray[i])
                if byte[0] == "1":
                    ejector_status_diagnostics += "Manual mode active \n"
                if byte[1] == "1":
                    ejector_status_diagnostics += "Free flow vacuum too high \n"
                if byte[2] == "1":
                    ejector_status_diagnostics += "H1 not reached in suction cycle \n"
                if byte[3] == "1":
                    ejector_status_diagnostics += "Leakage rate greater than limit \n"
                if byte[4] == "1":
                    ejector_status_diagnostics += "Evacuation time greater than limit \n"
                if byte [5] == "1":
                    ejector_status_diagnostics += "Valve protection active \n"
                if byte[6] == "1":
                    ejector_status_diagnostics += "H2 level reached (part present) in ejector \n"
                if byte[7] == "1":
                    ejector_status_diagnostics += "H1 level reached (air saving fuction) in ejector \n"
        return ejector_status_diagnostics

    # gets the device supply pressure currently and returns the pressure in bar units
    def get_supply_pressure(self):
        supply_pressure_value = self.decodingByteMessageToInt(self.get_message(SUPPLY_PRESSURE, None))
        # rospy.loginfo(f"$$$$$$$$$$$4 {supply_pressure_value}")
        return supply_pressure_value[0] * 0.1

    # sets the device supply pressure and value has to be in range 6.3 and 1.9 bar units or else nothing happens.
    # this is to ensure that the unit does not trigger and active error/ warning
    def set_supply_pressure(self, value):
        if (value > 1.9 and value < 6.3):
            value = int(value*10) #was in 0.1 bar steps
            data = (value).to_bytes(2, byteorder='little')
            self.set_message(SUPPLY_PRESSURE, data)

    # returns an int list for each unit and if that unit is blowing, suctioning, or closed
    # 0 for closed, 1 for suctioning, 2 for blowing.
    def get_ejector_control(self):
        return self.decodingByteMessageToInt(self.get_message(EJECTOR_CONTROL, None))

    # returns an int list of the ejector's maximim vacuum range in mbars
    def get_max_vacuum_range(self):
        msg = self.decodingByteMessageToInt(self.get_message(MAX_VACUUM_REACHED, None))
        vacuum_range = []
        for i in range (1, len(msg), 2):
                binary = self.decodingIntToBinary(msg[i]) + self.decodingIntToBinary(msg[i-1])
                intVacuumRange = int(binary, 2)
                vacuum_range.append(intVacuumRange)
        return vacuum_range

    # returns an int list of the unobstructed suction (for more detail refer to the manual) in mbars
    def get_free_flow_vacuum(self):
        msg = self.decodingByteMessageToInt(self.get_message(FREE_FLOW_VACUUM, None))
        free_flow = []
        for i in range (1, len(msg), 2):
            binary = self.decodingIntToBinary(msg[i]) + self.decodingIntToBinary(msg[i-1])
            intFreeFlow = int(binary, 2)
            free_flow.append(intFreeFlow)
        return free_flow

    # gets the voltage measured by the device and returns it as an float in Volts
    def get_supply_voltage(self):
        msg = self.decodingByteMessageToInt(self.get_message(SUPPLY_VOLTAGE, None))
        binary = self.decodingIntToBinary(msg[1]) + self.decodingIntToBinary(msg[0])
        data = int(binary, 2) * 0.1
        min = int((self.decodingIntToBinary(msg[3]) + self.decodingIntToBinary(msg[2])), 2) *0.1
        max = int((self.decodingIntToBinary(msg[5]) + self.decodingIntToBinary(msg[4])), 2) *0.1
        if (data < min or data > max):
            rospy.logwarn("Primary supply voltage exceeded threshold")
        return data

    # returns a string message of any control unit errors, returns empty string if no errors
    def control_unit_errors(self):
        msg = self.decodingByteMessageToInt(self.get_message(CU_ERRORS, None))
        cu_errors = ""
        activeErrorsInCU = self.decodingIntToBinary(msg[len(msg)-1]) # this could be flipped
        if (activeErrorsInCU[7] == "1"):
            cu_errors += "Internal error: data corruption \n"
        if (activeErrorsInCU[6] == "1"):
            cu_errors += "Internal error: bus fault \n"
        if (activeErrorsInCU[5] == "1"):
            cu_errors += "Primary voltage too low \n"
        if (activeErrorsInCU[4] == "1"):
            cu_errors += "Primary voltage too high \n"
        if (activeErrorsInCU[3] == "1"):
            cu_errors += "Secondary voltage too low \n"
        if (activeErrorsInCU[2] == "1"):
            cu_errors += "Secondary voltage too high \n"
        if (activeErrorsInCU[1] == "1"):
            cu_errors += "Supply pressure too low (<1.9 bar) or too high (> 6.3 bar)"
        if (activeErrorsInCU[0] == "1"):
            cu_errors += "Error in one or more ejectors"
        for i in range(0, (len(msg)-1)): # this range could be off as in the 0 index could be the processed data
            if(msg[i] == "1"): # 0 = measurement range overrun
                cu_errors += "Ejector unit " +str(i+1)+ "failed vacuum calibration. \n"
        return cu_errors

    # returns an int list message of the leakrate of each ejector unit
    def get_leakage_rate(self):
        msg = self.decodingByteMessageToInt(self.get_message(LEAKAGE_RATE, None))
        leakRate = []
        for i in range (1, len(msg), 2):
            binary = self.decodingIntToBinary(msg[i]) + self.decodingIntToBinary(msg[i-1])
            intLeakRate = int(binary, 2)
            leakRate.append(intLeakRate)
                # leakRate += "Ejector unit " + str((i//2)+1) + " has a leakage rate"
                # leakRate += " of " + str(intLeakRate) + " mbar/s. \n"
        # if not leakRate:
        #     leakRate += "No leakage in any ejector unit. \n"
        return leakRate

    # gets the current vacuum in individual ejectors in mbars, returns an int list of the active ejectors
    # integers in the list represent the vacuum mbars, the index represents the ejector unit
    def get_system_vacuum(self):
        msg = self.decodingByteMessageToInt(self.get_message(SYSTEM_VACUUM, None))
        systemVacuum = []
        for i in range (1, len(msg), 2):
            binary = self.decodingIntToBinary(msg[i]) + self.decodingIntToBinary(msg[i-1])
            intSystemVacuum = int(binary, 2)
            systemVacuum.append(intSystemVacuum)
        return systemVacuum

    # gets the raw message byte value from a specific instance or status number and returns a python list
    # of the hex bytes converted into integer values
    def get_value_from_instance(self, instance):
        return self.decodingByteMessageToInt(self.get_message(instance, None))

    # interprets the extended values on any single ejector where any is the instance of the ejector in the
    # control unit. Returns a string of the extended values.
    def get_anyExtendedValuesEjector(self, any):
        msg = self.decodingByteMessageToInt(self.get_message(any, None))
        systemVacuum = int( (self.decodingIntToBinary(msg[0]) + self.decodingIntToBinary(msg[1])), 2)
        airConsumption = int( (self.decodingIntToBinary(msg[2]) + self.decodingIntToBinary(msg[3])), 2)
        leakageLastCycle = int( (self.decodingIntToBinary(msg[4]) + self.decodingIntToBinary(msg[5])), 2)
        evacuationTimeT1 = int( (self.decodingIntToBinary(msg[6]) + self.decodingIntToBinary(msg[7])), 2)
        lastFreeFlowVacuum = int( (self.decodingIntToBinary(msg[8]) + self.decodingIntToBinary(msg[9])), 2)
        ejectorData = ""
        ejectorData += "System vacuum (in mbar) " + str(systemVacuum) + "\n"
        ejectorData += "Air consumption (in l/min) " + str(airConsumption) + "\n"
        ejectorData += "Leakage of last cycle (in mbar/s) " + str(leakageLastCycle) + "\n"
        ejectorData += "Evacuation time T1 (in ms) " + str(evacuationTimeT1) + "\n"
        ejectorData += "Last free flow vacuum (in mbar) " + str(lastFreeFlowVacuum) + "\n"
        return ejectorData

    # retrieves all extended values in the ejectors that are active (i.e, in the constants EJECTORS_EXTENDED)
    def get_activeEjectorExtendedValues(self):
        extendedValMsg = ""
        for ejector in self.active_control_units:
            extendedValMsg += "Ejector unit" + str(ejector) + "\n"
            extendedValMsg += self.get_anyExtendedValuesEjector(11000-1+ejector) + "\n"
        return extendedValMsg

    # Gets the set point H1 of each ejector (H1 controls switching points)
    # and returns an int list h2 value in mbars.
    def get_SetPointH1(self):
        msg = self.decodingBytesMessageToInt(self.get_message(SETPOINT_H1, None))
        getSetPointH1 = []
        for i in range (1, len(msg), 2):
            binary = self.decodingIntToBinary(msg[i]) + self.decodingIntToBinary(msg[i-1])
            intSetPointH1 = int(binary, 2)
            getSetPointH1.append(intSetPointH1)
        return getSetPointH1

    # Gets the hysteresis h1 of each ejector (h1 is the Hysteresis of control switching point)
    # and returns an int list h2 value in mbars.
    def get_Hysteresis_h1(self):
        msg = self.decodingBytesMessageToInt(self.get_message(HYSTERESIS_h1, None))
        getHysteresisH1 = []
        for i in range (1, len(msg), 2):
            binary = self.decodingIntToBinary(msg[i]) + self.decodingIntToBinary(msg[i-1])
            intHysteresisH1 = int(binary, 2)
            getHysteresisH1.append(intHysteresisH1)
        return getHysteresisH1

    # Gets the set point H2 of each ejector (H2 controls switching points for component check)
    # and returns an int list h2 value in mbars.
    def get_SetPointH2(self):
        msg = self.decodingBytesMessageToInt(self.get_message(SETPOINT_H2, None))
        getSetPointH2 = []
        for i in range (1, len(msg), 2):
            binary = self.decodingIntToBinary(msg[i]) + self.decodingIntToBinary(msg[i-1])
            intSetPointH2 = int(binary, 2)
            getSetPointH2.append(intSetPointH2)
        return getSetPointH2

    # Gets the hysteresis h2 of each ejector (h2 is the Hysteresis of switching point for component check)
    # and returns an int list h2 value in mbars.
    def get_Hysteresis_h2(self):
        msg = self.decodingBytesMessageToInt(self.get_message(HYSTERESIS_h2, None))
        getHysteresisH2 = []
        for i in range (1, len(msg), 2):
            binary = self.decodingIntToBinary(msg[i]) + self.decodingIntToBinary(msg[i-1])
            intHysteresisH2 = int(binary, 2)
            getHysteresisH2.append(intHysteresisH2)
        return getHysteresisH2

    # Sets the set point H1 value of each active ejector with value as an integer measured in mbars.
    # value is an int list for the h1 values for the amount of ejectors
    def set_SetPointH1(self, value):
        if (value < MAX_RANGE_SETPOINT and value > MIN_RANGE_SETPOINT):
            data = self.helper_create_H_byte_message(self, value)
            self.set_message(SETPOINT_H1, data)

    # Sets the set Hysteresis h1 value of each active ejector with value as an integer measured in mbars.
    # value is an int list for the h1 values for the amount of ejectors
    def set_Hysteresis_h1(self, value):
        if (value < MAX_RANGE_SETPOINT and value > MIN_RANGE_SETPOINT):
            data = self.helper_create_H_byte_message(self, value)
            self.set_message(HYSTERESIS_h1, data)

    # Sets the set point H2 value of each active ejector with value as an integer measured in mbars.
    # value is an int list for the h1 values for the amount of ejectors
    def set_SetPointH2(self, value):
        if (value < MAX_RANGE_SETPOINT and value > MIN_RANGE_SETPOINT):
            data = self.helper_create_H_byte_message(self, value)
            self.set_message(SETPOINT_H2, data)

    # Sets the set Hysteresis h2 value of each active ejector with value as an integer measured in mbars.
    # value is an int list for the h1 values for the amount of ejectors
    def set_Hysteresis_h2(self, value):
        if (value < MAX_RANGE_SETPOINT and value > MIN_RANGE_SETPOINT):
            data = self.helper_create_H_byte_message(self, value)
            self.set_message(HYSTERESIS_h2, data)

    def helper_create_H_byte_message(self, listValue):
        data = bytes()
        for i in range(0, len(listValue)):
            if i < len(self.active_control_units):
                if i in self.active_control_units:
                    message = (listValue[i]).to_bytes(2, byteorder='little')
                    data += message
                else:
                    data += bytes(2)
        return data

    def helper_create_ejector_bytes_message(self, listValue):
        data = bytes()
        for integer in listValue:
            data += (integer).to_bytes(1, byteorder='little')
        return data

    # sends the command given a command with 2 integer list, one containing code for functionality,
    # 0 (close), 1 (open), or 2 (blow off) and the other list represents the ejector unit number.
    # This method will only send the command to the cip_driver if the ejector unit is active within the unit,
    # also if the command is either 0, 1, or 2, and that the 2 list matches in size.
    def sendCommand(self, command):
        ejector_unit = command.EJECTOR_CONTROL_UNIT
        ejector_function = command.EJECTOR_CONTROL_VALUE
        temp = self.get_ejector_control()
        for unit in ejector_unit:
            temp[unit] = ejector_function[unit]
        rospy.loginfo(f"################ {temp}")
        data = self.helper_create_ejector_bytes_message(temp)
        self.set_message(EJECTOR_CONTROL, data)


    # Checks to see if our setpoint H1 H2 and hysteresis values are valid for the vacuum unit
    # return true if all values are with ranges of each other, and false otherwise
    # our setpoint H1 H2 and hysteresis values h1 and h2 are integers.
    def is_valid_vacuum_controls(self, H1, H2, hysteresis1, hysteresis2):
        if not (MAX_RANGE_SETPOINT > H1 and H1 >= (H2-hysteresis2)):
            print("Setpoint H1 is out of vacuum unit range")
            return False
        elif not ((H1-H2) >= hysteresis1 and hysteresis1 > 10):
            print("Hysteresis 1 is out of vacuum unit range")
            return False
        elif not ((H1-hysteresis1) >= H2 and H2 >=(hysteresis2+2)): # +2 or -2 from manual
            print("Setpoint H1 is out of vacuum unit range")
            return False
        elif not ((H2-2) >= hysteresis2 >= 10):
            print("Hysteresis 2 is out of vacuum unit range")
            return False
        else:
            return True

    # pressure is an int list of the H1 values we want to set each active vacuum unit to be.
    # this method is a control strategy for setting H1, H2, h1, h2, set point and hysteresis levels.
    def set_default_vacuum_controls(self): #these are from the manual linked in schmalz.com
        H1 = [750] * len(self.active_control_units)
        H2 = [550] * len(self.active_control_units)
        hysteresis1 = [150]*len(self.active_control_units)
        hysteresis2 = [10]*len(self.active_control_units)
        if self.is_valid_vacuum_controls(H1, H2, hysteresis1, hysteresis2): # kind default should be valid but just in case something hard coded changes
            self.set_SetPointH1(SETPOINT_H1, H1)
            self.set_SetPointH2(SETPOINT_H2, H2)
            self.set_Hysteresis_h1(HYSTERESIS_h1, hysteresis1)
            self.set_Hysteresis_h2(HYSTERESIS_h2, hysteresis2)

def mainLoop():
    # Gripper is a C-Model that is connected to a UR controller with the Robotiq URCap installed.
    # Commands are published to port 63352 as ASCII strings.
    gripper = VacuumGripper('192.168.137.2', "vacuum")
    gripper.open_connection()
    pub = rospy.Publisher('~status', vacuum_gripper_get, queue_size=10)
    # The Gripper command
    rospy.Subscriber('~command', vacuum_gripper_set, gripper.sendCommand)
    while not rospy.is_shutdown():
    # Get and publish the Gripper status
        status = gripper.getStatus()
        pub.publish(status)
        rospy.sleep(0.1)

if __name__ == '__main__':
    rospy.init_node('vacuum_gripper_socket_node')
    try:
        mainLoop()
    except rospy.ROSInterruptException: pass

"""
####### ASK QUEUE SIZE
#### ASK DO WE NEED THIS TO CONSTANTLY BE PUBLISHING
######## ALSO WARNINGS
## change to default settings???? before we do, record the current numbers
current, H1 = 850, hysteresis_h1 = 150, H2 = 550, hysteresis_h2 = 10 or default
FREE_FLOW_VACUUM
MAX_VACUUM_REACHED

from vacuum_gripper_control.msg import vacuum_gripper_set

command = vacuum_gripper_set()
command.SETPOINT_H1 = [850] * 16
command.HYSTERESIS_h1 = [150] * 16
command.SETPOINT_H2 = [550] * 16
command.HYSTERESIS_h2 = [10] * 16
command.SUPPLY_PRESSURE = 0.0
command.EJECTOR_CONTROL_UNIT = []  # which entries
command.EJECTOR_CONTROL_VALUE = [] # 0, 1, or 2

rospy.Publisher('gripper_to_vacuum', vacuum_gripper_set, queue_size = 10)
pub.publish(command)

"""