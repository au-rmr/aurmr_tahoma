from pycomm3 import CIPDriver, Services, configure_default_logger, LOG_VERBOSE, exceptions, SINT, UINT, USINT
import numpy as np
import time

#try small, try 1

CLOSE_VALVE = 0
SUCTION = 1
BLOW_OFF = 2
MAX_RANGE_SETPOINT = 999
MIN_RANGE_SETPOINT = 0

FREQUENCY = 0.05
AMPLITUDE = 150
MIDPOINT = 700

default_H1 = 850
default_hystersis1 = 150 # bounds of pressure from H2
default_H2 = 550
default_hystersis2 = 10 # bounds for deactivation

def get_status(instance):
    """
    Takes in an int instance and returns the value that
    instance is associated with the vacuum ejector unit
    as a byte array.
    """
    message = cip_driver.generic_message(
        service = Services.get_attribute_single,
        class_code= 0xA2,
        instance = instance,
        attribute = 5,
        data_type=None
    )
    return message.value

def send_message(instance, byte_message):
    """
    Takes in an int instance and a byte array and sends
    it to the vacuum ejector. Returns None.
    """
    cip_driver.generic_message(
        service = Services.set_attribute_single,
        class_code= 0xA2,
        instance = instance,
        attribute = 5,
        request_data=byte_message
    )

def is_valid_vacuum_controls(H1, H2, hysteresis1, hysteresis2):
        """
        """
        if not (MAX_RANGE_SETPOINT > H1 and H1 >= (H2+hysteresis1)):
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

def toInt_16x02(byteArray):
    return int.from_bytes(byteArray[0:2], "little") 
### change at some point since this is only for ejector unit 1

def osc_func(time):
    return int(AMPLITUDE * np.sin(2*np.pi*FREQUENCY*time) + MIDPOINT)

def reset():
    send_message(100, default_H1.to_bytes(2, 'little')*16)
    send_message(101, default_hystersis1.to_bytes(2, 'little')*16)
    send_message(102, default_H2.to_bytes(2, 'little')*16)
    send_message(103, default_hystersis2.to_bytes(2, 'little')*16)

if __name__ == '__main__':
    ip = '192.168.137.2'
    cip_driver = CIPDriver(ip)
    cip_driver.open()
    # current controls
    # H1 = toInt_16x02(get_status(100))
    # hysteresis1 = toInt_16x02(get_status(101))
    # H2 = toInt_16x02(get_status(102))
    # hysteresis2 = toInt_16x02(get_status(103))
    # print(H1, H2, hysteresis1, hysteresis2)
    # reset()

    send_message(13, SUCTION.to_bytes(2, 'little') + CLOSE_VALVE.to_bytes(2, 'little')*7)
    start = time.time()
    while time.time() - start < 1:
        print(get_status(109))
    send_message(13, CLOSE_VALVE.to_bytes(2, 'little')*8)

    # new_H1 = 950
    # new_hysteresis1 = 49
    # new_H2 = 500
    # if is_valid_vacuum_controls(new_H1, new_H2, new_hysteresis1, hysteresis2):
    #     byteH1 = new_H1.to_bytes(2, 'little')
    #     byteHYS = new_hysteresis1.to_bytes(2, 'little')
    #     byteH2 = new_H2.to_bytes(2, 'little')

    #     send_message(100, byteH1+15*default_H1.to_bytes(2, 'little'))
    #     send_message(101, byteHYS+15*default_hystersis1.to_bytes(2, 'little'))
    #     send_message(102, byteH2+15*default_H2.to_bytes(2, 'little'))


    #     print("NICE, " + str(toInt_16x02(get_status(100))) + " " + str(toInt_16x02(get_status(101))) )
    #     send_message(13, SUCTION.to_bytes(2, 'little') + CLOSE_VALVE.to_bytes(2, 'little')*7)
    #     time.sleep(5)
    # send_message(13, CLOSE_VALVE.to_bytes(2, 'little')*8)
    # print(toInt_16x02(get_status(100)))
    # new_H1 = 550
    # new_hysteresis1 = 49
    # if is_valid_vacuum_controls(new_H1, new_H2, new_hysteresis1, hysteresis2):
    #     byteH1 = new_H1.to_bytes(2, 'little')
    #     send_message(100, byteH1+15*default_H1.to_bytes(2, 'little'))
    #     print("NICE, " + str(toInt_16x02(get_status(100))) + " " + str(toInt_16x02(get_status(101))) )
    #     send_message(13, SUCTION.to_bytes(2, 'little') + CLOSE_VALVE.to_bytes(2, 'little')*7)
    #     time.sleep(5)
    # ## close reopen
    # send_message(13, CLOSE_VALVE.to_bytes(2, 'little')*8)
    # reset()
    """
    send_message(13, SUCTION.to_bytes(2, 'little') + CLOSE_VALVE.to_bytes(2, 'little')*7)
    # #suction and see values and if it fluctuates
    for i in range(200):
        new_H1 = osc_func(i/10)
        byte = new_H1.to_bytes(2, 'little')
        if is_valid_vacuum_controls(new_H1, H2, hysteresis1, hysteresis2):
            send_message(100, byte+15*default_H1.to_bytes(2, 'little'))
        print(toInt_16x02(get_status(100)))
    send_message(13, CLOSE_VALVE.to_bytes(2, 'little')*8)
    reset()

    #check if valid for ranges of cirle
    #plug and chug"
    """




#initial offset
# a=550
# print(a.to_bytes(2, 'little')*16 == get_status(102))

# send function
#   if is_valid_vacuum_controls(H1, H2, hysteresis1, hysteresis2):
    #   send_message()




#   cip_driver.generic_message(service=Services.set_attribute_single, class_code=0xA2, instance=13, attribute=5, request_data=bytearray([0b00000000] + [0b00000000] + [0b00000000] + [0b00000000] + [0b00000000]*12))


"""
check for vacuum error/ status when possible
"""