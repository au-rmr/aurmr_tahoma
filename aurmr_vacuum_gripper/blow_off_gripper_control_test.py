from pycomm3 import CIPDriver, Services, configure_default_logger, LOG_VERBOSE, exceptions, SINT, UINT, USINT

if __name__ == '__main__':
    valves = [False, False, True, True] #length 4 because we will only use 4 vacuum plates
    cip_driver = CIPDriver('192.168.137.2')
    cip_driver.open()
    # cip_driver.generic_message(service=Services.set_attribute_single, class_code=0xA2, instance=13, attribute=5, request_data=bytearray([0b00000000] + [0b00000000] + [0b00000010] + [0b00000010] + [0b00000000]*12))


'''
#for terminal only
from pycomm3 import CIPDriver, Services, configure_default_logger, LOG_VERBOSE, exceptions, SINT, UINT, USINT, BYTE
ip = '192.168.137.2'
cip_driver = CIPDriver(ip)
cip_driver.open()

cip_driver.generic_message(service=Services.get_attribute_single, class_code=0xA2, instance=66, attribute=5)

sets blow,cip_driver.generic_message(service=Services.set_attribute_single, class_code=0xA2, instance=13, attribute=5, request_data=bytearray([0b00000000] + [0b00000000] + [0b00000000] + [0b00000000] + [0b00000000]*12))
active = (2).to_bytes(1, byteorder='little')
zeroByte = bytes(1)
data = zeroByte + zeroByte+active+active+(zeroByte*12)
cip_driver.generic_message(service=Services.set_attribute_single, class_code=0xA2, instance=13, attribute=5, request_data=data)
'''