from pycomm3 import CIPDriver, Services, configure_default_logger, LOG_VERBOSE, exceptions, SINT, UINT, USINT

if __name__ == '__main__':
    ip = '192.168.137.2'
    cip_driver = CIPDriver(ip)
    cip_driver.open()
    cip_driver.generic_message(service=Services.set_attribute_single, class_code=0xA2, instance=13, attribute=5, request_data=bytearray([0b00000000] + [0b00000000] + [0b00000010] + [0b00000010] + [0b00000000]*12))