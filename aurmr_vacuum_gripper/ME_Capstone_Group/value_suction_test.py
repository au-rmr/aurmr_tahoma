from pycomm3 import CIPDriver, Services, configure_default_logger, LOG_VERBOSE, exceptions, SINT, UINT, USINT


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

def toInt_16x02(byteArray):
    converted = []
    for i in range(0,16,2):
        converted.append(int.from_bytes(byteArray[0:2], "little"))
    return converted

if __name__ == '__main__':
    ip = '192.168.137.2'
    cip_driver = CIPDriver(ip)
    cip_driver.open()
    cip_driver.generic_message(service=Services.set_attribute_single, class_code=0xA2, instance=13, attribute=5, request_data=bytearray([0b00000001] + [0b00000000] + [0b00000000] + [0b00000000] + [0b00000000]*12))
    while True:
        print(toInt_16x02(get_status(515)))