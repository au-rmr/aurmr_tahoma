from pycomm3 import CIPDriver, Services, configure_default_logger, LOG_VERBOSE, exceptions, SINT, UINT, USINT
import time

CLOSE_VALVE = 0
SUCTION = 1
BLOW_OFF = 2

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

if __name__ == '__main__':
    ip = '192.168.137.2'
    cip_driver = CIPDriver(ip)
    cip_driver.open()

    send_message(13, CLOSE_VALVE.to_bytes(1, 'little')*2 + SUCTION.to_bytes(1, 'little') + CLOSE_VALVE.to_bytes(1, 'little')*13)
    time.sleep(2)

    send_message(13, SUCTION.to_bytes(1, 'little')*4 + CLOSE_VALVE.to_bytes(1, 'little')*12)
    time.sleep(1)
    send_message(13, CLOSE_VALVE.to_bytes(1, 'little')*2 + SUCTION.to_bytes(1, 'little')+ CLOSE_VALVE.to_bytes(1, 'little')*13)
    time.sleep(1)
    send_message(13, BLOW_OFF.to_bytes(1, 'little')*2 + SUCTION.to_bytes(1, 'little')+ BLOW_OFF.to_bytes(1, 'little')*13)
    time.sleep(1)
    send_message(13, CLOSE_VALVE.to_bytes(1, 'little')*2 + SUCTION.to_bytes(1, 'little')+ CLOSE_VALVE.to_bytes(1, 'little')*13)
    time.sleep(1)

    send_message(13, SUCTION.to_bytes(1, 'little') + CLOSE_VALVE.to_bytes(1, 'little') + SUCTION.to_bytes(1, 'little') + CLOSE_VALVE.to_bytes(1, 'little')*13)
    time.sleep(1)
    send_message(13, CLOSE_VALVE.to_bytes(1, 'little') + SUCTION.to_bytes(1, 'little') + SUCTION.to_bytes(1, 'little') + CLOSE_VALVE.to_bytes(1, 'little')*13)
    time.sleep(1)
    send_message(13, BLOW_OFF.to_bytes(1, 'little') + CLOSE_VALVE.to_bytes(1, 'little') + SUCTION.to_bytes(1, 'little') + CLOSE_VALVE.to_bytes(1, 'little')*13)
    time.sleep(1)

    send_message(13, CLOSE_VALVE.to_bytes(1, 'little')*2 + SUCTION.to_bytes(1, 'little') + SUCTION.to_bytes(1, 'little') + CLOSE_VALVE.to_bytes(1, 'little')*12)
    time.sleep(1)
    send_message(13, CLOSE_VALVE.to_bytes(1, 'little')+ BLOW_OFF.to_bytes(1, 'little') + SUCTION.to_bytes(1, 'little') + CLOSE_VALVE.to_bytes(1, 'little')*13)
    time.sleep(1)
    send_message(13, SUCTION.to_bytes(1, 'little') + CLOSE_VALVE.to_bytes(1, 'little') + SUCTION.to_bytes(1, 'little') + CLOSE_VALVE.to_bytes(1, 'little')*13)
    time.sleep(1)
    send_message(13, CLOSE_VALVE.to_bytes(1, 'little')*2 + SUCTION.to_bytes(1, 'little') + BLOW_OFF.to_bytes(1, 'little') + CLOSE_VALVE.to_bytes(1, 'little')*12)
    time.sleep(1)

    send_message(13, CLOSE_VALVE.to_bytes(1, 'little')*2 + SUCTION.to_bytes(1, 'little') + CLOSE_VALVE.to_bytes(1, 'little')*13)
    time.sleep(1)
    send_message(13, BLOW_OFF.to_bytes(1, 'little')*2 + SUCTION.to_bytes(1, 'little') + BLOW_OFF.to_bytes(1, 'little')*13)
    time.sleep(1)
    send_message(13, CLOSE_VALVE.to_bytes(1, 'little')*2 + SUCTION.to_bytes(1, 'little') + CLOSE_VALVE.to_bytes(1, 'little')*13)
    time.sleep(1)

    # send_message(13, CLOSE_VALVE.to_bytes(1, 'little')*2 + BLOW_OFF.to_bytes(1, 'little') + CLOSE_VALVE.to_bytes(1, 'little')*13)
    # time.sleep(1)
    # send_message(13, CLOSE_VALVE.to_bytes(1, 'little')*16)
    # time.sleep(1)
