import smbus
import json
import Jetson.GPIO as GPIO

bus = smbus.SMBus(1)
ADDRESS = 0x04
SWITCH_INPUT_PIN = 6
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SWITCH_INPUT_PIN, GPIO.IN)


def write_data_to_avr(value):
    byteValue = [ord(c) for c in value]
    return bus.write_i2c_block_data(ADDRESS, 0x00, byteValue)

def read_data_from_avr():
    # TODO: test...
    number = bus.read_byte(ADDRESS)
    # number = bus.read_byte_data(ADDRESS, 1)
    return number


# def main():
#     try:
#         while GPIO.input(SWITCH_INPUT_PIN) == GPIO.HIGH:
#             coordinates = []
#             write_data_to_avr(json.dumps(coordinates))
#             while read_data_from_avr() != "done": pass
#             coordinates = []
#             write_data_to_avr(json.dumps(coordinates))
#             while read_data_from_avr() != "done": pass
#             ## TODO: Move a lateral step.
#             pass
#     except:
#         main()
#     return main()


fake_data_testing = [
    [69.55, 70.95, 80.99],
    [57.48, 68.57, 90.00],
    [22.33, 21.77, 30.32],
    [53.26, 34.89, 10.44],
]

write_data_to_avr(json.dumps(fake_data_testing))

x = read_data_from_avr()
print(x)

# while read_data_from_avr() != "done": print("not yes")





GPIO.cleanup()