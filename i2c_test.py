import smbus, json

bus = smbus.SMBus(1)
ADDRESS = 0x04


def write_data_to_avr(value):
    byteValue = [ord(c) for c in value]
    return bus.write_i2c_block_data(ADDRESS, 0x00, byteValue)

def read_data_from_avr():
    number = bus.read_byte(ADDRESS)
    # number = bus.read_byte_data(ADDRESS, 1)
    return number

# sudo apt-get install libi2c-dev i2c-tools
# sudo i2cdetect -y -r 1
fake_data_testing = [
    [69.55, 70.95, 80.99],
    [57.48, 68.57, 90.00],
    [22.33, 21.77, 30.32],
    [53.26, 34.89, 10.44],
]
message = '["Hello World."]' #json.dumps(fake_data_testing)
write_data_to_avr(message)

x = read_data_from_avr()
print(x)
