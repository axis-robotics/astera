import serial

port = '/dev/ttyTHS1'
baud = 9600
s = serial.Serial(port,baud)

def send_byte(byte):
    byte_data = byte.to_bytes(1, 'little')
    s.write(byte_data)

def send_angle(angle):
    msb = angle // 255
    lsb = angle % 255
    send_byte(msb)
    send_byte(lsb)

while True:
    x = int(input("Enter first angle: "))
    y = int(input("Enter second angle: "))
    z = int(input("Enter third angle: "))
    send_angle(x)
    send_angle(y)
    send_angle(z)
    c = input("Enter c to continue, x to exit: ")
    if c == 'x':
        break
