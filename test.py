import smbus, json

bus = smbus.SMBus(1)
ADDRESS = 0x04


def write_data_to_avr(value):
    return bus.write_byte(ADDRESS, value)

def read_data_from_avr():
    number = bus.read_byte(ADDRESS)
    # number = bus.read_byte_data(ADDRESS, 1)
    return number

# sudo apt-get install libi2c-dev i2c-tools
# sudo i2cdetect -y -r 1
fake_data_testing = [
    [6955, 17095, 17999],
    [5748, 6857, 9000],
    [12233, 2177, 3032],
    [5326, 13489, 1044],
]
message = '["Hello World."]' #json.dumps(fake_data_testing)
write_data_to_avr(message)

x = read_data_from_avr()
print(x)


exit()
import cv2
cam = cv2.VideoCapture()
cam.open("rtsp://192.168.42.129:8080/h264_pcm.sdp")

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    
