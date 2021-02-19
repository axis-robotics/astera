import base
import cv2

for i in range(1, 4):
    img = base.preprocess_cam(f"examples/{i}.jpg")
    flowers = base.detect_flowers(img)
    y = base.classify_flowers(flowers)
    ## MAP TRANSFORM y directly
    original_image = cv2.imread(f"examples/{i}.jpg")
    print(len(flowers))
    for j in flowers:
        original_image = cv2.circle(original_image, j['center'], 1, (0, 0, 255), 5)
    cv2.imwrite(f"examples/done - {i}.jpg", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

exit()
import smbus2
import json
import Jetson.GPIO as GPIO
import base

bus = smbus2.SMBus(1)
ADDRESS = 0x04
SWITCH_INPUT_PIN = 6
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SWITCH_INPUT_PIN, GPIO.IN)

def write_data_to_avr(value):
    byteValue = [ord(c) for c in value]
    return bus.write_i2c_block_data(ADDRESS, 0x00, byteValue)

def read_data_from_avr():
    number = bus.read_byte(ADDRESS)
    # number = bus.read_byte_data(ADDRESS, 1)
    return number

def get_coordinates():
    img = base.preprocess_cam('test.jpg')
    detected_flowers = base.detect_flowers(img)
    chamomile_flowers = base.classify_flowers(detected_flowers)
    angles = list(map(base.transformation_matrix, chamomile_flowers))
    return angles

def main():
    try:
        while GPIO.input(SWITCH_INPUT_PIN) == GPIO.HIGH:
            coordinates = get_coordinates()
            write_data_to_avr(json.dumps(coordinates))
            while read_data_from_avr() != "done": pass
            ## TODO: Take another shot.
            ## TODO: Move a lateral step.
            pass
    except:
        main()
    return main()

main()
GPIO.cleanup()