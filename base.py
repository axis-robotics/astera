import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Supress TF debugging info. (Uncomment in case of debugging)

import tensorflow as tf
import cv2
import numpy as np

import core.utils as utils
from core.yolov4 import filter_boxes

from tensorflow.python.saved_model import tag_constants
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

class FLAGS: # Customizable settings.
    detection_weights = "weights\\yolov4-detection.tflite"
    classification_weights = "weights\\yolov4-detection.tflite"
    size = 416
    iou = 0.45
    score = 0.52

def preprocess_cam(image_path): # Generate the image used in detection.

    original_image = cv2.imread(image_path)
    return cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

session = InteractiveSession(config=ConfigProto(device_count = {'GPU': 0})) # Disables the GPU. (Remove the device_count parameter to re-enable)

# Detection model.

## Load TFLite model and allocate tensors.
detection_interpreter = tf.lite.Interpreter(model_path=FLAGS.detection_weights)
detection_interpreter.allocate_tensors()

## Get input and output tensors.
detection_input_details = detection_interpreter.get_input_details()
detection_output_details = detection_interpreter.get_output_details()

def detect_flowers(original_image):

    image_data = cv2.resize(original_image, (FLAGS.size, FLAGS.size)) / 255.0
    image_data = np.asarray([image_data]).astype(np.float32)


    detection_interpreter.set_tensor(detection_input_details[0]['index'], image_data)
    detection_interpreter.invoke()

    pred = [detection_interpreter.get_tensor(detection_output_details[i]['index']) for i in range(len(detection_output_details))]
    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([FLAGS.size, FLAGS.size]))

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=FLAGS.iou,
        score_threshold=FLAGS.score
    )

    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    return utils.detect_coordinates(original_image, pred_bbox)

# Classification model.

## Load TFLite model and allocate tensors.
classification_interpreter = tf.lite.Interpreter(model_path=FLAGS.classification_weights)
classification_interpreter.allocate_tensors()

## Get input and output tensors.
classification_input_details = classification_interpreter.get_input_details()
classification_output_details = classification_interpreter.get_output_details()


def classify_flowers(output_flowers):
    # images_data = [
    #     cv2.resize(flower['image'], (FLAGS.size, FLAGS.size)) / 255.0 for flower in output_flowers
    # ]
    # input_data = np.asarray(images_data, dtype=np.float32)
    # classification_interpreter.set_tensor(classification_input_details[0]['index'], images_data)

    # classification_interpreter.invoke()

    # output_data = classification_interpreter.get_tensor(classification_output_details[0]['index'])
    # print(output_data)
    return [flower['center'] for flower in output_flowers]

def transformation_matrix(flower):
    x, y = flower
    return (x, y)

img = preprocess_cam('test.jpg')
detected_flowers = detect_flowers(img)
chamomile_flowers = classify_flowers(detected_flowers)
angles = list(map(transformation_matrix, chamomile_flowers))
print(angles)
