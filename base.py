import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Supress TF debugging info. (Uncomment in case of debugging)

import random
random.seed(0)

import cv2
import colorsys
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

class FLAGS: # Customizable settings.
    
    ## AI code settings
    WEIGHTS_PATH = "weights/yolov4-detection.tflite"
    CLASS_NAMES = read_class_names("weights/classes.names")
    INPUT_SIZE = 416
    IOU_THRESHOLD = 0.45
    MIN_SCORE = 0.52
    
    ## Transformation settings
    L, l = 0.160, 0.500
    R, r = 0.085, 0.065
    z = -0.400
    ANGLE_SHIFT, ANGLE_SCALE = 0, 1
    X_RATIO, Y_RATIO = 1, 1

    ## Camera settings
    CAM_WIDTH, CAM_HEIGHT = 1280, 720



def detect_coordinates(image, bboxes):

    num_classes = len(FLAGS.CLASS_NAMES)
    image_h, image_w, _ = image.shape

    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.shuffle(colors)

    out_boxes, out_scores, out_classes, num_boxes = bboxes
    return_value = []
    for i in range(num_boxes[0]):
        if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue
        coor = out_boxes[0][i]
        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)

        c1, c2 = (int(coor[1]), int(coor[0])), (int(coor[3]), int(coor[2]))
        return_value.append(
            {
                "image": image[c1[1]:c2[1], c1[0]:c2[0]],
                "center": ((c1[0]+c2[0]) // 2, (c1[1]+c2[1]) // 2)
            }
        )
    return return_value


def filter_boxes(box_xywh, scores, score_threshold=0.4, input_shape = tf.constant([416,416])):
    scores_max = tf.math.reduce_max(scores, axis=-1)

    mask = scores_max >= score_threshold
    class_boxes = tf.boolean_mask(box_xywh, mask)
    pred_conf = tf.boolean_mask(scores, mask)
    class_boxes = tf.reshape(class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]])
    pred_conf = tf.reshape(pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])

    box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)

    input_shape = tf.cast(input_shape, dtype=tf.float32)

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    box_mins = (box_yx - (box_hw / 2.)) / input_shape
    box_maxes = (box_yx + (box_hw / 2.)) / input_shape
    boxes = tf.concat([
        box_mins[..., 0:1], box_mins[..., 1:2],
        box_maxes[..., 0:1], box_maxes[..., 1:2]
    ], axis=-1)
    return (boxes, pred_conf)

# Camera

## TODO: Camera and blending.
camera_session = cv2.VideoCapture(f"nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int){FLAGS.CAM_WIDTH}, height=(int){FLAGS.CAM_HEIGHT},format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert !  appsink")

def read_cam(): # Generate the image used in detection.
    if camera_session.isOpened():
        _, img = camera_session.read()
        return img
    return None

def release_cam(): camera_session.release()

session = InteractiveSession(config=ConfigProto())

# Detection model.

## Load TFLite model and allocate tensors.
detection_interpreter = tf.lite.Interpreter(model_path=FLAGS.WEIGHTS_PATH)
detection_interpreter.allocate_tensors()

## Get input and output tensors.
detection_input_details = detection_interpreter.get_input_details()
detection_output_details = detection_interpreter.get_output_details()

def detect_flowers(original_image):

    image_data = cv2.resize(original_image, (FLAGS.INPUT_SIZE, FLAGS.INPUT_SIZE)) / 255.0
    image_data = np.asarray([image_data]).astype(np.float32)

    detection_interpreter.set_tensor(detection_input_details[0]['index'], image_data)
    detection_interpreter.invoke()

    pred = [detection_interpreter.get_tensor(detection_output_details[i]['index']) for i in range(len(detection_output_details))]
    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([FLAGS.INPUT_SIZE, FLAGS.INPUT_SIZE]))

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=FLAGS.IOU_THRESHOLD,
        score_threshold=FLAGS.MIN_SCORE
    )

    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    return detect_coordinates(original_image, pred_bbox)


# Transformation algorithm.

def transformation_matrix(flower):
    SQRT_3 = np.sqrt(3)

    L, l = FLAGS.L, FLAGS.l
    R, r = FLAGS.R, FLAGS.r
    z = FLAGS.z

    x, y = flower[0] * FLAGS.X_RATIO, flower[1] * FLAGS.Y_RATIO
    w_b, u_b, s_b = R/2, R, SQRT_3 * R
    w_p, u_p, s_p = r/2, r, SQRT_3 * r
    a = w_b - u_p
    b = s_p / 2 - 0.5 * SQRT_3 * w_b
    c = w_p - 0.5 * w_b

    E = np.array([
        2 * L * (y + a),
        -L * ( SQRT_3 * (x + b) + y + c ),
        L * ( SQRT_3 * (x - b) - y - c ),
    ])
    G = np.array([
        x**2 + y**2 + z**2 + a**2 + L**2 + 2*y*a - l**2,
        x**2 + y**2 + z**2 + b**2 + c**2 + L**2 + 2*x*b + 2*y*c - l**2,
        x**2 + y**2 + z**2 + b**2 + c**2 + L**2 - 2*x*b + 2*y*c - l**2,
    ])
    F = np.array(
        [2 * z * L] * 3
    )

    t1_eqn = (-F + np.sqrt(E**2 + F**2 - G**2)) / (G - E)
    t2_eqn = (-F - np.sqrt(E**2 + F**2 - G**2)) / (G - E)
    t1 = (2 * np.arctan(t1_eqn) * 180 / np.pi).reshape((3, 1))
    t2 = (2 * np.arctan(t2_eqn) * 180 / np.pi).reshape((3, 1))
    t = np.round(np.concatenate((t1, t2), axis=1).tolist(), 2)
    return [
        ((i[0] + FLAGS.ANGLE_SHIFT) * FLAGS.ANGLE_SCALE if np.abs(i[0]) < np.abs(i[1]) else i[1]) for i in t
    ]
