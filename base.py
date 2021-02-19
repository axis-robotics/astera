import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Supress TF debugging info. (Uncomment in case of debugging)

import math
import cv2
import colorsys, random
import numpy as np
import tensorflow as tf

def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def detect_coordinates(image, bboxes, classes=read_class_names("./weights/classes.names"), show_label=True):
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    out_boxes, out_scores, out_classes, num_boxes = bboxes
    return_value = []
    for i in range(num_boxes[0]):
        if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue
        coor = out_boxes[0][i]
        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)

        fontScale = 0.5
        score = out_scores[0][i]
        class_ind = int(out_classes[0][i])
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
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)
    # return tf.concat([boxes, pred_conf], axis=-1)
    return (boxes, pred_conf)







from tensorflow.python.saved_model import tag_constants
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

class FLAGS: # Customizable settings.
    
    ## AI code settings
    detection_weights = "weights/yolov4-detection.tflite"
    classification_weights = "weights/yolov4-detection.tflite"
    size = 416
    iou = 0.45
    score = 0.52
    
    ## Transformation settings
    L, l = 0.160, 0.500
    R, r = 0.085, 0.065
    z = -0.400
    ANGLE_SHIFT, ANGLE_SCALE = 90, 100
    X_RATIO, Y_RATIO = 1, 1


def preprocess_cam(image_path): # Generate the image used in detection.
    ## TODO: Camera and blending.
    original_image = cv2.imread(image_path)
    return cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

session = InteractiveSession(config=ConfigProto())

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
    return detect_coordinates(original_image, pred_bbox)

# Classification model.

## Load TFLite model and allocate tensors.
classification_interpreter = tf.lite.Interpreter(model_path=FLAGS.classification_weights)
classification_interpreter.allocate_tensors()

## Get input and output tensors.
classification_input_details = classification_interpreter.get_input_details()
classification_output_details = classification_interpreter.get_output_details()


def classify_flowers(output_flowers):
    return [flower['center'] for flower in output_flowers]

    images_data = [
        cv2.resize(flower['image'], (FLAGS.size, FLAGS.size)) / 255.0 for flower in output_flowers
    ]
    input_data = np.asarray(images_data, dtype=np.float32)
    classification_interpreter.set_tensor(classification_input_details[0]['index'], images_data)

    classification_interpreter.invoke()

    output_data = classification_interpreter.get_tensor(classification_output_details[0]['index'])
    print(output_data)

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



for i in range(1, 3):
    img = preprocess_cam(f"examples/{i}.jpg")
    flowers = detect_flowers(img)
    y = classify_flowers(flowers)
    original_image = cv2.imread(f"examples/{i}.jpg")
    print(len(flowers))
    for j in flowers:
        original_image = cv2.circle(original_image, j['center'], 1, (0, 0, 255), 5)
    cv2.imwrite(f"examples/done - {i}.jpg", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()