# Author: Jerry Xia <jerry_xiazj@outlook.com>

import cv2
import time
import numpy as np
import tensorflow as tf
import core.utils as utils
from config import CFG
from core.model import MobileYolo_small

video_path = './data/test.mp4'
output_path = './log/test_out.avi'
output_format = 'XVID'

tf.keras.backend.set_learning_phase(False)
model_input = tf.keras.layers.Input([CFG.input_shape[0], CFG.input_shape[1], 3])
model_output = MobileYolo_small(model_input, training=False)
model = tf.keras.Model(model_input, model_output)

ckpt = tf.train.Checkpoint(model=model)
manager = tf.train.CheckpointManager(ckpt, CFG.checkpoint_dir, max_to_keep=3)
if manager.latest_checkpoint:
    ckpt.restore(manager.latest_checkpoint)
    tf.print("Restored from ", manager.latest_checkpoint)
else:
    raise Exception("No valid checkpoint!")


@tf.function
def pred(model, batch):
    boxes, scores, classes, valid_detection = model(batch)
    return boxes, scores, classes, valid_detection


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

vid = cv2.VideoCapture(video_path)
times = []
# by default VideoCapture returns float instead of int
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(vid.get(cv2.CAP_PROP_FPS))
codec = cv2.VideoWriter_fourcc(*output_format)
out = cv2.VideoWriter(output_path, codec, fps, (width, height))

while True:
    _, img = vid.read()

    if img is None:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_img = utils.resize_img(img_rgb, CFG.input_shape)
    batch = tf.cast(tf.expand_dims(resized_img, 0), tf.float32)
    t1 = time.time()
    boxes, scores, classes, valid_detection = pred(model, batch)
    t2 = time.time()
    times.append(t2-t1)

    s_box = np.array(boxes[0, 0:valid_detection[0], :])
    s_score = np.array(scores[0, 0:valid_detection[0]])
    s_class = np.array(classes[0, 0:valid_detection[0]], dtype=np.int32)
    s_box = utils.restore_box(img, CFG.input_shape, s_box[:])
    img = utils.draw_bbox(img, s_box, s_score, s_class, CFG.classes, show_label=True)
    img = cv2.putText(img, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),
                      cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    out.write(img)
    if cv2.waitKey(1) == ord('q'):
        break
