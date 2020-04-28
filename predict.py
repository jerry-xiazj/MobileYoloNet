# Author: Jerry Xia <jerry_xiazj@outlook.com>

import cv2
import time
import numpy as np
import tensorflow as tf
import core.utils as utils
from config import CFG
from core.model import MobileYolo_small


output_path = CFG.log_dir

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
    tf.print("Initializing from scratch.")

with open(CFG.train_file, 'r') as rf:
    ann_lines = rf.readlines()
ann_lines = [ann.rstrip('\n') for ann in ann_lines]

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

start = time.time()
for i in ann_lines:
    img_path = i.split(' ')[0]
    img_name = img_path.split('/')[-1]

    # img = cv2.imread(img_path)
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    resized_img = utils.resize_img(img, CFG.input_shape)
    batch = tf.cast(tf.expand_dims(resized_img, 0), tf.float32)
    boxes, scores, classes, valid_detection = model(batch)

    s_box = np.array(boxes[0, 0:valid_detection[0], :])
    s_score = np.array(scores[0, 0:valid_detection[0]])
    s_class = np.array(classes[0, 0:valid_detection[0]], dtype=np.int32)
    s_box = utils.restore_box(img, CFG.input_shape, s_box[:])
    label_img = utils.draw_bbox(cv2.imread(img_path), s_box, s_score, s_class, CFG.classes, show_label=True)
    cv2.imwrite(output_path+'pred_'+img_name, label_img)

tf.print("Finish predicting. Time taken: ", time.time()-start, " sec.")
