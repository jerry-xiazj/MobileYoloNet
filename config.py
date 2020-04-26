# Author: Jerry Xia <jerry_xiazj@outlook.com>

import os
import numpy as np
from easydict import EasyDict as edict


CFG = edict()

CFG.train_file = "./data/train_file"
CFG.log_dir = "./log/"
CFG.checkpoint_dir = CFG.log_dir + "ckpt-voc"
CFG.checkpoint_prefix = os.path.join(CFG.checkpoint_dir, "ckpt")

# config train
CFG.data_to_use = 100
CFG.data_aug = False
CFG.batch_size = 5
CFG.train_epoch = 100
CFG.batch_per_epoch = 20
CFG.output_step = CFG.batch_per_epoch
CFG.lr_init = 1.0e-5
CFG.lr_peak = 7.0e-5
CFG.lr_end = 1.0e-5
CFG.warmup_steps = 3 * CFG.batch_per_epoch
CFG.total_steps = CFG.train_epoch * CFG.batch_per_epoch

# config data
CFG.classes = ["person", "bird", "cat", "cow", "dog", "horse",
               "sheep", "aeroplane", "bicycle", "boat", "bus",
               "car", "motorbike", "train", "bottle", "chair",
               "diningtable", "pottedplant", "sofa", "tvmonitor"]
CFG.num_classes = len(CFG.classes)

# config model
CFG.input_shape = np.array([416, 416])
CFG.scales = np.array([32, 16, 8])
CFG.num_scales = np.array(CFG.scales.shape[0])
CFG.fm_size = np.array(CFG.input_shape[np.newaxis, :] // CFG.scales[:, np.newaxis])
CFG.anchors = np.reshape(
    [116, 90, 156, 198, 373, 326, 30, 61, 62, 45, 59, 119, 10, 13, 16, 30, 33, 23],
    (CFG.num_scales, -1, 2))
CFG.anchor_mask = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
CFG.anc_per_scale = CFG.anchors.shape[1]
CFG.ignore_thresh = 0.5  # used to determine False Negtive
CFG.nms_thresh = 0.5  # used to determine whether to drop a box with lower confidence
CFG.iou_thresh = 0.5  # used to determine whether to drop a box with higher iou
CFG.max_boxes = 20
