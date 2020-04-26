# Author: Jerry Xia <jerry_xiazj@outlook.com>

import cv2
import random
import colorsys
import numpy as np
import tensorflow as tf


def resize_img(img, input_shape, true_box=None):
    """
    Resize the image according to the aspect ratio, fill the blank area with (128, 128, 128).

    Input:
    img: Image data read in by cv2.imread().
    input_shape: (h, w), default=[416,416].
    true_box: Contain (xmin, ymin, xmax, ymax, class), shape=(num_boxes, 5).

    Output:
    img_data: Regularized images.
    true_box: Ground truth box in resized image.
    """
    img_h, img_w, _ = img.shape
    inp_h, inp_w = input_shape

    new_w = int(img_w * min(inp_w/img_w, inp_h/img_h))
    new_h = int(img_h * min(inp_w/img_w, inp_h/img_h))
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    img_data = np.full(shape=[inp_h, inp_w, 3], fill_value=128.0)
    dw, dh = (inp_w-new_w)//2, (inp_h-new_h)//2
    img_data[dh:dh+new_h, dw:dw+new_w, :] = resized_img
    img_data = img_data / 255.

    if true_box is None:
        return img_data

    # true_box = np.array(true_box).astype(np.int32)
    true_box[:, [0, 2]] = true_box[:, [0, 2]] * new_w // img_w + dw
    true_box[:, [1, 3]] = true_box[:, [1, 3]] * new_h // img_h + dh

    return img_data, true_box


def org2reg(data, input_shape, use_np=False):
    if use_np is True:
        data = data.astype(np.float32)
        input_shape = input_shape.astype(np.float32)
        temp_xy = data[..., 0:2] / input_shape[::-1]
        temp_wh = data[..., 2:4] / input_shape[::-1]
        return np.concatenate([temp_xy, temp_wh, data[..., 4:]], axis=-1)
    else:
        data = tf.cast(data, tf.float32)
        input_shape = tf.cast(input_shape, tf.float32)
        temp_xy = data[..., 0:2] / input_shape[::-1]
        temp_wh = data[..., 2:4] / input_shape[::-1]
        return tf.concat([temp_xy, temp_wh, data[..., 4:]], axis=-1)


def reg2fm(data, fm_size, use_np=False):
    if use_np is True:
        data = data.astype(np.float32)
        fm_size = fm_size.astype(np.float32)
        temp_xy = data[..., 0:2] * fm_size[::-1]
        temp_wh = data[..., 2:4] * fm_size[::-1]
        return np.concatenate([temp_xy, temp_wh, data[..., 4:]], axis=-1)
    else:
        data = tf.cast(data, tf.float32)
        fm_size = tf.cast(fm_size, tf.float32)
        temp_xy = data[..., 0:2] * fm_size[::-1]
        temp_wh = data[..., 2:4] * fm_size[::-1]
        return tf.concat([temp_xy, temp_wh, data[..., 4:]], axis=-1)


def org2fm(data, input_shape, fm_size, use_np=False):
    if use_np is True:
        data = data.astype(np.float32)
        input_shape = input_shape.astype(np.float32)
        fm_size = fm_size.astype(np.float32)
        temp_xy = data[..., 0:2] / input_shape[::-1] * fm_size[::-1]
        temp_wh = data[..., 2:4] / input_shape[::-1] * fm_size[::-1]
        return np.concatenate([temp_xy, temp_wh, data[..., 4:]], axis=-1)
    else:
        data = tf.cast(data, tf.float32)
        input_shape = tf.cast(input_shape, tf.float32)
        fm_size = tf.cast(fm_size, tf.float32)
        temp_xy = data[..., 0:2] / input_shape[::-1] * fm_size[::-1]
        temp_wh = data[..., 2:4] / input_shape[::-1] * fm_size[::-1]
        return tf.concat([temp_xy, temp_wh, data[..., 4:]], axis=-1)


def fm2org(data, input_shape, fm_size, use_np=False):
    if use_np is True:
        data = data.astype(np.float32)
        input_shape = input_shape.astype(np.float32)
        fm_size = fm_size.astype(np.float32)
        temp_xy = data[..., 0:2] / fm_size[::-1] * input_shape[::-1]
        temp_wh = data[..., 2:4] / fm_size[::-1] * input_shape[::-1]
        return np.concatenate([temp_xy, temp_wh, data[..., 4:]], axis=-1)
    else:
        data = tf.cast(data, tf.float32)
        input_shape = tf.cast(input_shape, tf.float32)
        fm_size = tf.cast(fm_size, tf.float32)
        temp_xy = data[..., 0:2] / fm_size[::-1] * input_shape[::-1]
        temp_wh = data[..., 2:4] / fm_size[::-1] * input_shape[::-1]
        return tf.concat([temp_xy, temp_wh, data[..., 4:]], axis=-1)


def xyxy2xywh(data, use_np=False):
    temp_xy = (data[..., 0:2] + data[..., 2:4]) / 2
    temp_wh = data[..., 2:4] - data[..., 0:2]
    if use_np is True:
        return np.concatenate([temp_xy, temp_wh, data[..., 4:]], axis=-1)
    else:
        return tf.concat([temp_xy, temp_wh, data[..., 4:]], axis=-1)


def xywh2yxyx(data, use_np=False):
    temp_xymin = data[..., 0:2] - data[..., 2:4] / 2
    temp_xymax = data[..., 0:2] + data[..., 2:4] / 2
    if use_np is True:
        return np.concatenate([temp_xymin[..., ::-1], temp_xymax[..., ::-1], data[..., 4:]], axis=-1)
    else:
        return tf.concat([temp_xymin[..., ::-1], temp_xymax[..., ::-1], data[..., 4:]], axis=-1)


def t2xywh_fm(data, grid_cell, anchors):
    data = tf.cast(data, tf.float32)
    grid_cell = tf.cast(grid_cell, tf.float32)
    anchors = tf.cast(anchors, tf.float32)

    temp_xy = tf.sigmoid(data[..., 0:2]) + grid_cell
    temp_wh = anchors * tf.math.exp(data[..., 2:4])
    return tf.concat([temp_xy, temp_wh, data[..., 4:]], axis=-1)


def xywh_fm2t(data, grid_cell, anchors, conf_mask):
    """
    Use conf_mask to filter the empty boxes in data and fill with 1.
    Due to log(1)=0, the empty boxes are 0 in output.
    """
    data = tf.cast(data, tf.float32)
    grid_cell = tf.cast(grid_cell, tf.float32)
    anchors = tf.cast(anchors, tf.float32)

    temp_txy = data[..., 0:2] - grid_cell

    temp_twh = data[..., 2:4] / anchors
    temp_twh = tf.keras.backend.switch(tf.cast(conf_mask, tf.bool), temp_twh, tf.ones_like(temp_twh))
    temp_twh = tf.math.log(temp_twh)
    return tf.concat([temp_txy, temp_twh, data[..., 4:]], axis=-1)


def make_grid_cell(fm_size):
    grid_cell = tf.meshgrid(tf.range(fm_size[1]), tf.range(fm_size[0]))
    grid_cell = tf.stack(grid_cell, axis=-1)[tf.newaxis, :, :, tf.newaxis, :]  # [1, gy, gx, 1, 2]
    grid_cell = tf.cast(grid_cell, tf.float32)
    return grid_cell


def iou(box1, box2, use_np=False):
    """ box* = [..., (x, y, w, h, ...)] """
    box1_area = box1[..., 2] * box1[..., 3]
    box2_area = box2[..., 2] * box2[..., 3]

    box1_min = box1[..., 0:2] - 0.5 * box1[..., 2:4]
    box1_max = box1[..., 0:2] + 0.5 * box1[..., 2:4]
    box2_min = box2[..., 0:2] - 0.5 * box2[..., 2:4]
    box2_max = box2[..., 0:2] + 0.5 * box2[..., 2:4]
    if use_np is True:
        inter_min = np.maximum(box1_min[..., 0:2], box2_min[..., 0:2])
        inter_max = np.minimum(box1_max[..., 0:2], box2_max[..., 0:2])
    else:
        inter_min = tf.math.maximum(box1_min[..., 0:2], box2_min[..., 0:2])
        inter_max = tf.math.minimum(box1_max[..., 0:2], box2_max[..., 0:2])
    inter_wh = inter_max - inter_min
    if use_np is True:
        inter_wh = np.where(inter_wh > 0., inter_wh, np.zeros_like(inter_wh))
    else:
        inter_wh = tf.keras.backend.switch(inter_wh > 0., inter_wh, tf.zeros_like(inter_wh))
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]
    union_area = box1_area + box2_area - inter_area

    return 1.0 * inter_area / union_area


def restore_box(image, input_shape, box):
    """box: yxyx"""
    image_h, image_w = image.shape[0], image.shape[1]
    input_h, input_w = input_shape[0], input_shape[1]
    ratio = min(input_w/image_w, input_h/image_h)
    new_w = int(image_w * ratio)
    new_h = int(image_h * ratio)
    dw = (input_w-new_w)//2
    dh = (input_h-new_h)//2

    box = np.concatenate(
        [(box[:, 0:1]-dh)/ratio, (box[:, 1:2]-dw)/ratio,
         (box[:, 2:3]-dh)/ratio, (box[:, 3:4]-dw)/ratio], axis=-1).astype(np.int32)
    return box


def draw_bbox(image, s_box, s_score, s_class, classes, show_label=True):
    """
    s_box: [y_min, x_min, y_max, x_max]
    """

    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i in range(s_score.shape[0]):
        fontScale = 0.5
        bbox_color = colors[s_class[i]]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (s_box[i, 1], s_box[i, 0]), (s_box[i, 3], s_box[i, 2])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[s_class[i]], s_score[i])
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
            cv2.rectangle(image, c1,
                          (c1[0] + t_size[0], c1[1] - t_size[1] - 3),
                          bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)

    return image
