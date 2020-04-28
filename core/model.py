# Author: Jerry Xia <jerry_xiazj@outlook.com>

import tensorflow as tf
import core.utils as utils
from config import CFG


class Identity(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, input):
        return input


class HardSigmoid(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.relu6 = tf.keras.layers.ReLU(max_value=6)

    def call(self, input):
        return self.relu6(input + 3.0) / 6.0


class HardSwish(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.hardSigmoid = HardSigmoid()

    def call(self, input):
        return input * self.hardSigmoid(input)


class ConvBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: int,
        stride: int,
        bias: bool,
        bn: bool,
        nl: str
    ):
        super().__init__()
        self.pad = tf.keras.layers.ZeroPadding2D(
            padding=(
                ((kernel_size-1)//2, kernel_size//2),
                ((kernel_size-1)//2, kernel_size//2)
            )
        ) if kernel_size > 1 else Identity()
        self.conv = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size, strides=stride,
            kernel_regularizer=tf.keras.regularizers.l2(1e-5),
            use_bias=bias
        )
        self.norm = tf.keras.layers.BatchNormalization(momentum=0.99) if bn else Identity()
        _available_act = {
            "relu": tf.keras.layers.ReLU(),
            "hswish": HardSwish(),
            "hsigmoid": HardSigmoid(),
            "softmax": tf.keras.layers.Softmax()
        }
        self.act = _available_act[nl] if nl else Identity()

    def call(self, input):
        x = self.pad(input)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class SqueezeExcite(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.pool = tf.keras.layers.AveragePooling2D(
            pool_size=tuple(map(int, input_shape[1:3]))
        )
        self.conv1 = ConvBlock(
            filters=int(input_shape[3]) // 4, kernel_size=1, stride=1,
            bias=False, bn=False, nl="relu"
        )
        self.conv2 = ConvBlock(
            filters=int(input_shape[3]), kernel_size=1, stride=1,
            bias=False, bn=False, nl="hsigmoid"
        )
        super().build(input_shape)

    def call(self, input):
        x = self.pool(input)
        x = self.conv1(x)
        x = self.conv2(x)
        return input * x


class Bneck(tf.keras.layers.Layer):
    def __init__(
        self,
        kernel_size: int,
        exp_channel: int,
        out_channel: int,
        se: bool,
        nl: str,
        stride: int
    ):
        self.stride = stride
        self.out_channel = out_channel
        super().__init__()
        self.expand = ConvBlock(
            filters=exp_channel, kernel_size=1, stride=1,
            bias=False, bn=True, nl=nl
        )
        self.dwise = tf.keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size, strides=stride,
            padding="same", use_bias=False,
            depthwise_regularizer=tf.keras.regularizers.l2(1e-5),
        )
        self.norm = tf.keras.layers.BatchNormalization(momentum=0.999)
        self.se = SqueezeExcite() if se else Identity()
        _available_act = {
            "relu": tf.keras.layers.ReLU(),
            "hswish": HardSwish()
        }
        self.act = _available_act[nl] if nl else Identity()
        self.project = ConvBlock(
            filters=out_channel, kernel_size=1, stride=1,
            bias=False, bn=True, nl=None
        )

    def build(self, input_shape):
        self.in_channel = int(input_shape[3])
        self.connect = self.stride == 1 and self.in_channel == self.out_channel
        super().build(input_shape)

    def call(self, input):
        x = self.expand(input)
        x = self.dwise(x)
        x = self.norm(x)
        x = self.se(x)
        x = self.act(x)
        x = self.project(x)
        return input + x if self.connect else x


def MobileNet_small(x):
    x = ConvBlock(filters=16, kernel_size=3, stride=2, bias=False, bn=True, nl="hswish")(x)
    x = Bneck(kernel_size=3, exp_channel=16,  out_channel=16, se=True,  nl="relu",   stride=2)(x)
    x = Bneck(kernel_size=3, exp_channel=72,  out_channel=24, se=False, nl="relu",   stride=2)(x)
    x = Bneck(kernel_size=3, exp_channel=88,  out_channel=24, se=False, nl="relu",   stride=1)(x)
    scale2 = x
    x = Bneck(kernel_size=5, exp_channel=96,  out_channel=40, se=True,  nl="hswish", stride=2)(x)
    x = Bneck(kernel_size=5, exp_channel=240, out_channel=40, se=True,  nl="hswish", stride=1)(x)
    x = Bneck(kernel_size=5, exp_channel=240, out_channel=40, se=True,  nl="hswish", stride=1)(x)
    x = Bneck(kernel_size=5, exp_channel=120, out_channel=48, se=True,  nl="hswish", stride=1)(x)
    x = Bneck(kernel_size=5, exp_channel=144, out_channel=48, se=True,  nl="hswish", stride=1)(x)
    scale1 = x
    x = Bneck(kernel_size=5, exp_channel=288, out_channel=96, se=True,  nl="hswish", stride=2)(x)
    x = Bneck(kernel_size=5, exp_channel=576, out_channel=96, se=True,  nl="hswish", stride=1)(x)
    x = Bneck(kernel_size=5, exp_channel=576, out_channel=96, se=True,  nl="hswish", stride=1)(x)
    scale0 = x
    return scale0, scale1, scale2


def YoloFPN(x):
    if isinstance(x, tuple):
        x0, x_skip = x
        x = Bneck(kernel_size=3, exp_channel=288, out_channel=48, se=False,  nl="hswish", stride=1)(x0)
        x = tf.keras.layers.UpSampling2D(size=2)(x)
        x = tf.keras.layers.Concatenate()([x, x_skip])

    x = Bneck(kernel_size=3, exp_channel=288, out_channel=96, se=False,  nl="hswish", stride=1)(x)
    x = Bneck(kernel_size=3, exp_channel=288, out_channel=96, se=False,  nl="hswish", stride=1)(x)
    return x


def YoloOutput(x, anc_per_scale, num_classes):
    x = ConvBlock(filters=288, kernel_size=3, stride=1, bias=False, bn=True, nl="hswish")(x)
    x = ConvBlock(
        filters=anc_per_scale * (num_classes + 5),
        kernel_size=1, stride=1, bias=True, bn=False, nl=None
    )(x)
    x = tf.keras.layers.Lambda(
        lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anc_per_scale, num_classes + 5))
    )(x)
    return x


def yolo_boxes(pred, anchors_org):
    # pred: (batch_size, gridy, gridx, anchors, (tx, ty, tw, th, conf, ...classes))

    y_pred_xywh_t, y_pred_conf, y_pred_clss = tf.split(pred, (4, 1, CFG.num_classes), axis=-1)
    y_pred_conf = tf.sigmoid(y_pred_conf)
    y_pred_clss = tf.sigmoid(y_pred_clss)

    fm_size = tf.cast(tf.shape(pred)[1:3], tf.float32)
    grid_cell = utils.make_grid_cell(fm_size)
    anchor_fm = tf.cast(anchors_org, tf.float32) / CFG.input_shape[::-1] * fm_size[::-1]
    y_pred_xywh_fm = utils.t2xywh_fm(y_pred_xywh_t, grid_cell, anchor_fm)
    y_pred_xywh_org = utils.fm2org(y_pred_xywh_fm, CFG.input_shape, fm_size)
    y_pred_yxyx_org = utils.xywh2yxyx(y_pred_xywh_org)

    return y_pred_yxyx_org, y_pred_conf, y_pred_clss


def yolo_nms(outputs):
    # boxes, conf, type
    box, cof, cla = [], [], []

    for o in outputs:
        box.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        cof.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        cla.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(box, axis=1)
    conf = tf.concat(cof, axis=1)
    clss = tf.concat(cla, axis=1)

    scores = conf * clss
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=bbox[:, :, tf.newaxis, :], scores=scores,
        max_output_size_per_class=tf.cast(CFG.max_boxes, tf.int32),
        max_total_size=CFG.max_boxes,
        iou_threshold=CFG.iou_thresh,
        score_threshold=CFG.nms_thresh,
        clip_boxes=False
    )

    return boxes, scores, classes, valid_detections


def MobileYolo_small(x, training=True):
    scale0, scale1, scale2 = MobileNet_small(x)

    x = YoloFPN(scale0)
    output_0 = YoloOutput(x, len(CFG.anchors[0]), CFG.num_classes)

    x = YoloFPN((x, scale1))
    output_1 = YoloOutput(x, len(CFG.anchors[1]), CFG.num_classes)

    x = YoloFPN((x, scale2))
    output_2 = YoloOutput(x, len(CFG.anchors[2]), CFG.num_classes)

    if training:
        return output_0, output_1, output_2

    boxes_0 = tf.keras.layers.Lambda(lambda x: yolo_boxes(x, CFG.anchors[0]), name='yolo_boxes_0')(output_0)
    boxes_1 = tf.keras.layers.Lambda(lambda x: yolo_boxes(x, CFG.anchors[1]), name='yolo_boxes_1')(output_1)
    boxes_2 = tf.keras.layers.Lambda(lambda x: yolo_boxes(x, CFG.anchors[2]), name='yolo_boxes_2')(output_2)

    outputs = tf.keras.layers.Lambda(lambda x: yolo_nms(x), name='yolo_nms')((boxes_0, boxes_1, boxes_2))

    return outputs


def YoloLoss(fm_size, anchors_org):
    def yolo_loss(y_true_xywh_reg, pred):
        # 0. prepare
        grid_cell = utils.make_grid_cell(fm_size)
        anchor_fm = anchors_org / CFG.input_shape[::-1] * fm_size[::-1]

        # 1. transform pred
        y_pred_xy_t, y_pred_wh_t, y_pred_conf, y_pred_clss = tf.split(pred, (2, 2, 1, CFG.num_classes), axis=-1)
        y_pred_xywh_t = tf.concat([tf.sigmoid(y_pred_xy_t), y_pred_wh_t], axis=-1)
        y_pred_conf = tf.sigmoid(y_pred_conf)
        y_pred_clss = tf.sigmoid(y_pred_clss)
        y_pred_xywh_fm = utils.t2xywh_fm(pred[..., 0:4], grid_cell, anchor_fm)

        # 2. transform true
        y_true_conf = y_true_xywh_reg[..., 4:5]
        y_true_clss = y_true_xywh_reg[..., 5:]
        y_true_xywh_fm = utils.reg2fm(y_true_xywh_reg[..., 0:4], fm_size)
        y_true_xywh_t = utils.xywh_fm2t(y_true_xywh_fm, grid_cell, anchor_fm, y_true_conf)[..., 0:4]

        # 3. calculate mask
        conf_mask = tf.squeeze(y_true_conf, axis=-1)
        best_iou = tf.map_fn(
            lambda x: tf.math.reduce_max(utils.iou(
                x[0][:, :, :, tf.newaxis, :],
                tf.boolean_mask(x[1], tf.cast(x[2], tf.bool))[tf.newaxis, tf.newaxis, tf.newaxis, :, :]), axis=-1),
            (y_pred_xywh_fm, y_true_xywh_fm, conf_mask),
            tf.float32)
        noobj_mask = (1.0 - conf_mask) * tf.cast(best_iou < CFG.ignore_thresh, tf.float32)

        # 4. calculate loss_xywh
        box_loss_scale = 2.0 - y_true_xywh_reg[..., 2]*y_true_xywh_reg[..., 3]
        loss_xywh = conf_mask * box_loss_scale * tf.math.reduce_sum(
            tf.math.square(y_true_xywh_t-y_pred_xywh_t), axis=-1
        )

        # 5. calculate loss_conf
        loss_conf = tf.losses.binary_crossentropy(y_true_conf, y_pred_conf)
        loss_conf = conf_mask * loss_conf + noobj_mask * loss_conf

        # 6. calculate loss_clss
        loss_clss = conf_mask * tf.losses.binary_crossentropy(y_true_clss, y_pred_clss)

        # 7. average over batch (batch, gridy, gridx, anchors) => (1)
        loss_total = tf.math.reduce_sum(loss_xywh + loss_conf + loss_clss) / CFG.batch_size

        return loss_total
    return yolo_loss
