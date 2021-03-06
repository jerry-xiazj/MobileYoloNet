# Author: Jerry Xia <jerry_xiazj@outlook.com>

import os
import cv2
import random
import numpy as np
import tensorflow as tf
import core.utils as utils
from config import CFG


class Dataset:

    def __init__(self, file_path, batch_size, batch_per_epoch, train=True):
        self.ann_lines, self.num_samples = self._load_annotations(file_path)
        self.batch_count = 0
        self.batch_size = batch_size
        self.batch_per_epoch = batch_per_epoch
        self.train = train

    def __iter__(self):
        return self

    def __len__(self):
        return self.batch_per_epoch

    def __next__(self):
        with tf.device('/cpu:0'):
            if self.batch_count < CFG.batch_per_epoch:
                self.batch_count += 1
                return self.batch_count, self.generate_batch()
            else:
                self.batch_count = 0
                raise StopIteration

    def _load_annotations(self, file_path):
        with open(file_path, 'r') as rf:
            ann_lines = rf.readlines()
        ann_lines = [ann.rstrip('\n') for ann in ann_lines]
        return ann_lines, len(ann_lines)

    def _random_horizontal_flip(self, image, bboxes):
        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]
        return image, bboxes

    def _random_crop(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate(
                [np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(
                0, max_bbox[0] - int(random.uniform(0, max_l_trans)))
            crop_ymin = max(
                0, max_bbox[1] - int(random.uniform(0, max_u_trans)))
            crop_xmax = max(
                w, max_bbox[2] + int(random.uniform(0, max_r_trans)))
            crop_ymax = max(
                h, max_bbox[3] + int(random.uniform(0, max_d_trans)))

            image = image[crop_ymin: crop_ymax, crop_xmin: crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
        return image, bboxes

    def _random_translate(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate(
                [np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = np.around(random.uniform(-(max_l_trans - 1), (max_r_trans - 1)))
            ty = np.around(random.uniform(-(max_u_trans - 1), (max_d_trans - 1)))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty
        return image, bboxes

    def _parse_annotation(self, ann_line):
        """Parse train_file.
        ann_line: one line of train_file
        """
        ann_line = ann_line.split(' ')
        if not os.path.exists(ann_line[0]):
            raise KeyError("%s does not exist ... " % ann_line[0])
        image = cv2.cvtColor(cv2.imread(ann_line[0]), cv2.COLOR_BGR2RGB)

        if not self.train:
            return utils.resize_img(image, CFG.input_shape)

        bboxes = np.array([list(map(int, box.split(','))) for box in ann_line[1:]])  # int64

        if CFG.data_aug:
            image, bboxes = self._random_horizontal_flip(image, bboxes)
            image, bboxes = self._random_crop(image, bboxes)
            image, bboxes = self._random_translate(image, bboxes)

        return utils.resize_img(image, CFG.input_shape, bboxes)

    def _make_y_true(self, box_data):
        """Convert box_data to y_true.
        box_data: ndarray, shape=(num_boxes, 5)
            The 2nd dimension includes (x_min, y_min, x_max, y_max, class_id) without regularization.
        """

        y_true = [np.zeros(
            (CFG.fm_size[i, 0], CFG.fm_size[i, 1], CFG.anc_per_scale, 5+CFG.num_classes), dtype=np.float32)
            for i in range(CFG.num_scales)]

        anc_xywh_reg = utils.org2reg(
            np.concatenate([CFG.anchors[..., 0:2]*0.0, CFG.anchors[..., 0:2]], axis=-1), CFG.input_shape, use_np=True)
        box_xywh_reg = utils.org2reg(utils.xyxy2xywh(box_data, use_np=True), CFG.input_shape, use_np=True)

        if box_xywh_reg.shape[0] == 0:
            return y_true

        iou_list = np.zeros((box_xywh_reg.shape[0], CFG.anchors.shape[0]*CFG.anchors.shape[1]))
        for s in range(CFG.num_scales):
            box_fm = utils.reg2fm(box_xywh_reg, CFG.fm_size[s], use_np=True)
            anc_fm = utils.reg2fm(anc_xywh_reg[s], CFG.fm_size[s], use_np=True)
            box_fm[..., 0:2] = box_fm[..., 0:2] - np.floor(box_fm[..., 0:2]) - 0.5
            iou_fm = utils.iou(box_fm[:, np.newaxis, :], anc_fm[np.newaxis, :, :], use_np=True)
            iou_list[:, s*CFG.anc_per_scale:(s+1)*CFG.anc_per_scale] = iou_fm
        best_anc = np.argmax(iou_list, axis=-1)
        for i_box, i_anc in enumerate(best_anc):
            idx_scl = i_anc // CFG.anc_per_scale
            idx_anc = i_anc % CFG.anc_per_scale
            box_fm = utils.reg2fm(box_xywh_reg, CFG.fm_size[idx_scl], use_np=True)
            idx_y = int(np.floor(box_fm[i_box, 1]))
            idx_x = int(np.floor(box_fm[i_box, 0]))
            idx_cls = int(box_fm[i_box, 4])
            y_true[idx_scl][idx_y, idx_x, idx_anc, 0:4] = box_xywh_reg[i_box, 0:4]
            y_true[idx_scl][idx_y, idx_x, idx_anc, 4] = 1
            y_true[idx_scl][idx_y, idx_x, idx_anc, 5+idx_cls] = 1
        return y_true

    def generate_batch(self):
        batch_image = np.zeros(
            (CFG.batch_size, CFG.input_shape[0], CFG.input_shape[1], 3), dtype=np.float32)
        if self.train:
            batch_boxes = [np.zeros(
                (CFG.batch_size, CFG.fm_size[i, 0], CFG.fm_size[i, 1], CFG.anc_per_scale, 5+CFG.num_classes),
                dtype=np.float32)
                for i in range(CFG.num_scales)]
        num = 0
        while num < CFG.batch_size:
            index = self.batch_count * CFG.batch_size + num
            if index >= self.num_samples:
                index %= self.num_samples
                np.random.shuffle(self.ann_lines)
            ann_line = self.ann_lines[index]
            res = self._parse_annotation(ann_line)
            if self.train:
                image, boxes = res
                boxes = self._make_y_true(boxes)
                for s in range(CFG.num_scales):
                    batch_boxes[s][num, ...] = boxes[s]
            else:
                image = res
            batch_image[num, :, :, :] = image
            num += 1
        if self.train:
            return batch_image, batch_boxes
        else:
            return batch_image

    def generate_origin(self):
        for line in self.ann_lines:
            ann_line = line.split(' ')
            img = cv2.imread(ann_line[0])
            if self.train:
                box = np.array([list(map(int, box.split(','))) for box in ann_line[1:]])
                yield img, box
            else:
                yield img
