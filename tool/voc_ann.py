# Author: Jerry Xia <jerry_xiazj@outlook.com>

import os
import numpy as np
import xml.etree.ElementTree as et


train_file = "/home/jerry/MobileYoloNet/data/train_file"
val_file = "/home/jerry/MobileYoloNet/data/val_file"

ann_dir = "/home/jerry/data/VOCdevkit/VOC2012/Annotations/"
img_dir = "/home/jerry/data/VOCdevkit/VOC2012/JPEGImages/"
classes = ["person", "bird", "cat", "cow", "dog", "horse",
           "sheep", "aeroplane", "bicycle", "boat", "bus",
           "car", "motorbike", "train", "bottle", "chair",
           "diningtable", "pottedplant", "sofa", "tvmonitor"]


def voc_annotation_all(output):
    """Convert Pascal VOC annotation file to standard format."""

    file_list = os.listdir(ann_dir)
    np.random.shuffle(file_list)

    with open(output, 'w') as wf:
        for ann in file_list:
            root = et.parse(ann_dir + ann).getroot()
            img_name = root.find('filename').text
            wf.write(img_dir + img_name)
            for obj in root.findall('object'):
                xmin = str(obj.find('bndbox').find('xmin').text)
                ymin = str(obj.find('bndbox').find('ymin').text)
                xmax = str(obj.find('bndbox').find('xmax').text)
                ymax = str(obj.find('bndbox').find('ymax').text)
                class_index = str(classes.index(obj.find('name').text))
                wf.write(' '+xmin+','+ymin+','+xmax+','+ymax+','+class_index)
            wf.write('\n')
    print("Number of samples: {}".format(len(file_list)))
    return len(file_list)


def voc_annotation(output, num_sample, num_class=20):
    """Convert Pascal VOC annotation file to standard format."""
    if num_class > 20:
        raise Exception("Only 20 classes in Pascal VOC dataset.")
    file_list = os.listdir(ann_dir)
    np.random.shuffle(file_list)

    count_sample = 0
    count_class = np.zeros(num_class, dtype=np.int)
    max_class = num_sample // num_class

    with open(output, 'w') as wf:
        for ann in file_list:
            root = et.parse(ann_dir + ann).getroot()
            img_name = root.find('filename').text
            bool_class = False
            box_str = ''
            box_str = box_str + img_dir + img_name
            for obj in root.findall('object'):
                xmin = str(obj.find('bndbox').find('xmin').text)
                ymin = str(obj.find('bndbox').find('ymin').text)
                xmax = str(obj.find('bndbox').find('xmax').text)
                ymax = str(obj.find('bndbox').find('ymax').text)
                class_index = classes.index(obj.find('name').text)
                if class_index < num_class:
                    box_str = box_str + ' ' + xmin + ',' + ymin + ',' + xmax + ',' + ymax + ',' + str(class_index)
                    if (not bool_class) and (count_class[class_index] < max_class):
                        count_class[class_index] = count_class[class_index] + 1
                        bool_class = True
                        print('class: {:>2d}'.format(class_index), end=' count:')
                        print('{:>3d}'.format(count_class[class_index]))
            if bool_class:
                wf.write(box_str + '\n')
                count_sample = count_sample + 1
                if count_sample == num_sample:
                    break

    print("Number of samples: {:<5d}".format(count_sample))
    print("Number of samples in each category: {}".format(count_class))

    return count_sample, count_class


if __name__ == "__main__":
    voc_annotation(train_file, num_sample=120, num_class=2)
    voc_annotation(val_file, num_sample=6, num_class=2)
