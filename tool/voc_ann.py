# Author: Jerry Xia <jerry_xiazj@outlook.com>

import os
import xml.etree.ElementTree as et

ann_dir = "/home/jerry/data/VOCdevkit/VOC2012/Annotations/"
img_dir = "/home/jerry/data/VOCdevkit/VOC2012/JPEGImages/"
train_file = "/home/jerry/MobileYoloNet/data/train_file"
classes = ["person", "bird", "cat", "cow", "dog", "horse",
           "sheep", "aeroplane", "bicycle", "boat", "bus",
           "car", "motorbike", "train", "bottle", "chair",
           "diningtable", "pottedplant", "sofa", "tvmonitor"]


def voc_annotation(num='all'):
    """Convert VOC Pascal annotation file to standard format."""

    file_list = os.listdir(ann_dir)
    if isinstance(num, int):
        file_num = len(file_list)
        file_list = file_list[:min(file_num, num)]
    elif num == 'all':
        pass
    else:
        raise NotImplementedError('Only int and "all" are invalid argument.')

    with open(train_file, 'w') as wf:
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
    return len(file_list)


if __name__ == "__main__":
    num_sample = voc_annotation(100)
    print("Number of samples is {}".format(num_sample))
