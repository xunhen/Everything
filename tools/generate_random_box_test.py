import Faster_RCNN.Tools.generate_random_box as generate_random_box
import numpy as np
import tensorflow as tf
import lxml.etree as etree
from object_detection.utils import dataset_util
from Faster_RCNN.Tools.generate_random_box import random_rpn,random_test
import os
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageDraw as ImageDraw


def extract_data(image_path):
    with tf.gfile.GFile(image_path + '.xml', 'r') as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
    width = int(data['size']['width'])
    height = int(data['size']['height'])
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    if 'object' in data:
        for obj in data['object']:
            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
    gt = np.stack([ymin, xmin, ymax, xmax], axis=1)
    rpn = random_test(gt)
    rpn = np.stack(rpn, axis=1)
    return rpn, gt, height, width


def explame_test(image_path):
    rpn, gt, height, width = extract_data(image_path)

    img = Image.open(image_path + '.jpg')
    for i in rpn:
        plt.figure(num='rpn is {}'.format(i))
        img_=img.copy()
        draw = ImageDraw.Draw(img_)
        rect = [(i[1] * width, i[0] * height), (i[3] * width), i[2] * height]
        draw.rectangle(rect, outline='blue')
        for j in gt:
            rect = [(j[1] * width, j[0] * height), (j[3] * width), j[2] * height]
            print(j)
            draw.rectangle(rect, outline='red')
        plt.imshow(img_)

    pass


if __name__ == '__main__':
    PATH_TO_TEST_IMAGES_DIR = 'F:\PostGraduate\DataSet\MOTFromWinDataSet'
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, '00000{}'.format(i)) for i in range(1, 2)]
    for image in TEST_IMAGE_PATHS:
        explame_test(image)
    plt.show()
