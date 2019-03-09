# coding: utf-8
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_pascal_tf_record.py \
        --data_dir=/home/user/VOCdevkit \
        --year=VOC2012 \
        --output_path=/home/user/pascal.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

from Tool.generate_random_box import random_rpn
import numpy as np

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                                    'merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('year', 'VOC2007', 'Desired challenge year.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                                                          'difficult instances')
FLAGS = flags.FLAGS

FLAGS.label_map_path = 'E:\\CODE\\Python\\TensorFlowTest\\Faster_RCNN\\DataSet\\mot_label_map.pbtxt'
FLAGS.data_dir = 'F:\\PostGraduate\\DataSet\\Others\\CarDataSetForHe\\CarDataSetForHe'
FLAGS.output_path = 'E:\\CODE\\Python\\TensorFlowTest\\Faster_RCNN\\DataSet\\TFrecordMultiBatch\\car_train.record'

SETS = ['train', 'val', 'trainval', 'test']
YEARS = ['VOC2007', 'VOC2012', 'merged']


def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       filename,
                       ignore_difficult_instances=False):
    full_path = os.path.join(dataset_directory, filename)
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    for object in data:
        width = object['width']
        height = object['height']
        xmin.append(object['xmin'])
        ymin.append(object['ymin'])
        xmax.append(object['xmax'])
        ymax.append(object['ymax'])
        classes.append(object['id'])

    rpn = random_rpn(zip(ymin, xmin, ymax, xmax), need_same=True)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            filename.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),

        'image/rpn/bbox/xmin': dataset_util.float_list_feature(rpn[1]),
        'image/rpn/bbox/xmax': dataset_util.float_list_feature(rpn[3]),
        'image/rpn/bbox/ymin': dataset_util.float_list_feature(rpn[0]),
        'image/rpn/bbox/ymax': dataset_util.float_list_feature(rpn[2]),

        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return example


num = np.zeros(shape=[10])


def parse_txt(data_):
    datas = []
    flag=False
    for i in data_.split('\n'):
        temp = i.split()
        if len(temp) < 7:
            continue
        flag=True
        data = dict()
        data['id'] = int(temp[0])+1
        num[data['id']] += 1
        data['width'] = int(temp[1])
        data['height'] = int(temp[2])
        x, y, w, h = int(temp[3]), int(temp[4]), int(temp[5]), int(temp[6])
        data['xmin'] = x / data['width']
        data['ymin'] = y / data['height']
        data['xmax'] = (x + w) / data['width']
        data['ymax'] = (y + h) / data['height']
        #print(data)
        datas.append(data)
    return datas,flag
    pass


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    logging.info('Reading from PASCAL %s dataset.', FLAGS.data_dir)
    examples_path = os.path.join(FLAGS.data_dir, 'train.txt')
    annotations_dir = FLAGS.data_dir
    examples_list = dataset_util.read_examples_list(examples_path)
    for idx, example in enumerate(examples_list):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(examples_list))
        path = os.path.join(annotations_dir, example + 'org.txt')
        with tf.gfile.GFile(path, 'r') as fid:
            txt = fid.read()

        data,flag = parse_txt(txt)
        if not flag:
            continue
        tf_example = dict_to_tf_example(data, FLAGS.data_dir, label_map_dict, example + '.jpg',
                                        FLAGS.ignore_difficult_instances)
        writer.write(tf_example.SerializeToString())
    writer.close()

    print(num)


if __name__ == '__main__':
    tf.app.run()
