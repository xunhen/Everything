import numpy as np
import os

import tensorflow as tf
import zipfile
import lxml.etree as etree
from object_detection.utils import dataset_util

from distutils.version import StrictVersion
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from PIL import Image
import tensorflow.contrib.slim as slim
from object_detection.legacy import trainer
from object_detection.builders import model_builder

from object_detection.utils import config_util
from object_detection.utils import ops as utils_ops

from tools.generate_random_box import random_rpn

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')
from object_detection.utils import label_map_util

from config import cfg

import time
from tools.create_txt_tf_record import parse_txt

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(cfg.DATASET_PATH, 'mscoco_label_map.pbtxt')
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'F:\PostGraduate\DataSet\CarDataSetOfHe\cyc'


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def get_boxes(filename, format, random_jittle=cfg.RANDOM_JITTLE):
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    if format == 'txt':
        with tf.gfile.GFile(filename + 'org.txt', 'r') as fid:
            txt = fid.read()
        datas, flag = parse_txt(txt)
        for data in datas:
            xmin.append(data['xmin'])
            ymin.append(data['ymin'])
            xmax.append(data['xmax'])
            ymax.append(data['ymax'])
    else:
        with tf.gfile.GFile(filename + '.xml', 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        width = int(data['size']['width'])
        height = int(data['size']['height'])
        if 'object' in data:
            for obj in data['object']:
                xmin.append(float(obj['bndbox']['xmin']) / width)
                ymin.append(float(obj['bndbox']['ymin']) / height)
                xmax.append(float(obj['bndbox']['xmax']) / width)
                ymax.append(float(obj['bndbox']['ymax']) / height)
    if random_jittle:
        boxes = random_rpn(zip(ymin, xmin, ymax, xmax))
        boxes = [[ymin, xmin, ymax, xmax] for (ymin, xmin, ymax, xmax) in zip(*boxes)]
    else:
        boxes = list(zip(ymin, xmin, ymax, xmax))
        boxes = np.array(boxes)
        boxes = boxes.reshape([1, -1, 4])
    return boxes


def analyze(test_image_list, model_which='resnet50', has_rpn=True, using_filter_fn=False,
            first_stage_max_proposals=None, image_fix=True,
            image_divide=False):
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, i) for i in test_image_list]
    # pipeline_config_path = 'E:\\CODE\\Python\\TensorFlowTest\\Faster_RCNN\\Model\\pipeline.config'
    # pipeline_config_path = 'E:\\CODE\\Python\\TensorFlowTest\\Faster_RCNN\\Log_Temp\\pipeline.config'
    pipeline_config_path = cfg.TRAINED_MODEL[model_which]['config']
    if image_fix:
        pipeline_config_path = pipeline_config_path.split('.')[0] + '_fix.config'
    if image_divide:
        pipeline_config_path = pipeline_config_path.split('.')[0] + '_2.config'
    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    model_config = configs['model']
    if first_stage_max_proposals:
        model_config.faster_rcnn.first_stage_max_proposals = first_stage_max_proposals

    graph = tf.Graph()
    batch_size = 1
    with graph.as_default():
        cfg.HAS_RPN = has_rpn
        model = model_builder.build(model_config=model_config, is_training=False, using_filter_fn=using_filter_fn)
        with tf.variable_scope('placeholder'):
            image_tensor = tf.placeholder(tf.float32, shape=(batch_size, None, None, 3), name='images')
            if not has_rpn:
                rpn_box = tf.placeholder(tf.float32, shape=(batch_size, None, 4), name='rpn_box')
                rpn_class = tf.ones_like(rpn_box, dtype=tf.int32)
                rpn_class = rpn_class[:, :, :2]
                model.provide_rpn_box(rpn_box, rpn_class)
            if using_filter_fn:
                filter_box = [tf.placeholder(tf.float32, shape=(None, 4), name='filter_box{}'.format(i)) for i in
                              range(batch_size)]
                model.provide_filter_box(filter_box)
        preprocessed_inputs, true_image_shapes = model.preprocess(image_tensor)
        prediction_dict = model.predict(preprocessed_inputs, true_image_shapes)
        output_dict = model.postprocess(prediction_dict, true_image_shapes)

        saver = tf.train.Saver()
        save_path = cfg.TRAINED_MODEL[model_which]['model']

        time_summary = []
        with tf.Session() as sess:
            # if not os.path.exists(save_path):
            #     print('{} does not exist'.format(save_path))
            saver.restore(sess, save_path)
            for _, image_path in enumerate(TEST_IMAGE_PATHS):
                image = Image.open(image_path + '.jpg')
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                feed_dict = {image_tensor: image_np_expanded}
                filter_boxes_ = get_boxes(image_path, format='txt')
                if not has_rpn:
                    feed_dict[rpn_box] = filter_boxes_
                if using_filter_fn:
                    for index, box in enumerate(filter_box):
                        feed_dict[box] = filter_boxes_[index]

                _, _ = sess.run([prediction_dict, output_dict], feed_dict=feed_dict)

                first_stage_list = ['rpn_box_predictor_features', 'rpn_features_to_crop', 'image_shape',
                                    'rpn_box_encodings', 'rpn_objectness_predictions_with_background', 'anchors']
                prediction_dict_first_feed = [prediction_dict[i] for i in first_stage_list]
                start = time.time()
                _ = sess.run([prediction_dict_first_feed, ], feed_dict=feed_dict)
                mid = time.time()
                prediction_dict_ = sess.run(prediction_dict, feed_dict=feed_dict)
                end = time.time()
                total = end - mid
                first = mid - start
                time_summary.append([total, first, total - first, prediction_dict_['num_proposals']])
        print(time_summary)
    return time_summary


def draw_doubleY(x, y1, y2, title, style, label_name, ylabel):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel(ylabel[0])
    ax1.set_title(title)

    ax2 = ax1.twinx()
    ax2.set_ylabel(ylabel[1])
    ax2.set_xlabel('number of proposals')
    type = ['-', '--']
    for x_, y1_, y2_, s_, n_ in zip(x, y1, y2, style, label_name):
        ax1.plot(x_, y1_, s_ + type[0], label=n_)
        ax2.plot(x_, y2_, s_ + type[1], label=n_)
    plt.legend()


def draw(x, y, title, label_name, style):
    plt.title(title)
    plt.xlabel('number of proposals')
    plt.ylabel('ms')
    for x_, y_, label_, style_ in zip(x, y, label_name, style):
        plt.plot(x_, y_, style_, label=label_)

    plt.legend()


def analyze_time_faster(number_of_proposals, models, test_image, save_path='analyze_result', image_fixed_size=True,
                        image_divide_size=False,
                        has_rpn=True, using_filter_fn=False, suffix=''):
    if image_fixed_size:
        suffix += '_fixed'
    if image_divide_size:
        suffix += '_2'
    if not has_rpn:
        suffix += '_noRPN'
    if using_filter_fn:
        suffix += '_filter'
    for model_item in models:
        time_summary = []
        if suffix:
            filename = model_item + '_' + suffix + '.npy'
        else:
            filename = model_item + '.npy'
        save_file_name = os.path.join(save_path, filename)
        if not os.path.exists(save_file_name):
            for i in number_of_proposals:
                time_summary.append(analyze(test_image, model_which=model_item, first_stage_max_proposals=i,
                                            image_fix=image_fixed_size, image_divide=image_divide_size, has_rpn=has_rpn,
                                            using_filter_fn=using_filter_fn))
            time_summary = np.array(time_summary)
            # 第一个数值预热
            time_summary = time_summary[1:, 1:]
            np.save(save_file_name, time_summary)
        else:
            time_summary = np.load(save_file_name)
        summary = np.average(time_summary, axis=1)
        print('analyze time of {}:'.format(filename.split('.')[0]))
        print('the number of proposals   total          first-stage-time       rcnn-time')
        for summary_ in summary:
            print(
                '          {},           {:.3f}ms      {:.3f}ms {:.3f}%     {:.3f}ms {:.3f}%'.format(summary_[-1],
                                                                                                     summary_[
                                                                                                         0] * 1000,
                                                                                                     summary_[
                                                                                                         1] * 1000,
                                                                                                     summary_[1] /
                                                                                                     summary_[0] * 100,
                                                                                                     summary_[
                                                                                                         2] * 1000,
                                                                                                     summary_[2] /
                                                                                                     summary_[
                                                                                                         0] * 100))
            # print('the number of proposals: {}\n --first-stage:{:.3f} {:.3f}%\n rcnn time: {:.3f} {:.3f}% '.
            #       format(summary_[-1], summary_[1], summary_[1] / summary_[0] * 100, summary_[2],
            #              summary_[2] / summary_[0] * 100))
        temp = summary[1:] - summary[:-1]

        print('{}ms per proposals'.format(np.average(temp[:, 2] / temp[:, -1]) * 1000))
    times = []
    for model_item in models:
        if suffix:
            filename = model_item + '_' + suffix + '.npy'
        else:
            filename = model_item + '.npy'
        save_file_name = os.path.join(save_path, filename)
        times.append(np.load(save_file_name))
    times = np.array(times)
    times = np.average(times, axis=2)

    x = times[..., -1]
    y1 = times[..., 2] * 1000
    y2 = times[..., 2] / times[..., 0] * 100
    style = np.array(['r', 'b'])
    name = ['resnet50', 'inception_v2']
    ylabel = ['ms for solid line', '% for dashed line']
    draw_doubleY(x, y1, y2,
                 'analyze_rcnn_time' + suffix, style,
                 name, ylabel)
    plt.show()

    y2 = y1
    y1 = times[..., 0] * 1000
    y = [y1[0], y2[0], y1[1], y2[1]]
    style = ['r-', 'r--', 'b-', 'b--']
    label_name = ['total for resnet50', 'rcnn for resnet50', 'total for inception_v2', 'rcnn for inception_v2']
    draw(np.tile(x, (2, 1)), y,
         'analyze_time' + '_fixed' if image_fixed_size else '' + '_2' if image_divide_size else '', label_name, style)
    plt.show()


def analyze_improve_percent():
    number_of_proposals = np.array([10, 50, 100, 200, 300, 400, 500, 600])
    models = ['resnet50', 'inception_v2']
    test_image = ['car6228', 'car6229']
    divide_options = [True, False]
    path = 'analyze_result'
    for image_item in test_image:
        for model_item in models:
            suffix = model_item + '_' + image_item + '_fixed'
            for divide_flag in divide_options:
                filename=suffix
                if divide_flag:
                    filename += '_2'
                filename_filter = filename + '_filter.npy'
                filename += '.npy'
                summary = np.load(os.path.join(path, filename))
                summary = np.average(summary, axis=1)
                summary_filter = np.load(os.path.join(path, filename_filter))
                summary_filter = np.average(summary_filter, axis=1)
                result = np.zeros([np.shape(summary)[0], 5])
                result[..., 0] = summary[..., 0] * 1000  # total time
                result[..., 1] = summary_filter[..., 0] * 1000  # total time after filter
                result[..., 2] = summary[..., -1] - summary_filter[..., -1]  # the number be filtered
                result[..., 3] = result[..., 0] - result[..., 1]  # the time improved
                result[..., 4] = result[..., 3] / result[..., 0]*100  # the percent improved

                print('------------------{}------------------'.format(filename))
                print(
                    'number of proposal | the number be filtered | total time(ms) | total time after filter | improved(ms/%)')
                for index, number in enumerate(number_of_proposals):
                    print('{0:^10d} {1:^10d} {2[0]:^2f}ms   {2[1]:^2f}ms   {2[3]:.2f}ms/{2[4]:.2f}%'.
                          format(number,int(result[index][2]),result[index]))


def execuct_analyze_and_draw():
    # 第一个100预热
    number_of_proposals = np.array([100, 10, 50, 100, 200, 300, 400, 500, 600])
    models = ['resnet50', 'inception_v2']
    test_image = ['car6228', 'car6228', 'car6228', 'car6228', 'car6228']
    suffix = 'car6228'
    analyze_time_faster(number_of_proposals, models, test_image, image_fixed_size=True, image_divide_size=False,
                        has_rpn=True, using_filter_fn=True, suffix=suffix)
    analyze_time_faster(number_of_proposals, models, test_image, image_fixed_size=True, image_divide_size=True,
                        has_rpn=True, using_filter_fn=True, suffix=suffix)
    analyze_time_faster(number_of_proposals, models, test_image, image_fixed_size=True, image_divide_size=False,
                        has_rpn=True, using_filter_fn=False, suffix=suffix)
    analyze_time_faster(number_of_proposals, models, test_image, image_fixed_size=True, image_divide_size=True,
                        has_rpn=True, using_filter_fn=False, suffix=suffix)

    test_image = ['car6229', 'car6229', 'car6229', 'car6229', 'car6229']
    suffix = 'car6229'
    analyze_time_faster(number_of_proposals, models, test_image, image_fixed_size=True, image_divide_size=False,
                        has_rpn=True, using_filter_fn=True, suffix=suffix)
    analyze_time_faster(number_of_proposals, models, test_image, image_fixed_size=True, image_divide_size=True,
                        has_rpn=True, using_filter_fn=True, suffix=suffix)
    analyze_time_faster(number_of_proposals, models, test_image, image_fixed_size=True, image_divide_size=False,
                        has_rpn=True, using_filter_fn=False, suffix=suffix)
    analyze_time_faster(number_of_proposals, models, test_image, image_fixed_size=True, image_divide_size=True,
                        has_rpn=True, using_filter_fn=False, suffix=suffix)


if __name__ == '__main__':
    analyze_improve_percent()
