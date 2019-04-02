import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from PIL import Image

from distutils.version import StrictVersion
from object_detection.builders import model_builder
from object_detection.utils import config_util
from tools.generate_box_vibe import Generate_Box_By_ViBe
import cv2
import time
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from config import cfg
from matplotlib import pyplot as plt

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')


class Detection(object):
    def __init__(self, pipeline_config_path, restore_path, filter_threshold=None, rpn_type='orginal_rpn',
                 max_number=20, debug=False):
        self._rpn_type = rpn_type
        self._filter_fn_arg = None
        if filter_threshold:
            self._filter_fn_arg = {'filter_threshold': filter_threshold}
        self._get_filter_boxes_fn = Generate_Box_By_ViBe()
        self._pipeline_config_path = pipeline_config_path
        self._restore_path = restore_path
        self._max_number = max_number
        self._replace_rpn_arg = None
        self._graph = None
        self._first_stage_max_proposals = None
        self._debug = debug
        self._time_per = []
        self._average_time = 0
        self._average_filter_bboxes = 0
        self._total = 0
        pass

    def init_some(self):
        pass

    def _process(self):
        pass

    def test(self):
        print(self._pipeline_config_path)
        print(self._restore_path)

    def build_model(self):
        print('------------build_model_begin------------')
        batch_size = 1  # detect one image once
        configs = config_util.get_configs_from_pipeline_file(self._pipeline_config_path)
        model_config = configs['model']
        model_config.faster_rcnn.first_stage_max_proposals = 300
        self._first_stage_max_proposals = model_config.faster_rcnn.first_stage_max_proposals
        print('first_stage_max_proposals:', model_config.faster_rcnn.first_stage_max_proposals)
        self._graph = tf.Graph()
        with self._graph.as_default():
            model = model_builder.build(model_config=model_config, is_training=False, rpn_type=self._rpn_type,
                                        filter_fn_arg=self._filter_fn_arg, replace_rpn_arg=self._replace_rpn_arg)

            with tf.variable_scope('placeholder'):
                self._image_tensor = tf.placeholder(tf.float32, shape=(batch_size, None, None, 3), name='images')
                if self._filter_fn_arg:
                    self._filter_box_list = [tf.placeholder(tf.float32, shape=(None, 4), name='filter_box{}'.format(i))
                                             for i in range(batch_size)]
                    model.provide_filter_box_list(self._filter_box_list)

            self._preprocessed_inputs, self._true_image_shapes = model.preprocess(self._image_tensor)
            self._prediction_dict = model.predict(self._preprocessed_inputs, self._true_image_shapes)
            self._output_dict = model.postprocess(self._prediction_dict, self._true_image_shapes)

            self._saver = tf.train.Saver()
            self._sess = tf.Session()
            self._saver = tf.train.Saver()
            self._saver.restore(sess=self._sess, save_path=self._restore_path)
            self._options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            self._run_metadata = tf.RunMetadata()

        print('------------build_model_end------------')

    def detection(self, image, gray_image=None, need_count=1):
        # when using filter ,the gray_image is necessary!!
        print('------------detection_begin------------')
        image_expanded = np.expand_dims(np.array(image), axis=0)
        # print(image_expanded.shape, image_expanded.dtype)
        feed_dict = {self._image_tensor: image_expanded}
        bboxes = None
        if self._filter_fn_arg:
            if gray_image is None:
                print('when using filter ,the gray_image is necessary!!')
            print('get filter bboxes begin!')
            bboxes = self._get_filter_boxes_fn.processAndgenerate(np.array(gray_image))
            print('get filter bboxes end!')
            if len(bboxes) == 0:
                return None
            print('filter_boxes: ', bboxes.shape)
            feed_dict[self._filter_box_list[0]] = bboxes  # only batch_size=1
        print('run begin!')

        tic = time.time()
        # output_dict_, prediction_dict_ = self._sess.run([self._output_dict, self._prediction_dict],
        #                                                 feed_dict=feed_dict,run_metadata=self._run_metadata,options=self._options)
        output_dict_, prediction_dict_ = self._sess.run([self._output_dict, self._prediction_dict],
                                                        feed_dict=feed_dict)
        toc = time.time()
        print('run end!')
        output_dict_['detection_classes'][0] += 1
        number = np.minimum(self._max_number, output_dict_['num_detections'])
        result = np.concatenate(
            (output_dict_['detection_boxes'][0][:int(number)],
             np.expand_dims(output_dict_['detection_scores'][0][:int(number)], axis=1),
             np.expand_dims(output_dict_['detection_classes'][0][:int(number)], axis=1)), axis=1)
        print('result: ', result.shape)
        print('filtered number: ', self._first_stage_max_proposals - prediction_dict_['num_proposals'][0])
        print('detection timer: {} ms'.format((toc - tic) * 1000))
        if need_count == 1:
            self._average_filter_bboxes = (self._average_filter_bboxes * self._total + self._first_stage_max_proposals -
                                           prediction_dict_['num_proposals'][0]) / (self._total + 1)
            self._average_time = (self._average_time * self._total + (toc - tic) * 1000) / (self._total + 1)
            self._time_per.append((toc - tic) * 1000)
            self._total += 1
            print('average filtered number: ', self._average_filter_bboxes)
            print('average detection timer: {} ms'.format(self._average_time))
        print('------------detection_end------------')
        print([np.array(result, dtype='double'), np.array(bboxes, dtype='double')])
        # return [output_dict_['detection_boxes'][0], output_dict_['detection_classes'][0]]
        return [np.array(result, dtype='double'), np.array(bboxes, dtype='double')]

    def timeline(self):
        fetched_timeline = timeline.Timeline(self._run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open('Experiment_1.json', 'w') as f:
            f.write(chrome_trace)
        pass

    def draw(self):
        plt.xlabel('num')
        plt.ylabel('time(ms)')
        self._time_per = np.array(self._time_per)
        x = list(range(self._time_per.size))
        plt.plot(x, self._time_per)
        plt.draw()
        plt.show()
        pass

    def finished(self):
        self._sess.close()


def draw(image, boxes, title):
    width = image.shape[1]
    height = image.shape[0]
    for box in boxes:
        xmin = int(box[1] * width)
        ymin = int(box[0] * height)
        xmax = int(box[3] * width)
        ymax = int(box[2] * height)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
        cv2.imshow(title, image)
    pass


if __name__ == '__main__':
    pipeline_config_path = r'..\model\pipeline\pipeline_resnet50.config'
    restore_path = r'..\weights\resnet50\train_org\model.ckpt-200000'

    image_path = os.path.join(cfg.ViBeProjectPath, 'video\pre\{}.jpg')
    gray_path = os.path.join(cfg.ViBeProjectPath, 'video\post\{}.jpg')

    detection = Detection(pipeline_config_path, restore_path, filter_threshold=0.5)
    detection.build_model()

    image = cv2.imread(image_path.format(6))
    image_gray = cv2.imread(gray_path.format(6), cv2.IMREAD_GRAYSCALE)
    wrap_num = 5
    for i in range(wrap_num):
        boxes, filter = detection.detection(image, image_gray, need_count=0)
    for i in range(0, 500):
        image = cv2.imread(image_path.format(i + 1))
        image_gray = cv2.imread(gray_path.format(i + 1), cv2.IMREAD_GRAYSCALE)
        print('-----------------', i + 1, '-------------------')
        boxes, filter = detection.detection(image, image_gray)

    detection.draw()
    detection.finished()
