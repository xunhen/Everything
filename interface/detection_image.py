import numpy as np
import tensorflow as tf
from PIL import Image

from distutils.version import StrictVersion
from object_detection.builders import model_builder
from object_detection.utils import config_util
from tools.generate_box_vibe import Generate_Box_By_ViBe
import cv2
import time

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')


class Detection(object):
    def __init__(self, pipeline_config_path, restore_path, rpn_type='orginal_rpn', filter_threshold=0.5, max_number=20):
        self._rpn_type = rpn_type
        self._filter_fn_arg = {'filter_threshold': filter_threshold}
        self._get_filter_boxes_fn = Generate_Box_By_ViBe()
        self._pipeline_config_path = pipeline_config_path
        self._restore_path = restore_path
        self._max_number = max_number
        self._replace_rpn_arg = None
        self._graph = None
        self._first_stage_max_proposals = None

        self._average_time = 0
        self._average_filter_bboxes = 0
        self._total = 0
        pass

    def init_some(self):
        pass

    def _process(self):
        pass

    def test(self):
        print(self.__pipeline_config_path)
        print(self._restore_path)

    def build_model(self):
        print('------------build_model_begin------------')
        batch_size = 1  # detect one image once
        configs = config_util.get_configs_from_pipeline_file(self._pipeline_config_path)
        model_config = configs['model']
        self._first_stage_max_proposals = model_config.faster_rcnn.first_stage_max_proposals
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
        print('------------build_model_end------------')

    def detection(self, image, gray_image=None, need_count=1):
        # when using filter ,the gray_image is necessary!!
        print('------------detection_begin------------')
        image_expanded = np.expand_dims(np.array(image), axis=0)
        # print(image_expanded.shape, image_expanded.dtype)
        feed_dict = {self._image_tensor: image_expanded}
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
            self._total += 1
            print('average filtered number: ', self._average_filter_bboxes)
            print('average detection timer: {} ms'.format(self._average_time))
        print('------------detection_end------------')
        # return [output_dict_['detection_boxes'][0], output_dict_['detection_classes'][0]]
        return [np.array(result, dtype='double'), np.array(bboxes, dtype='double')]

    def finished(self):
        self._sess.close()


if __name__ == '__main__':
    pipeline_config_path = r'..\model\pipeline\pipeline_resnet50.config'
    restore_path = r'..\weights\resnet50\train_org\model.ckpt-200000'
    image_path = r'F:\\PostGraduate\\Projects\\background\\video\\post\\278.jpg'
    detection = Detection(pipeline_config_path, restore_path)
    detection.build_model()
    image = Image.open(image_path)
    (im_width, im_height) = image.size
    image = np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)
    image_path = r'F:\\PostGraduate\\Projects\\background\\video\\pre\\278.jpg'
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    image_path = r'F:\\PostGraduate\\Projects\\background\\video\\post\\{}.jpg'
    gray_path = r'F:\\PostGraduate\\Projects\\background\\video\\pre\\{}.jpg'
    for i in range(1100, 100000):
        image = Image.open(image_path.format(i + 1))
        (im_width, im_height) = image.size
        image = np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)
        image_gray = cv2.imread(gray_path.format(i + 1), cv2.IMREAD_GRAYSCALE)
        print('-----------------', i, '-------------------')
        print(detection.detection(image, image_gray))
        print(detection.detection(image, image_gray))
    pass
