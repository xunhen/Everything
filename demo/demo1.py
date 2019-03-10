import numpy as np
import os
import sys
import skimage.io as io
import tensorflow as tf
import lxml.etree as etree
from object_detection.utils import dataset_util

from distutils.version import StrictVersion
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from PIL import Image
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import ops as utils_ops

from tools.generate_random_box import random_rpn

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')
from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util
from tools.create_txt_tf_record import parse_txt


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def parse(image_path_or_id, dataAnnFormat, coco_annFile=None, random_jittle=False):
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    if dataAnnFormat == 'txt':
        with tf.gfile.GFile(image_path_or_id + 'org.txt', 'r') as fid:
            txt = fid.read()
        datas, flag = parse_txt(txt)
        for data in datas:
            xmin.append(data['xmin'])
            ymin.append(data['ymin'])
            xmax.append(data['xmax'])
            ymax.append(data['ymax'])
    elif dataAnnFormat == 'xml':
        with tf.gfile.GFile(image_path_or_id + '.xml', 'r') as fid:
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
    else:
        coco = COCO(coco_annFile)
        imgs = coco.loadImgs(image_path_or_id)
        width = imgs[0]['width']
        height = imgs[0]['height']
        image = io.imread(imgs[0]['coco_url'])
        image_np = image
        annIds = coco.getAnnIds(imgIds=imgs[0]['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        for ann in anns:
            xmin.append(float(ann['bbox'][0]) / width)
            ymin.append(float(ann['bbox'][1]) / height)
            xmax.append(float(ann['bbox'][0] + ann['bbox'][2]) / width)
            ymax.append(float(ann['bbox'][1] + ann['bbox'][3]) / height)
    if random_jittle:
        bbox = random_rpn(zip(ymin, xmin, ymax, xmax))
        bbox = [[ymin, xmin, ymax, xmax] for (ymin, xmin, ymax, xmax) in zip(*bbox)]
    else:
        bbox = list(zip(ymin, xmin, ymax, xmax))
    bbox = np.array(bbox)
    bbox = bbox.reshape([1, -1, 4])
    if dataAnnFormat != 'coco':
        image = Image.open(image_path_or_id + '.jpg')
        image_np = load_image_into_numpy_array(image)

    return bbox, image_np


def test_images(pipeline_config_path, restore_path, category_index, images, bboxes, labels=None, path_to_log='log/test',
                show_with_proposal=False, filter_fn_arg=None, rpn_type=None, replace_rpn_arg=None, image_size=(12, 8)):
    batch_size = 1  # detect one image once
    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    model_config = configs['model']
    train_config = configs['train_config']
    input_config = configs['train_input_config']
    graph = tf.Graph()
    with graph.as_default():
        model = model_builder.build(model_config=model_config, is_training=False, rpn_type=rpn_type,
                                    filter_fn_arg=filter_fn_arg, replace_rpn_arg=replace_rpn_arg)

        with tf.variable_scope('placeholder'):
            image_tensor = tf.placeholder(tf.float32, shape=(batch_size, None, None, 3), name='images')
            if rpn_type == model_builder.RPN_TYPE_WITHOUT:
                rpn_box_list = [tf.placeholder(tf.float32, shape=(None, 4), name='rpn_box{}'.format(i)) for i in
                                range(batch_size)]
                model.provide_rpn_box_list(rpn_box_list)
            if filter_fn_arg:
                filter_box_list = [tf.placeholder(tf.float32, shape=(None, 4), name='filter_box{}'.format(i)) for i in
                                   range(batch_size)]
                model.provide_filter_box_list(filter_box_list)

        preprocessed_inputs, true_image_shapes = model.preprocess(image_tensor)
        prediction_dict = model.predict(preprocessed_inputs, true_image_shapes)
        output_dict = model.postprocess(prediction_dict, true_image_shapes)

        saver = tf.train.Saver()
        tf.summary.merge_all()

        with tf.Session() as sess:
            saver.restore(sess, restore_path)
            summary = tf.summary.FileWriter(path_to_log, sess.graph)
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            i = 0
            for index, image in enumerate(images):
                image_expanded = np.expand_dims(image, axis=0)
                feed_dict = {image_tensor: image_expanded}
                if rpn_type == model_builder.RPN_TYPE_WITHOUT:
                    for index_rpn, box in enumerate(rpn_box_list):
                        feed_dict[box] = bboxes[index][index_rpn]
                if filter_fn_arg:
                    for index_filter, box in enumerate(filter_box_list):
                        feed_dict[box] = bboxes[index][index_filter]
                output_dict_, prediction_dict_ = sess.run([output_dict, prediction_dict],
                                                          feed_dict=feed_dict,
                                                          options=run_options,
                                                          run_metadata=run_metadata)
                output_dict_['detection_classes'][0] += 1
                # Actual detection.
                # output_dict = run_inference_for_single_image(image_np, detection_graph)
                # Visualization of the results of a detection.
                if show_with_proposal:
                    proposal_boxes_normalized = prediction_dict_['proposal_boxes_normalized'][0]
                    proposal_boxes_scores = prediction_dict_['proposal_boxes_scores'][0]
                    if prediction_dict_['num_proposals'][0] < proposal_boxes_normalized.shape[0]:
                        proposal_boxes_normalized = proposal_boxes_normalized[:prediction_dict_['num_proposals'][0]]
                        proposal_boxes_scores = proposal_boxes_scores[:prediction_dict_['num_proposals'][0]]
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image,
                        proposal_boxes_normalized,
                        np.ones_like(proposal_boxes_scores, dtype=np.uint8),
                        proposal_boxes_scores,
                        category_index,
                        instance_masks=output_dict_.get('detection_masks'),
                        use_normalized_coordinates=True,
                        skip_labels=True,
                        line_thickness=1)
                    print('proposal_boxes_scores', proposal_boxes_scores)
                    plt.figure(figsize=image_size)
                    plt.imshow(image)

                vis_util.visualize_boxes_and_labels_on_image_array(
                    image,
                    output_dict_['detection_boxes'][0],
                    output_dict_['detection_classes'][0].astype(np.uint8),
                    output_dict_['detection_scores'][0],
                    category_index,
                    instance_masks=output_dict_.get('detection_masks'),
                    use_normalized_coordinates=True,
                    skip_labels=True,
                    line_thickness=1)
                print('detection_scores', output_dict_['detection_scores'])
                print('detection_classes', output_dict_['detection_classes'])
                plt.figure(figsize=image_size)
                plt.imshow(image)
                print('vaildation number', prediction_dict_['num_proposals'])
                i += 1
                summary.add_run_metadata(run_metadata, 'image%03d' % i)
                summary.flush()
            summary.close()
            plt.show()


if __name__ == '__main__':
    path_to_labels = os.path.join('..\dataset', 'mscoco_label_map.pbtxt')
    category_index = label_map_util.create_category_index_from_labelmap(path_to_labels, use_display_name=True)
    path_to_image = r'D:\DataSet\COCO'
    image_ids = [520, 536, 544]
    dataType = 'val2014'
    annFile = '{}/annotations/instances_{}.json'.format(path_to_image, dataType)
    pipeline_config_path = r'..\model\faster_rcnn_resnet50_coco_2018_01_28\pipeline.config'
    restore_path = r'..\model\faster_rcnn_resnet50_coco_2018_01_28\model.ckpt'

    random_jittle = False

    show_with_proposal = False

    filter_fn_arg = {'filter_threshold': 0.5}
    filter_fn_arg = None
    rpn_type = model_builder.RPN_TYPE_ORGIN
    rpn_type = model_builder.RPN_TYPE_WITHOUT
    replace_rpn_arg = {'type': 'rpn', 'scale': 1.1}
    # replace_rpn_arg = None

    bboxes = []
    images = []
    for image_id in image_ids:
        bbox, image = parse(image_id, dataAnnFormat='coco', coco_annFile=annFile,
                            random_jittle=random_jittle)
        bboxes.append(bbox)
        images.append(image)

    test_images(pipeline_config_path, restore_path, category_index, images, bboxes,
                show_with_proposal=show_with_proposal, filter_fn_arg=filter_fn_arg, rpn_type=rpn_type,
                replace_rpn_arg=replace_rpn_arg)

    pass
