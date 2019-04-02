import tensorflow as tf
from object_detection.utils import shape_utils
from object_detection.core import box_list
from object_detection.core import box_list_ops
import numpy as np
import time


def filter_pixel(proposal_boxes, filter_boxes, proposal_scores, proposal_classes=None, filter_threshold=0.5,
                 min_number=1, max_number=None):
    pass

def filter_bbox(proposal_boxes, filter_boxes, proposal_scores, proposal_classes=None, filter_threshold=0.5,
                min_number=100, max_number=None):
    """Returns proposal_boxes_result and validation.

        Args:
            proposal_boxes: A float tensor with shape
                [batch_size, num_proposals, 4] representing the (potentially zero
                padded) proposal boxes for all images in the batch.
                the format is such as [x1,y2,x2,y2]
            proposal_scores: A float tensor with shape
                [batch_size, num_proposals, num_class] representing the (potentially zero
                added) proposal boxes for all images in the batch.
            filter_boxes: A float tensor with shape
                [batch_size, num_filter, 4] representing the filter proposal boxes (potentially zero
                padded) proposal boxes in the batch.
            filter_threshold: A float number for threshold in iou'

        Returns:
            proposal_boxes_result: A float tensor with shape
                [batch_size, num_proposals, 4] representing the (potentially zero
                padded) proposal boxes for all images in the batch.
            validation: A float tensor with shape
                [batch_size,] representing the vaildative proposal boxes for the batch.
    """

    proposal_boxes_shape = shape_utils.combined_static_and_dynamic_shape(proposal_boxes)
    proposal_areas = tf.multiply(tf.maximum(0.0, proposal_boxes[..., 2] - proposal_boxes[..., 0]),
                                 tf.maximum(0.0, proposal_boxes[..., 3] - proposal_boxes[..., 1]))
    proposal_boxes = tf.expand_dims(proposal_boxes, axis=2)
    filter_boxes = tf.expand_dims(filter_boxes, axis=1)
    x1 = tf.maximum(proposal_boxes[..., 0], filter_boxes[..., 0])
    y1 = tf.maximum(proposal_boxes[..., 1], filter_boxes[..., 1])
    x2 = tf.minimum(proposal_boxes[..., 2], filter_boxes[..., 2])
    y2 = tf.minimum(proposal_boxes[..., 3], filter_boxes[..., 3])
    proposal_boxes = tf.squeeze(proposal_boxes, axis=2)
    filter_boxes = tf.squeeze(filter_boxes, axis=1)
    proposal_bg_areas = tf.multiply(tf.maximum(0.0, x2 - x1), tf.maximum(0.0, y2 - y1))
    proposal_bg_ious = tf.reduce_sum(proposal_bg_areas, axis=2)

    proposal_bg_areas_iou = tf.div(proposal_bg_ious,
                                   tf.where(tf.equal(proposal_areas, 0), tf.ones_like(proposal_areas), proposal_areas)[
                                       0])
    keeps = tf.greater_equal(proposal_bg_areas_iou, filter_threshold)
    validation = tf.reduce_sum(tf.to_int32(keeps), axis=1)
    max_numbers = tf.reduce_max(validation)
    max_numbers = tf.maximum(max_numbers, min_number)

    def _per_batch_gather_padding(args):
        """Returns proposal_boxes_result, proposal_score_result and validation.

        Args:
            args[0]: boxes. A float tensor with shape
                        [num_proposals, 4] representing the (potentially zero
                        padded) proposal boxes for all images in the batch.
                        the format is such as [x1,y1,x2,y2]
            args[1]: scores. A float tensor with shape
                        [num_proposals, num_class] representing the (potentially zero
                        added) proposal boxes for all images in the batch.
            args[2]: keep. A bool tensor with shape
                        [num_proposals, 1] representing the coordination needing to
                        keep with true.
            args[3]: max_number. A int tensor with shape
                        [,] representing the kept proposal's number with padding zeros

            Returns:
                    proposal_boxes_result: A float tensor with shape
                        [max_number, 4] representing the (potentially zero
                        padded) proposal boxes for all images in the batch.
                    proposal_score_result: A float tensor with shape
                        [max_number, num_class] representing the (potentially zero
                        padded) proposal boxes for all images in the batch.
            """
        boxes = args[0]
        scores = args[1]
        keep = args[2]
        if proposal_classes is not None:
            classes = args[3]
        result_indice = tf.where(keep)
        result_indice = tf.squeeze(result_indice, axis=-1)

        boxes_result = tf.gather(boxes, result_indice)
        boxes_result = shape_utils.pad_or_clip_tensor(
            boxes_result, max_numbers)

        score_result = tf.gather(scores, result_indice)
        score_result = shape_utils.pad_or_clip_tensor(
            score_result, max_numbers)

        result = [boxes_result, score_result]
        if proposal_classes is not None:
            class_result = tf.gather(classes, result_indice)
            class_result = shape_utils.pad_or_clip_tensor(
                class_result, max_numbers)
            result.append(class_result)
        return result

    if proposal_classes is not None:
        proposal_boxes_result, proposal_score_result, proposal_class_result = shape_utils.static_or_dynamic_map_fn(
            _per_batch_gather_padding,
            [proposal_boxes,
             proposal_scores,
             keeps, proposal_classes], dtype=[tf.float32, tf.float32, tf.float32])

        # test using
        # return proposal_boxes_result, vaildation, proposal_bg_areas,proposal_bg_ious,proposal_bg_areas_iou
        return proposal_boxes_result, proposal_score_result, proposal_class_result, validation, max_numbers  # change ???
    else:
        proposal_boxes_result, proposal_score_result = shape_utils.static_or_dynamic_map_fn(_per_batch_gather_padding,
                                                                                            [proposal_boxes,
                                                                                             proposal_scores,
                                                                                             keeps])

        # test using
        # return proposal_boxes_result, vaildation, proposal_bg_areas,proposal_bg_ious,proposal_bg_areas_iou
        return proposal_boxes_result, proposal_score_result, validation, max_numbers  # change ???


if __name__ == '__main__':
    proposal_boxes = tf.convert_to_tensor([[[1, 1, 5, 5], [6, 6, 8, 8]], [[1, 1, 5, 5], [6, 6, 8, 8]]],
                                          dtype=tf.float32)
    proposal_score = tf.convert_to_tensor([[[1, 1, 5, 5], [6, 6, 8, 8]], [[1, 1, 5, 5], [6, 6, 8, 8]]],
                                          dtype=tf.float32)
    bg_boxes = tf.convert_to_tensor(
        [[[0, 0, 4, 4], [5, 5, 7, 7], [7, 7, 8, 8]], [[0, 0, 3, 3], [5, 5, 8, 6.5], [7, 7, 8, 8]]],
        dtype=tf.float32)
    bg_filter_threshold = 0.5
    # proposal_boxes = np.random.rand(1, 600, 4)
    proposal_boxes = np.array([[[1, 1, 5, 5], [6, 6, 8, 8],[7, 7, 9, 9]]], dtype=np.float32)
    proposal_boxes = np.tile(proposal_boxes, (1, 100, 1))
    print(proposal_boxes.shape)
    proposal_boxes = tf.convert_to_tensor(proposal_boxes, dtype=tf.float32)
    proposal_score = tf.ones_like(proposal_boxes, dtype=tf.float32)
    # filter_boxes = np.random.rand(1, 20, 4)
    filter_boxes = np.array([[[1, 1, 2, 2], [6, 6, 6.1, 6.1]]],
                            dtype=np.float32)
    filter_boxes = np.tile(filter_boxes, (1, 10, 1))
    print(filter_boxes.shape)
    filter_boxes = tf.convert_to_tensor(filter_boxes, dtype=tf.float32)

    proposal_boxes_result_, proposal_score_result_, validation_, max_number_=filter_bbox(proposal_boxes, filter_boxes, proposal_score, filter_threshold=bg_filter_threshold)
    with tf.Session() as sess:
        for i in range(20):
            tic = time.time()
            proposal_boxes_result, proposal_score_result, validation, max_number = sess.run(
                [proposal_boxes_result_, proposal_score_result_, validation_, max_number_])
            toc = time.time()
            print('time', toc - tic)
            print(max_number)
            print(validation)
    pass
