import tensorflow as tf

from Lib import faster_rcnn_meta_arch
import tensorflow.contrib.slim as slim


class FasterRCNNVGG16FeatureExtractor(
    faster_rcnn_meta_arch.FasterRCNNFeatureExtractor):
    """Faster R-CNN Resnet V1 feature extractor implementation."""

    def __init__(self,
                 is_training,
                 first_stage_features_stride,
                 batch_norm_trainable=False,
                 second_use_dropout=True,
                 second_dropout_keep_prob=0.5,
                 reuse_weights=None,
                 weight_decay=0.0):
        """Constructor.

        Args:
          architecture: Architecture name of the Resnet V1 model.
          resnet_model: Definition of the Resnet V1 model.
          is_training: See base class.
          first_stage_features_stride: See base class.
          batch_norm_trainable: See base class.
          reuse_weights: See base class.
          weight_decay: See base class.

        Raises:
          ValueError: If `first_stage_features_stride` is not 8 or 16.
        """
        print(first_stage_features_stride)
        if first_stage_features_stride != 8 and first_stage_features_stride != 16:
            raise ValueError('`first_stage_features_stride` must be 8 or 16.')
        self._architecture = 'faster_rcnn_vgg16'
        self._second_use_dropout = second_use_dropout
        self._second_dropout_keep_prob = second_dropout_keep_prob
        super(FasterRCNNVGG16FeatureExtractor, self).__init__(
            is_training, first_stage_features_stride, batch_norm_trainable,
            reuse_weights, weight_decay)

    def preprocess(self, resized_inputs):
        """Faster R-CNN Resnet V1 preprocessing.

        VGG style channel mean subtraction as described here:
        https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
        Note that if the number of channels is not equal to 3, the mean subtraction
        will be skipped and the original resized_inputs will be returned.

        Args:
          resized_inputs: A [batch, height_in, width_in, channels] float32 tensor
            representing a batch of images with values between 0 and 255.0.

        Returns:
          preprocessed_inputs: A [batch, height_out, width_out, channels] float32
            tensor representing a batch of images.

        """
        if resized_inputs.shape.as_list()[3] == 3:
            channel_means = [123.68, 116.779, 103.939]
            return resized_inputs - [[channel_means]]
        else:
            return resized_inputs

    def _extract_proposal_features(self, preprocessed_inputs, scope):
        """Extracts first stage RPN features.

        Args:
          preprocessed_inputs: A [batch, height, width, channels] float32 tensor
            representing a batch of images.
          scope: A scope name.

        Returns:
          rpn_feature_map: A tensor with shape [batch, height, width, depth]
          activations: A dictionary mapping feature extractor tensor names to
            tensors

        Raises:
          InvalidArgumentError: If the spatial size of `preprocessed_inputs`
            (height or width) is less than 33.
          ValueError: If the created network is missing the required activation.
        """
        if len(preprocessed_inputs.get_shape().as_list()) != 4:
            raise ValueError('`preprocessed_inputs` must be 4 dimensional, got a '
                             'tensor of shape %s' % preprocessed_inputs.get_shape())
        shape_assert = tf.Assert(
            tf.logical_and(
                tf.greater_equal(tf.shape(preprocessed_inputs)[1], 33),
                tf.greater_equal(tf.shape(preprocessed_inputs)[2], 33)),
            ['image size must at least be 33 in both height and width.'])

        with tf.control_dependencies([shape_assert]):
            with tf.variable_scope(self._architecture, reuse=self._reuse_weights) as sc:
                end_points_collection = sc.original_name_scope + '_end_points'
                # Collect outputs for conv2d, fully_connected and max_pool2d.
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    outputs_collections=end_points_collection):
                    net = slim.repeat(preprocessed_inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                    net = slim.max_pool2d(net, [2, 2], scope='pool1')
                    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                    net = slim.max_pool2d(net, [2, 2], scope='pool2')
                    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                    net = slim.max_pool2d(net, [2, 2], scope='pool3')
                    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                    net = slim.max_pool2d(net, [2, 2], scope='pool4')
                    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                    # Convert end_points_collection into a end_point dict.
                    end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        return net, end_points

    def _extract_box_classifier_features(self, proposal_feature_maps, scope):
        """Extracts second stage box classifier features.

        Args:
          proposal_feature_maps: A 4-D float tensor with shape
            [batch_size * self.max_num_proposals, crop_height, crop_width, depth]
            representing the feature map cropped to each proposal.
          scope: A scope name (unused).

        Returns:
          proposal_classifier_features: A 4-D float tensor with shape
            [batch_size * self.max_num_proposals, height, width, depth]
            representing box classifier features for each proposal.
        """
        with tf.variable_scope(self._architecture, reuse=self._reuse_weights):
            # Use conv2d instead of fully_connected layers.
            net = slim.conv2d(proposal_feature_maps, 1024, [7, 7], padding='VALID', scope='fc6')
            if self._second_use_dropout:
                net = slim.dropout(net, self._second_dropout_keep_prob, is_training=self._train_batch_norm,
                                   scope='dropout6')
            net = slim.conv2d(net, 1024, [1, 1], scope='fc7')

        return net

    def restore_from_classification_checkpoint_fn(
            self,
            first_stage_feature_extractor_scope,
            second_stage_feature_extractor_scope):
        """Returns a map of variables to load from a foreign checkpoint.

        Args:
          first_stage_feature_extractor_scope: A scope name for the first stage
            feature extractor.
          second_stage_feature_extractor_scope: A scope name for the second stage
            feature extractor.

        Returns:
          A dict mapping variable names (to load from a checkpoint) to variables in
          the model graph.
        """
        variables_to_restore = {}
        for variable in tf.global_variables():
            for scope_name in [first_stage_feature_extractor_scope,
                               second_stage_feature_extractor_scope]:
                if variable.op.name.startswith(scope_name):
                    var_name = variable.op.name.replace(scope_name + '/', '')
                    if var_name.startswith('faster_rcnn_vgg16'):
                        var_name = var_name.replace('faster_rcnn_vgg16', 'vgg_16')
                        variables_to_restore[var_name] = variable
        return variables_to_restore
