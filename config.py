from easydict import EasyDict as edict
import os

__C = edict()
cfg = __C
__C.ROOT = r'E:\\CODE\\Python\\TensorFlowTest\\Faster_RCNN'
# __C.PATH = r'D:\\Software\\Miniconda\\envs\tensorflow\\Lib\\site-packages\\tensorflow\\models\\research\\object_detection'
# __C.DATASET_PATH = os.path.join(__C.ROOT, 'DataSet')
# __C.PATH_TO_LABELS = os.path.join(__C.DATASET_PATH, 'mot_label_map.pbtxt')
# __C.LOG_PATH = os.path.join(__C.ROOT, 'Log')

_resnet50 = {
    'config': os.path.join('Model', 'faster_rcnn_resnet50_coco_2018_01_28/faster_rcnn_resnet50_coco_2018_01_28',
                           'pipeline.config'),
    'model': os.path.join('Model', 'faster_rcnn_resnet50_coco_2018_01_28/faster_rcnn_resnet50_coco_2018_01_28',
                          'model.ckpt')}
_inception_v2 = {'config': os.path.join('Model', 'faster_rcnn_inception_v2_coco_2018_01_28', 'pipeline.config'),
                 'model': os.path.join('Model', 'faster_rcnn_inception_v2_coco_2018_01_28', 'model.ckpt')}
__C.TRAINED_MODEL = {'resnet50': _resnet50, 'inception_v2': _inception_v2}


__C.ViBeProjectPath=r'D:\PostGraduate\Projects\background'
__C.DataSetPath=r'D:\DataSet'

# __C.HAS_RPN = True
# __C.RANDOM_JITTLE = False
#
# __C.USING_FILTER_FN = True
# __C.filter_threshold = 0.5
