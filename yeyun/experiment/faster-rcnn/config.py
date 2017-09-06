import numpy as np
from easydict import EasyDict as edict
import os

config = edict()

config.category = 'all'
config.num_classes = 4
config.NUM_CLASSES = 4
config.task = 'detection'
config.anchor_param = edict()
config.target_size = 384
config.max_size = 1280

config.IMAGE_STRIDE = 0
config.RPN_FEAT_STRIDE = (4,8,16,32)
config.RCNN_FEAT_STRIDE = (4,8,16,32)

config.SCALES = [(384, 1280)]  # first is scale (the shorter side); second is max size
config.ANCHOR_SCALES = [8]
config.ANCHOR_RATIOS = [0.5, 1, 2]
config.NUM_ANCHORS = len(config.ANCHOR_SCALES) * len(config.ANCHOR_RATIOS)

config.input_channel = 3
config.feat_stride = 16
config.rpn_cls_agnostic = False
# config.input_mean = np.array([128, 128, 128], dtype=np.float32)
config.input_mean = np.array([103.939, 116.779, 123.68])#[b,g,r]
config.scale = 1
config.fixed_params = ['conv1', 'conv2']

config.proposal_type = 'rpn'

config.num_anchors = config.NUM_ANCHORS
config.init_from_voc=False

# dataset related params

    
config.TRAIN = edict()

# Faster-rcnn Train
config.TRAIN.ASPECT_GROUPING = True

# R-CNN
# rcnn rois batch size
config.TRAIN.BATCH_ROIS = 512
# rcnn rois sampling params
config.TRAIN.FG_FRACTION = 0.25
config.TRAIN.FG_THRESH = 0.5
config.TRAIN.BG_THRESH_HI = 0.5
config.TRAIN.BG_THRESH_LO = 0.0
# rcnn bounding box regression params
config.TRAIN.BBOX_REGRESSION_THRESH = 0.5
config.TRAIN.BBOX_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0])

# RPN anchor loader
# rpn anchors batch size
config.TRAIN.RPN_BATCH_SIZE = 256
# rpn anchors sampling params
config.TRAIN.RPN_FG_FRACTION = 0.5
config.TRAIN.RPN_POSITIVE_OVERLAP = 0.6
config.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
config.TRAIN.RPN_CLOBBER_POSITIVES = False
# rpn bounding box regression params
config.TRAIN.RPN_BBOX_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
config.TRAIN.RPN_POSITIVE_WEIGHT = -1.0

# used for end2end training
# RPN proposal
config.TRAIN.CXX_PROPOSAL = True
config.TRAIN.RPN_NMS_THRESH = 0.7
config.TRAIN.RPN_PRE_NMS_TOP_N = 12000
config.TRAIN.RPN_POST_NMS_TOP_N = 2000
config.TRAIN.RPN_MIN_SIZE = config.RPN_FEAT_STRIDE
# approximate bounding box regression
config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED = False
config.TRAIN.BBOX_MEANS = (0.0, 0.0, 0.0, 0.0)
config.TRAIN.BBOX_STDS = (0.1, 0.1, 0.2, 0.2)

config.TRAIN.BATCH_IMAGES = 1
config.TRAIN.BATCH_ROIS = 512

config.TEST = edict()

# R-CNN testing
# use rpn to generate proposal
config.TEST.HAS_RPN = False
# size of images for each device
config.TEST.BATCH_IMAGES = 1

# RPN proposal
config.FPN = edict()
config.FPN.TRAIN = edict()
config.FPN.TEST = edict()
config.TEST.CXX_PROPOSAL = True
config.TEST.RPN_NMS_THRESH = 0.7
config.TEST.RPN_PRE_NMS_TOP_N = 6000
config.TEST.RPN_POST_NMS_TOP_N = 300
config.FPN.TRAIN.RPN_POST_NMS_TOP_N = 2000#400*5=2000--->512
config.TEST.RPN_MIN_SIZE = config.RPN_FEAT_STRIDE

# RPN generate proposal
config.TEST.PROPOSAL_NMS_THRESH = 0.7
config.TEST.PROPOSAL_PRE_NMS_TOP_N = 20000
config.TEST.PROPOSAL_POST_NMS_TOP_N = 2000
config.FPN.TEST.RPN_POST_NMS_TOP_N = 1000#200*5=1000



config.TEST.PROPOSAL_MIN_SIZE = config.RPN_FEAT_STRIDE

# RCNN nms
config.TEST.NMS = 0.3
config.do_eval_each_epoch = True
config.eval_tool='kitti'
config.eval_on_coco_minival = False
config.overfit = False
config.use_voc_eval = False
config.txt_gt_path = '/home/ubuntu/Work/brbchen/unskychen/fpn_mxnet/kitti_mxnet/4rpn/experiment/faster-rcnn/voc_2007_test_person_det_gt.txt'


config.filter_strategy = edict()
config.filter_strategy.remove_empty = False
config.filter_strategy.remove_multi = False
config.filter_strategy.remove_unvis = False

config.eval_filter_strategy = edict()
config.eval_filter_strategy.remove_empty = True
config.eval_filter_strategy.remove_multi = False
config.eval_filter_strategy.remove_unvis = False

config.use_map_softmax = False
config.use_overlap_label = False

# default settings
default = edict()

# default.output_folder = 'vgg_voc2007_trainval_general_fix_bug'
default.output_folder = 'person_with_bg'

# default network
config.symbol_version = 'hobot'
default.network = 'fpn_resnet-50'
default.fpn_depth = 4
default.pretrained_prefix = 'model/resnet-50'
config.conv_stage = 5#choose RoIPooling stage
config.num_stage = 5#choose totall number of stage
default.pretrained_epoch = 0
default.base_lr = 0.001
# default dataset
default.dataset = 'PascalVOC'

default.image_set = '2007_trainval'
default.val_image_set = '2007_test'
default.dataset_root_path = '/data/yeyun/data/VOCdevkit'
default.valset_root_path = '/data/yeyun/data/VOCdevkit'
# default.valset_root_path = 'E:/data/horizon_data/hand/hand_test_data'
default.cache_path = 'data/train'
default.val_cache_path = 'data/val'
# default training
default.frequent = 20
default.kvstore = 'device'

default.save_prefix = '{0}/model/{0}'.format(default.output_folder)
default.train_epoch = 10
default.lr_step = '7'
default.log_file = '{0}/{0}.log'.format(default.output_folder)
default.gpu = '0'

# size of images for each device,
default.begin_epoch = 3
default.image_batch_size = 1
default.shuffle = True
default.flip =  False
default.cache_image = False

default.hand_track_benchmark_path = 'E:/data/horizon_data/hand/hand_test_data/hand_track_benckmark'

# network settings
network = edict()

network.hobot = edict()

network.hobot_s8 = edict()
network.feat_stride = 8

network.resnet = edict()
network.resnet.pretrained_prefix = '../../../common/model/resnet-101'
network.resnet.pretrained_epoch = 0
network.resnet.fixed_params = ['conv0', 'gamma', 'beta']
#network.resnet.fixed_params = ['conv1', 'conv2']
network.resnet.FIXED_PARAMS_SHARED = ['conv0', 'stage1', 'stage2', 'stage3', 'gamma', 'beta']

network.vgg = edict()
network.vgg.pretrained_prefix = '../../../common/model/vgg16'
network.vgg.pretrained_epoch = 0
network.vgg.fixed_params = ['conv1', 'conv2']

# dataset settings
dataset = edict()

dataset.hobot = edict()
dataset.PascalVOC = edict()

def generate_config(_network, _dataset):
    if _network.startswith('resnet_full_fpn_2'):
        depth = int(_network.split('-')[1])
        _network = 'resnet_full_fpn_2'
        network.resnet_full_fpn_2.pretrained_prefix = '../../common/model/resnet-{}'.format(depth)
    elif _network.startswith('resnet_full_fpn_3'):
        depth = int(_network.split('-')[1])
        _network = 'resnet_full_fpn_3'
        network.resnet_full_fpn_3.pretrained_prefix = '../../common/model/resnet-{}'.format(depth)
    elif _network.startswith('resnet_full_fpn_4'):
        depth = int(_network.split('-')[1])
        _network = 'resnet_full_fpn_4'
        network.resnet_full_fpn_4.pretrained_prefix = '../../common/model/resnet-{}'.format(depth)
    elif _network.startswith('resnet_full_fpn'):
        depth = int(_network.split('-')[1])
        _network = 'resnet_full_fpn'
        network.resnet_full_fpn.pretrained_prefix = '../../common/model/resnet-{}'.format(depth)
    elif _network.startswith('fpn_resnet-'):
        depth = int(_network.split('-')[1])
        _network = 'resnet'
        if config.init_from_voc:
            network.resnet.pretrained_prefix = '../../common/model/resnet_voc0712-{}'.format(depth)
        else:
            network.resnet.pretrained_prefix = '../../common/model/resnet-{}'.format(depth)
    if _network.startswith('resnet-'):
        depth = int(_network.split('-')[1])
        _network = 'resnet'
        network.resnet.pretrained_prefix = '../../common/model/resnet-{}'.format(depth)
    
    for k, v in network[_network].items():
        if k in config:
            config[k] = v
        elif k in default:
            default[k] = v
    # for k, v in dataset[_dataset].items():
    #     if k in config:
    #         config[k] = v
    #     elif k in default:
    #         default[k] = v

