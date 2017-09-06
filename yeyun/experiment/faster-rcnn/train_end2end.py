import os,sys
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,parentdir)
import argparse
import logging
import pprint
import mxnet as mx
import numpy as np

import common
from common.module import MutableModule
from common.utils.load_data import load_imageset, merge_roidb
from common.utils.load_model import load_param
from common.dataset.pascal_voc import PascalVOC
from common.dataset import *
import rcnn
from rcnn.rcnn_symbol_fpn_corr import get_rcnn_symbol
from rcnn.fpn_rpn_iter import FpnAnchorLoader
from rcnn import metric

from config import config, default, generate_config

def train_rcnn(network, dataset, image_set, val_image_set, dataset_root_path, valset_root_path,
               cache_path, val_cache_path, frequent, kvstore, work_load_list, flip, shuffle, ctx,
               save_prefix, begin_epoch, pretrained_prefix, pretrained_epoch, num_epochs,
               image_batch_size, base_lr, lr_step, log_file, cache_image, output_folder, eval_on_voc):
    #pydevd.settrace('10.98.39.244', port=10001, stdoutToServer=True, stderrToServer=True)
    # debug = True
    # if debug:
    #     import pudb; pu.db
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    # setup logger
    logger = logging.getLogger()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)-15s %(levelname)s %(message)s',
                        filename=log_file,
                        filemode='a')
    console = logging.StreamHandler()
#     console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)-8s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)
       
    # setup data folder and model folder
    if not os.path.exists('data'):
        os.mkdir('data')
    model_dir = os.path.dirname(save_prefix)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    # setup config
    config.TRAIN.BATCH_IMAGES = 1
    #config.TRAIN.BATCH_ROIS = 400
    config.TRAIN.END2END = True
    config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED = True
    
    # load symbol
    sym = get_rcnn_symbol(network, config.proposal_type, config.num_classes, config.num_anchors, config, is_train=True)
    if network.startswith('fpn_resnet-'):
        
        feat_deubg = []
        for i in range(default.fpn_depth-1):
            inter_feat_sym = sym.get_internals()["up_stage%d_conv_3x3_output"%(i+2)]
            feat_deubg.append(inter_feat_sym)
        feat_deubg.append(sym.get_internals()["stage5_conv_1x1_output"])
        max_data_shape = [('data',(1,3,640,992))]
        max_label_shape = []
        max_shapes = dict(max_data_shape + max_label_shape)
        #input_batch_size = max_shapes['data'][0]
        for index, feat_sym  in enumerate(feat_deubg) :
            _, feat_shape, _ = feat_sym.infer_shape(**max_shapes)
            print 'up_conv%d'%(index+2)
            print feat_shape

        feat_sym = []
        for i in range(len(config.RPN_FEAT_STRIDE)):
            inter_feat_sym = sym.get_internals()["rpn_cls_score_p%d_output"%(i+2)]
            feat_sym.append(inter_feat_sym)
    else:
        feat_sym = sym.get_internals()['rpn_cls_score_output']


    # setup multi-gpu
    batch_size = len(ctx)
    input_batch_size = config.TRAIN.BATCH_IMAGES * batch_size

    # print config
    pprint.pprint(config)
    
    datasets = [iset for iset in dataset.split('+')]
    image_sets = [iset for iset in image_set.split('+')]
    dataset_root_paths = [iset for iset in dataset_root_path.split('+')]
    if len(datasets)!=len(dataset_root_paths):
        assert('number of datasets not equal to number of dataset root paths')
    if len(datasets)!=len(image_sets):
        assert('number of datasets not equal to number of image_sets')
    train_roidbs = []
    for index in range(len(datasets)):
        print 'loading data from: dataset:{} image_set:{} image_root:{}'.format(datasets[index], image_sets[index], dataset_root_paths[index])
        train_roidbs.append(load_imageset(datasets[index], image_sets[index], dataset_root_paths[index], cache_path,
                                config.filter_strategy, category=config.category, flip=flip, task=config.task))    
    if config.eval_on_coco_minival:
        train_roidbs[1] = (train_roidbs[1])[:35000]
    train_roidb = merge_roidb(train_roidbs)
    if config.overfit:
        train_roidb = train_roidb[1:50]
    
    logging.info('Training images: %d' % (len(train_roidb)))

    # load training data
    train_data = FpnAnchorLoader(feat_sym, train_roidb, config, train_end2end=config.TRAIN.END2END, batch_size=input_batch_size, shuffle=shuffle, ctx=ctx,
                              feat_stride=config.RPN_FEAT_STRIDE, anchor_scales=config.ANCHOR_SCALES,
                              anchor_ratios=config.ANCHOR_RATIOS, aspect_grouping=config.TRAIN.ASPECT_GROUPING)


    # infer max shape
    max_data_shape = [('data', (input_batch_size, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]
    #max_data_shape = [('data',(input_batch_size,3,500,800))]
    max_data_shape, max_label_shape = train_data.infer_shape(max_data_shape)
    max_data_shape.append(('gt_boxes', (input_batch_size, 100, 5)))
    print 'providing maximum shape', max_data_shape, max_label_shape

    # infer shape
    data_shape_dict = dict(train_data.provide_data + train_data.provide_label)
    print train_data.provide_data

    arg_shape, out_shape, aux_shape = sym.infer_shape(**data_shape_dict)
    arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
    out_shape_dict = dict(zip(sym.list_outputs(), out_shape))
    aux_shape_dict = dict(zip(sym.list_auxiliary_states(), aux_shape))
    print 'output shape'
    pprint.pprint(out_shape_dict)

    # load and initialize params
    if begin_epoch != 0:
        assert save_prefix is not None
        logging.info('resume from %s-%d' % (save_prefix, begin_epoch))
        sym, arg_params, aux_params = mx.model.load_checkpoint(save_prefix, begin_epoch)
    elif pretrained_epoch is not None:
        assert pretrained_prefix is not None
        logging.info('init from pretrained model %s-%d' % (pretrained_prefix, pretrained_epoch))
        arg_params, aux_params = load_param(pretrained_prefix, pretrained_epoch, convert=True)
        # del arg_params['fc8_weight']
        # del arg_params['fc8_bias']
        arg_params['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_conv_3x3_weight'])
        arg_params['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_conv_3x3_bias'])
        arg_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_cls_score_weight'])
        arg_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_cls_score_bias'])
        arg_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_bbox_pred_weight'])
        arg_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_bbox_pred_bias'])
        arg_params['cls_score_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['cls_score_weight'])
        arg_params['cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['cls_score_bias'])
        arg_params['bbox_pred_weight'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['bbox_pred_weight'])
        arg_params['bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['bbox_pred_bias'])
        if (network.startswith('resnet-') or network.startswith('fpn_resnet')) and config.conv_stage==5:
            arg_params['fc6_weight'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['fc6_weight'])
            arg_params['fc6_bias'] = mx.nd.zeros(shape=arg_shape_dict['fc6_bias'])
            arg_params['fc7_weight'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['fc7_weight'])
            arg_params['fc7_bias'] = mx.nd.zeros(shape=arg_shape_dict['fc7_bias'])
        if network.startswith('fpn_resnet'):
            arg_params['stage5_conv_1x1_weight'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['stage5_conv_1x1_weight'])
            arg_params['stage5_conv_1x1_bias'] = mx.nd.zeros(shape=arg_shape_dict['stage5_conv_1x1_bias'])
            arg_params['up_stage4_conv_1x1_weight'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['up_stage4_conv_1x1_weight'])
            arg_params['up_stage4_conv_1x1_bias'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['up_stage4_conv_1x1_bias'])
            arg_params['up_stage4_conv_3x3_weight'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['up_stage4_conv_3x3_weight'])
            arg_params['up_stage4_conv_3x3_bias'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['up_stage4_conv_3x3_bias'])
            arg_params['up_stage3_conv_1x1_weight'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['up_stage3_conv_1x1_weight'])
            arg_params['up_stage3_conv_1x1_bias'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['up_stage3_conv_1x1_bias'])    
            arg_params['up_stage3_conv_3x3_weight'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['up_stage3_conv_3x3_weight'])
            arg_params['up_stage3_conv_3x3_bias'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['up_stage3_conv_3x3_bias'])          
            arg_params['up_stage2_conv_1x1_weight'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['up_stage2_conv_1x1_weight'])
            arg_params['up_stage2_conv_1x1_bias'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['up_stage2_conv_1x1_bias']) 
            arg_params['up_stage2_conv_3x3_weight'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['up_stage2_conv_3x3_weight'])
            arg_params['up_stage2_conv_3x3_bias'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['up_stage2_conv_3x3_bias'])     
    else:
        arg_params = None
        aux_params = None
        
    

    # check parameter shapes
    for k in sym.list_arguments():
        if k in data_shape_dict:
            continue
        assert k in arg_params, k + ' not initialized'
        assert arg_params[k].shape == arg_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(arg_shape_dict[k]) + ' provided ' + str(arg_params[k].shape)
    for k in sym.list_auxiliary_states():
        assert k in aux_params, k + ' not initialized'
        assert aux_params[k].shape == aux_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(aux_shape_dict[k]) + ' provided ' + str(aux_params[k].shape)

    # create solver
    fixed_param_prefix = config.fixed_params
    data_names = [k[0] for k in train_data.provide_data]
    label_names = [k[0] for k in train_data.provide_label]
    mod = MutableModule(sym, data_names=data_names, label_names=label_names,
                        logger=logger, context=ctx, work_load_list=work_load_list,
                        max_data_shapes=max_data_shape, max_label_shapes=max_label_shape,
                        fixed_param_prefix=fixed_param_prefix)

    # decide training params
    # metric
    rpn_eval_metric = metric.FPNRPNAccMetric()
    rpn_cls_metric = metric.FPNRPNLogLossMetric()
    rpn_bbox_metric = metric.FPNRPNL1LossMetric()
    eval_metric = metric.FPNRCNNAccMetric()
    cls_metric = metric.FPNRCNNLogLossMetric()
    bbox_metric = metric.FPNRCNNL1LossMetric()
    eval_metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [rpn_eval_metric, rpn_cls_metric, rpn_bbox_metric, eval_metric, cls_metric, bbox_metric]:
        eval_metrics.add(child_metric)
    #for child_metric in []:
    #    eval_metrics.add(child_metric)
    # callback
    batch_end_callback = common.callback.Speedometer(train_data.batch_size, frequent=frequent)
    
    means = np.tile(np.array(config.TRAIN.BBOX_MEANS), config.NUM_CLASSES)
    stds = np.tile(np.array(config.TRAIN.BBOX_STDS), config.NUM_CLASSES)
    epoch_end_callback = [common.callback.do_checkpoint(save_prefix, means, stds)]
    if config.do_eval_each_epoch:
        logging.info("eval on val data each epoch")
        print 'loading data from: dataset:{} image_set:{} image_root:{}'.format(config.eval_tool, val_image_set, valset_root_path)
        if config.eval_tool=='voc':
            imdb = PascalVOC(val_image_set, val_cache_path, valset_root_path, category=config.category)
            roidb = imdb.gt_roidb()
        if config.eval_tool=='coco':
            imdb = coco(val_image_set, val_cache_path, valset_root_path, category=config.category, task=config.task)
            roidb = imdb.gt_roidb()
            
        if config.eval_tool=='kitti':
            imdb = kitti(val_image_set, val_cache_path, valset_root_path, category=config.category)
            roidb = imdb.gt_roidb()
        
        if config.eval_tool == 'coco' and config.eval_on_coco_minival:
            roidb = roidb[35000:]
            imdb.num_images = len(roidb)
            imdb.image_set_index = imdb.image_set_index[35000:]
        if config.overfit:
            roidb = roidb[1:50]
            imdb.num_images = len(roidb)
            imdb.image_set_index = imdb.image_set_index[1:50]
        logging.info('Testing images: %d' % (len(roidb)))
        eval_on_end_callback = rcnn.callback.do_eval(network, output_folder, ctx[0], config, imdb, roidb, fpn=True)
        epoch_end_callback.append(eval_on_end_callback)
    lr_factor = 0.1
    lr_epoch = [int(epoch) for epoch in lr_step.split(',')]
    lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
    lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(epoch * len(train_roidb) / batch_size) for epoch in lr_epoch_diff]
    print 'lr', lr, 'lr_epoch_diff', lr_epoch_diff, 'lr_iters', lr_iters
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(lr_iters, lr_factor)
    # optimizer
    optimizer_params = {'momentum': 0.9,
                        'wd': 0.0005,
                        'learning_rate': lr,
                        'lr_scheduler': lr_scheduler,
                        'rescale_grad': (1.0 / batch_size),
                        'clip_gradient': 5}

    # train
    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback, kvstore=kvstore,
            optimizer='sgd', optimizer_params=optimizer_params,
            arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=num_epochs)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN network')
    # general
    parser.add_argument('--network', help='network name', default=default.network, type=str)
    parser.add_argument('--dataset', help='dataset name', default=default.dataset, type=str)
    args, rest = parser.parse_known_args()
    generate_config(args.network, args.dataset)
    parser.add_argument('--image_sets', help='image set names', default=default.image_set, type=str)
    parser.add_argument('--val_image_sets', help='val image sets', default=default.val_image_set, type=str)
    parser.add_argument('--dataset_root_path', help='dataset_root_path path', default=default.dataset_root_path, type=str)
    parser.add_argument('--valset_root_path', help='dataset root path', default=default.valset_root_path, type=str)
    parser.add_argument('--cache_path', help='cache path', default=default.cache_path, type=str)
    parser.add_argument('--val_cache_path', help='val cache path', default=default.val_cache_path, type=str)
    # training
    parser.add_argument('--frequent', help='frequency of logging', default=default.frequent, type=int)
    parser.add_argument('--kvstore', help='the kv-store type', default=default.kvstore, type=str)
    parser.add_argument('--work_load_list', help='work load for different devices', default=None, type=list)
    parser.add_argument('--flip', help='disable flip images', action='store_true', default=False)
    parser.add_argument('--shuffle', help='disable random shuffle', action='store_true', default=True)
    parser.add_argument('--image_batch_size', help='image size per mini-batch', default=default.image_batch_size, type=int)
    # e2e
    parser.add_argument('--gpus', help='GPU device to train with', default='0', type=str)
    parser.add_argument('--pretrained_prefix', help='pretrained model prefix', default=default.pretrained_prefix, type=str)
    parser.add_argument('--pretrained_epoch', help='pretrained model epoch', default=default.pretrained_epoch, type=int)
    parser.add_argument('--save_prefix', help='saved model prefix', default=default.save_prefix, type=str)
    parser.add_argument('--begin_epoch', help='begin epoch of training, use with resume', default=default.begin_epoch, type=int)
    parser.add_argument('--num_epoch', help='end epoch of training', default=default.train_epoch, type=int)
    parser.add_argument('--base_lr', help='base learning rate', default=default.base_lr, type=float)
    parser.add_argument('--lr_step', help='learning rate steps (in epoch)', default=default.lr_step, type=str)
    parser.add_argument('--log_file', help='log file', default=default.log_file, type=str)
    parser.add_argument('--cache_image', help='cache image or not', default=default.cache_image, type=bool)
    parser.add_argument('--output_folder', help='output_folder', default=default.output_folder, type=str)
    parser.add_argument('--eval_on_voc', help='eval on voc or not', action='store_true', default=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print 'Called with argument:', args
    ctx = [mx.cpu()] if args.gpus is None or args.gpus is '' else [
           mx.gpu(int(i)) for i in args.gpus.split(',')]
    
    train_rcnn(args.network, args.dataset, args.image_sets, args.val_image_sets,
               args.dataset_root_path, args.valset_root_path, args.cache_path, args.val_cache_path,
               args.frequent, args.kvstore, args.work_load_list, args.flip, args.shuffle, ctx,
               args.save_prefix, args.begin_epoch, args.pretrained_prefix, args.pretrained_epoch, args.num_epoch,
               args.image_batch_size, args.base_lr, args.lr_step, args.log_file, args.cache_image,
               args.output_folder, args.eval_on_voc)

if __name__ == '__main__':
    main()
