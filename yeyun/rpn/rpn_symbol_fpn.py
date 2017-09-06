#-*- coding: UTF-8 -*- 
'''
Created on 2017年7月24日

@author: Yeyun
'''
import os,sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)

import mxnet as mx
import numpy as np
from common.symbol import get_resnet_conv, get_vgg_conv, get_hobot_conv, get_hobot_s8_conv, get_resnet_conv5, get_fpn_resnet_conv, get_top_down_fpn_resnet_conv
from common.symbol import residual_unit
from config import config
import proposal
import roi_concat
import proposal_target
import fpn_proposal_target
import fpn_proposal

eps = 2e-5
use_global_stats = False
res_deps = {'18': (2, 2, 2, 2), '34': (3, 4, 6, 3), '50': (3, 4, 6, 3), '101': (3, 4, 23, 3), '152': (3, 8, 36, 3), '200': (3, 24, 36, 3)}

def get_rpn_symbol(network, proposal_type, num_classes, num_anchors, config, is_train=False):
    """ resnet symbol for scf train and test """
    data = mx.symbol.Variable(name="data")

    rpn_conv_weight = mx.symbol.Variable(name='rpn_conv_3x3_weight')
    rpn_conv_bias = mx.symbol.Variable(name='rpn_conv_3x3_bias')
    rpn_cls_weight = mx.symbol.Variable(name='rpn_cls_score_weight')
    rpn_cls_bias = mx.symbol.Variable(name='rpn_cls_score_bias')
    rpn_bbox_weight = mx.symbol.Variable(name='rpn_bbox_pred_weight')
    rpn_bbox_bias = mx.symbol.Variable(name='rpn_bbox_pred_bias')

    if network.startswith('fpn_resnet-'):
        depth = int(network.split('-')[1])
        conv_feat = get_fpn_resnet_conv(data, depth)
        rpn_depth = len(config.RPN_FEAT_STRIDE)
        rcnn_depth = len(config.RCNN_FEAT_STRIDE)

    im_info = mx.symbol.Variable(name="im_info")
    feat_shape = mx.symbol.Variable(name='feat_shape')
    # RPN
    if network.startswith('fpn_resnet-'):
        rpn_cls_score_reshape = []
        rpn_bbox_pred = []
        for i in range(rpn_depth):
            rpn_conv_ = mx.symbol.Convolution(data=conv_feat[i], weight=rpn_conv_weight, bias=rpn_conv_bias, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_p%d"%(i+2))
            rpn_conv_ = mx.symbol.Activation(data=rpn_conv_, act_type="relu", name="rpn_relu_p%d"%(i+2))
            rpn_cls_score_ = mx.symbol.Convolution(data=rpn_conv_, weight=rpn_cls_weight, bias=rpn_cls_bias, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score_p%d"%(i+2))
            rpn_bbox_pred_ = mx.symbol.Convolution(data=rpn_conv_, weight=rpn_bbox_weight, bias=rpn_bbox_bias, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred_p%d"%(i+2))
            rpn_bbox_pred.append(rpn_bbox_pred_)
            rpn_cls_score_reshape.append(mx.symbol.Reshape(data=rpn_cls_score_, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape_p%d" %(i+2)))        
    # ROI Proposal


    if is_train:
        # gt_boxes = mx.symbol.Variable(name="gt_boxes")
        if network.startswith('fpn_resnet-'):
            rpn_label = []
            rpn_bbox_target = []
            rpn_bbox_weight = []
            for i in range(rpn_depth):
                rpn_label.append(mx.symbol.Variable(name='label%d'%(i+1)))
                rpn_bbox_target.append(mx.symbol.Variable(name='bbox_target%d'%(i+1)))
                rpn_bbox_weight.append(mx.symbol.Variable(name='bbox_weight%d'%(i+1)))

            rpn_cls_prob = []
            rpn_bbox_loss = []
            for i in range(rpn_depth):
                rpn_cls_prob_ = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape[i], label=rpn_label[i], multi_output=True, normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob_p%d"%(i+2))
                rpn_cls_prob.append(rpn_cls_prob_)
                rpn_bbox_lpss_1 = rpn_bbox_weight[i] * mx.symbol.smooth_l1(name='rpn_bbox_loss_p%d_'%(i+2), scalar=3.0, data=(rpn_bbox_pred[i] - rpn_bbox_target[i]))
                rpn_bbox_loss_1 = mx.sym.MakeLoss(name='rpn_bbox_loss_p%d'%(i+2), data=rpn_bbox_lpss_1, grad_scale=1.0 / config.TRAIN.RPN_BATCH_SIZE)
                rpn_bbox_loss.append(rpn_bbox_loss_1)
    if network.startswith('fpn_resnet-'):
        if not is_train:#output roi:[400*5,5]
            rpn_cls_act_reshapes_list = []
            rpn_bbox_pred_list = []
            for i in range(rpn_depth):
                rpn_cls_act = mx.symbol.SoftmaxActivation(
                    data=rpn_cls_score_reshape[i], mode="channel", name="rpn_cls_act%d" %(i+2))
                rpn_cls_act_reshape = mx.symbol.Reshape(
                    data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape%d' %(i+2))
                rpn_cls_act_reshapes_list.append(mx.symbol.flatten(data=rpn_cls_act_reshape, name='flatten_rpn_cls_act_%d'%(i+2)))
                rpn_bbox_pred_list.append(mx.symbol.flatten(data=rpn_bbox_pred[i], name='flatten_rpn_bbox_pred_%d'%(i+2)))
            if rpn_depth == 5:
                concat_flat_rpn_cls_act = mx.symbol.Concat(rpn_cls_act_reshapes_list[0],rpn_cls_act_reshapes_list[1],rpn_cls_act_reshapes_list[2],rpn_cls_act_reshapes_list[3],rpn_cls_act_reshapes_list[4],dim=1,name='concat_rpn_cls_act')
                concat_flat_rpn_bbox_pred = mx.symbol.Concat(rpn_bbox_pred_list[0],rpn_bbox_pred_list[1],rpn_bbox_pred_list[2],rpn_bbox_pred_list[3],rpn_bbox_pred_list[4],dim=1,name='concat_rpn_bbox_pred')
            rois = mx.symbol.Custom(
                cls_prob=concat_flat_rpn_cls_act, bbox_pred=concat_flat_rpn_bbox_pred, im_info=im_info, name='rois',
                op_type='roi_concat', feat_stride='4,8,16,32,64', feat_shape=feat_shape,
                scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS), output_score=True,
                rpn_pre_nms_top_n=config.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.FPN.TEST.RPN_POST_NMS_TOP_N,
                threshold=config.TEST.RPN_NMS_THRESH, rpn_min_size='4,8,16,32,64')    #[1000,4]
    # group 
    if is_train:
        if network.startswith('fpn_resnet-'):
            output = [rpn_cls_prob[0]]
            for r in rpn_cls_prob[1:]:
                output.append(r)
            for r in rpn_bbox_loss:
                output.append(r)
            group = mx.symbol.Group(output)
    else:
        group = rois
    return group

if __name__ == "__main__":
    batch_size = 1 
    
    # deploy_symbol = get_scf_symbol('hobot', is_train=False, with_attribute=True, hobot_predict=True)
    # deploy_symbol.save('./scf_hobot_with_attribute-deploy.json')
    # print(deploy_symbol.list_arguments())
    
    deploy_symbol = get_rcnn_symbol('fpn_resnet-101', 'rpn', 20, 9, config, is_train=True)
    deploy_symbol.save('./fpn_resnet-101.json')
#     data_shape = (batch_size, 3, 128, 128);
#     label1_shape = (batch_size, 6, 4,4); 
      
    dot = mx.viz.plot_network(deploy_symbol)
    dot.render('test-output/round-table.gv', view=True)