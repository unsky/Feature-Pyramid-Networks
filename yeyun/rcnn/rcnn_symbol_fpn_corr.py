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
from common.symbol import get_resnet_conv, get_vgg_conv,get_resnet_conv5, get_fpn_resnet_conv
from common.symbol import residual_unit
from config import config
import proposal
import roi_concat
import proposal_target
import fpn_proposal_target
import fpn_proposal
import assign_rois

eps = 2e-5
use_global_stats = False
res_deps = {'18': (2, 2, 2, 2), '34': (3, 4, 6, 3), '50': (3, 4, 6, 3), '101': (3, 4, 23, 3), '152': (3, 8, 36, 3), '200': (3, 24, 36, 3)}

def get_rcnn_symbol(network, proposal_type, num_classes, num_anchors, config, is_train=False):
    """ resnet symbol for scf train and test """
    data = mx.symbol.Variable(name="data")

    rpn_conv_weight = mx.symbol.Variable(name='rpn_conv_3x3_weight')
    rpn_conv_bias = mx.symbol.Variable(name='rpn_conv_3x3_bias')
    rpn_cls_weight = mx.symbol.Variable(name='rpn_cls_score_weight')
    rpn_cls_bias = mx.symbol.Variable(name='rpn_cls_score_bias')
    rpn_bbox_weight = mx.symbol.Variable(name='rpn_bbox_pred_weight')
    rpn_bbox_bias = mx.symbol.Variable(name='rpn_bbox_pred_bias')

    rcnn_fc6_weight = mx.symbol.Variable(name='fc6_weight')
    rcnn_fc6_bias = mx.symbol.Variable(name='fc6_bias')
    rcnn_fc7_weight = mx.symbol.Variable(name='fc7_weight')
    rcnn_fc7_bias = mx.symbol.Variable(name='fc7_bias')
    rcnn_cls_weight = mx.symbol.Variable(name='cls_score_weight')
    rcnn_cls_bias = mx.symbol.Variable(name='cls_score_bias')
    rcnn_bbox_weight = mx.symbol.Variable(name='bbox_pred_weight')
    rcnn_bbox_bias = mx.symbol.Variable(name='bbox_pred_bias')

    if network.startswith('fpn_resnet-'):
        depth = int(network.split('-')[1])
        conv_feat = get_fpn_resnet_conv(data, depth)
        rpn_depth = len(config.RPN_FEAT_STRIDE)
        rcnn_depth = len(config.RCNN_FEAT_STRIDE)

    if proposal_type == 'existed_roi':
        rois = mx.symbol.Variable(name='rois')
        rois = mx.symbol.Reshape(data=rois, shape=(-1, 5), name='rois_reshape')
    elif proposal_type == 'rpn':
        im_info = mx.symbol.Variable(name="im_info")
        feat_shape = mx.symbol.Variable(name='feat_shape')
        # RPN
        #####################need to share params here!!!!!!!!!!!!!!##########################
        ##########P5
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

        if is_train:
            gt_boxes = mx.symbol.Variable(name="gt_boxes")
            
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
            rois_fpn = []
            score_fpn = []

            if is_train:#output roi:[400*5,5]
                rpn_cls_act_reshapes_list = []
                rpn_bbox_pred_list = []
                for i in range(rpn_depth):
                    rpn_cls_act = mx.symbol.SoftmaxActivation(
                        data=rpn_cls_score_reshape[i], mode="channel", name="rpn_cls_act%d" %(i+2))
                    rpn_cls_act_reshape = mx.symbol.Reshape(
                        data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape%d' %(i+2))#shape:[1,2*3,h,w]


                    rois, score = mx.sym.Custom(
                       cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred[i], im_info=im_info, name='rois%d' %(i+2),
                       op_type='proposal', feat_stride=str(2**(i+2)),feat_shape=feat_shape,
                       scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
                       rpn_pre_nms_top_n=config.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.FPN.TRAIN.RPN_POST_NMS_TOP_N,
                       threshold=config.TEST.RPN_NMS_THRESH, rpn_min_size= str(2**(i+2))
                       )
                    rois_fpn.append(rois)
                    score_fpn.append(score)
                    
                
            else: #output roi:[200*5,5]
                rpn_cls_act_reshapes_list = []
                rpn_bbox_pred_list = []
                for i in range(rpn_depth):
                    rpn_cls_act = mx.symbol.SoftmaxActivation(
                        data=rpn_cls_score_reshape[i], mode="channel", name="rpn_cls_act%d" %(i+2))
                    rpn_cls_act_reshape = mx.symbol.Reshape(
                        data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape%d' %(i+2))

                    rois,score = mx.sym.Custom(
                       cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred[i], im_info=im_info, name='rois%d' %(i+2),
                       op_type='proposal', feat_stride=str(2**(i+2)),feat_shape=feat_shape,
                       scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
                       rpn_pre_nms_top_n=config.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.FPN.TRAIN.RPN_POST_NMS_TOP_N,
                       threshold=config.TEST.RPN_NMS_THRESH, rpn_min_size= str(2**(i+2))
                       )

                    rois_fpn.append(rois)
                    score_fpn.append(score)
    #### train and test
    rois = mx.symbol.Concat(rois_fpn[0],rois_fpn[1],rois_fpn[2],rois_fpn[3],dim=0,name='concat_rois')
    score = mx.symbol.Concat(score_fpn[0],score_fpn[1],score_fpn[2],score_fpn[3],dim=0,name='concat_score')
    rois = mx.symbol.Custom(rois = rois ,score = score,op_type='assign_rois')  
    rois = mx.symbol.SliceChannel(data=rois, axis=0, num_outputs=rpn_depth, name='slice_rois')

    if is_train:
        # ROI proposal target

        if network.startswith('fpn_resnet-'):
            gt_boxes_reshape = mx.symbol.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
            rois_all =[]
            label =[]
            bbox_target =[]
            bbox_weight = []

            for i in range(rpn_depth):
                irois, ilabel, ibbox_target, ibbox_weight = mx.sym.Custom(rois=rois[i], gt_boxes=gt_boxes_reshape,
                                                                   op_type='proposal_target',
                                                                   num_classes=num_classes, batch_images=config.TRAIN.BATCH_IMAGES,
                                                                   batch_rois=config.TRAIN.BATCH_ROIS, fg_fraction=config.TRAIN.FG_FRACTION)
                rois_all.append(irois)#[512*4,5]
                label.append(ilabel)#[512*4,]
                bbox_target.append(ibbox_target)#[512*4,8]
                bbox_weight.append(ibbox_weight) 
            rois = rois_all

    # Fast R-CNN
    if network.startswith('fpn_resnet-'):
        cls_score = []
        bbox_pred = []
        for i in range(rpn_depth):
            roi_pool = mx.symbol.ROIPooling(name='roi_pool_%d' %(i+2), data=conv_feat[i], rois=rois[i], pooled_size=(7, 7), spatial_scale=1.0 / (config.RCNN_FEAT_STRIDE)[i])
            flatten = mx.symbol.Flatten(data=roi_pool, name="flatten_%d" %(i+2))
            fc6 = mx.symbol.FullyConnected(data=flatten, weight=rcnn_fc6_weight, bias=rcnn_fc6_bias, num_hidden=1024, name="fc6_%d" %(i+2))
            fc6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6_%d" %(i+2))
            fc6 = mx.symbol.Dropout(data=fc6, p=0.5, name="drop6_%d" %(i+2))
            # group 7
            fc7 = mx.symbol.FullyConnected(data=fc6, weight=rcnn_fc7_weight, bias=rcnn_fc7_bias, num_hidden=1024, name="fc7_%d" %(i+2))
            fc7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7_%d" %(i+2))
            fc7 = mx.symbol.Dropout(data=fc7, p=0.5, name="drop7_%d" %(i+2))
            cls_score.append(mx.symbol.FullyConnected(name='cls_score_%d' %(i+2), weight=rcnn_cls_weight, bias=rcnn_cls_bias, data=fc7, num_hidden=num_classes))     
            # bounding box regression
            bbox_pred.append(mx.symbol.FullyConnected(name='bbox_pred_%d' %(i+2), weight=rcnn_bbox_weight, bias=rcnn_bbox_bias, data=fc7, num_hidden=num_classes * 4))

    
    if is_train:
        if network.startswith('fpn_resnet-'):
            cls_prob = []
            bbox_loss = []
            labels = []
            for i in range(rpn_depth):##############label need to change
                cls_prob_ = mx.symbol.SoftmaxOutput(name='cls_prob_p%d'%(i+2), data=cls_score[i], label=label[i], normalization='valid', use_ignore=True, ignore_label=-1)
                bbox_loss_1 = bbox_weight[i] * mx.symbol.smooth_l1(name='bbox_loss_p%d_'%(i+2), scalar=1.0, data=(bbox_pred[i] - bbox_target[i]))#org:1
                bbox_loss_2 = mx.sym.MakeLoss(name='bbox_loss_p%d'%(i+2), data=bbox_loss_1, grad_scale=4.0 / config.TRAIN.BATCH_ROIS)
                cls_prob.append(mx.symbol.Reshape(data=cls_prob_, shape=(config.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape_p%d'%(i+2)))
                labels.append(mx.symbol.Reshape(data=label[i], shape=(config.TRAIN.BATCH_IMAGES, -1), name='label_reshape_p%d'%(i+2)))
                bbox_loss.append(mx.symbol.Reshape(data=bbox_loss_2, shape=(config.TRAIN.BATCH_IMAGES, -1, 4 * num_classes), name='bbox_loss_reshape_p%d'%(i+2)))        
    else:
        if network.startswith('fpn_resnet-'):
            cls_prob = []
            bbox_prob = []
            for i in range(rpn_depth):
                cls_prob_ = mx.symbol.SoftmaxActivation(name='cls_prob_p%d'%(i+2), data=cls_score[i])
                cls_prob.append(mx.symbol.Reshape(data=cls_prob_, shape=(config.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape_p%d'%(i+2)))
                bbox_prob.append(mx.symbol.Reshape(data=bbox_pred[i], shape=(config.TEST.BATCH_IMAGES, -1, 4 * num_classes), name='bbox_pred_reshape_p%d'%(i+2)))         
        
    
    # group 
    if is_train:
        if network.startswith('fpn_resnet-'):
            output = [rpn_cls_prob[0]]
            for r in rpn_cls_prob[1:]:
                output.append(r)
            for r in rpn_bbox_loss:
                output.append(r)
            for r in cls_prob:
                output.append(r)
            for r in bbox_loss:
                output.append(r)
            for r in labels:
                output.append(mx.symbol.BlockGrad(r))
            group = mx.symbol.Group(output)

    else:
        if proposal_type == 'existed_roi':
            if network.startswith('fpn_resnet-'):
                output = [cls_prob[0]]
                for r in cls_prob[1:]:
                    output.append(r)
                for r in bbox_pred:
                    output.append(r)
            group = mx.symbol.Group(output)
        elif proposal_type == 'rpn':
            if network.startswith('fpn_resnet-'):
                rois_return = mx.symbol.Concat(rois[0], rois[1], rois[2], rois[3],dim=0, name='rois')
                cls_prob_all = mx.symbol.Concat(cls_prob[0], cls_prob[1], cls_prob[2], cls_prob[3], dim=1, name='cls_prob_reshape')
                bbox_pred_all = mx.symbol.Concat(bbox_prob[0], bbox_prob[1], bbox_prob[2], bbox_prob[3] ,dim=1, name = 'bbox_pred_reshape')

                output = [rois_return, cls_prob_all, bbox_pred_all]
            group = mx.symbol.Group(output)
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

