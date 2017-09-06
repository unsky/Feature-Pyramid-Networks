#-*- coding: UTF-8 -*- 
'''
Created on 2017年3月18日

@author: Crown
'''
import os,sys
from matplotlib.rcsetup import all_backends
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)

import cv2
import numpy as np
import copy

import mxnet as mx

from common.vis.vis_im import draw_bbox, show_text, draw_points
from common.processing.image_roi import  get_roi_images
from common.processing.bbox_transform import bbox_pred, clip_boxes
from common.processing.generate_anchor import generate_base_anchors
from common.processing.image import resize, transform
from common.predict import MutablePredictor
from common.processing.nms import py_nms_wrapper

from rcnn_symbol import get_rcnn_symbol

nms = py_nms_wrapper(0.3)

class RCNNAlterPredictor(MutablePredictor):
    def __init__(self, network, output_folder, epoch, ctx, config, has_json_symbol=False):
        symbol = get_rcnn_symbol(network, config.proposal_type, config.num_classes, config.num_anchors, config)
        prefix = '{0}/model/{0}'.format(output_folder)
        
        
        prefix = '{0}/model/{0}'.format(output_folder)
        input_shapes = [('data', (1, config.input_channel,
                        config.target_size, config.max_size)), ('im_info', (1, 3))]
#         super(RCNNPredictor, self).__init__(symbol, prefix, epoch, input_shapes, provide_label=[('cls_prob_label', (1))], ctx=ctx)
        super(RCNNPredictor, self).__init__(symbol, prefix, epoch, input_shapes, ctx=ctx)


        self.target_size = config.target_size
        self.max_size = config.max_size
        self.feat_stride = config.feat_stride
        
        self.input_mean = config.input_mean
        self.input_scale = config.scale
        
        if config.category == 'all':
            self.classes = ('__background__',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')
        else:
            self.classes = ('__background__',
                            config.category)
        
        
    def rcnn_predict(self, im):        
        return self._rcnn_forward(im)
    
    def vis_det_result(self, im_vis, all_boxes):
        boxes_this_image = [[]] + [all_boxes[j] for j in range(1, len(self.classes))]
        
        for ix, name in enumerate(self.classes):
            draw_bbox(im_vis, boxes_this_image[ix], cls=name)
    
    def save_deploy_symbol(self, name):
        self.symbol.save(name)
    
    def eval_on_voc(self, imdb, roidb, vis=False):
        kps_results = []
        k = 0
        
        num_images = imdb.num_images
        num_classes = len(self.classes)
        
        all_boxes = [[[] for _ in xrange(num_images)]
                     for _ in xrange(num_classes)]
        
        for roi_rec in roidb:
            if k % 100 == 0:
                print '{}/{}'.format(k, len(roidb))
            
            kps_result = []
            image_name = roi_rec['image']
            assert os.path.exists(image_name), image_name + 'not found'
            im = cv2.imread(image_name)
            results_one_frame = self._rcnn_forward(im, thres=0.0)
            for j in range(1, num_classes):
                all_boxes[j][k] = results_one_frame[j]
            if vis:
                im_vis = im.copy()
                self.vis_kps(im_vis, kps_results_one_frame, vis_skeleton=False)
                draw_kps_points(im_vis, roi_rec['keypoints'])
                cv2.imshow('test', im_vis)
                cv2.waitKey()
                
            k += 1
            
        imdb.evaluate_detections(all_boxes)
    
    def _rcnn_forward(self, im, thres=0.5):
        im, im_scale = resize(im, self.target_size, self.max_size)
        im_tensor = transform(im, self.input_mean, self.input_scale)
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
        
        data = [mx.nd.array(im_tensor), mx.nd.array(im_info)] 
        data_shapes = [('data', im_tensor.shape), ('im_info', im_info.shape)]
        data_batch = mx.io.DataBatch(data=data, label=None, provide_data=data_shapes,
                                     provide_label=None)
        output = self.predict(data_batch)
        
        rois = output['rois_output'].asnumpy()[:, 1:]
        
        # save output
        scores = output['cls_prob_reshape_output'].asnumpy()[0]
        bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]
    
        # post processing
        pred_boxes = bbox_pred(rois, bbox_deltas)
        pred_boxes = clip_boxes(pred_boxes, im_tensor.shape[-2:])
    
        # we used scaled image & roi to train, so it is necessary to transform them back
        pred_boxes = pred_boxes / im_scale
        
        return self._post_process(scores, pred_boxes, thres)
    
    def _post_process(self, scores, pred_boxes, thres):
        
        all_boxes = [[] for cls in self.classes]
        for ix, cls in enumerate(self.classes):
            cls_boxes = pred_boxes[:, 4 * ix:4 * (ix + 1)]
            cls_scores = scores[:, ix, np.newaxis]
            keep = np.where(cls_scores >= thres)[0]
            dets = np.hstack((cls_boxes, cls_scores)).astype(np.float32)[keep, :]
            keep = nms(dets)
            all_boxes[ix] = dets[keep, :]
        
        return all_boxes
    