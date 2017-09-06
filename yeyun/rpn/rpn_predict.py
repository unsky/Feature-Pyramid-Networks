#-*- coding: UTF-8 -*- 
'''
Created on 2017年7月23日

@author: Yeyun
'''
import os,sys
from matplotlib.rcsetup import all_backends
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)
import time
import cv2
import numpy as np
import copy

import mxnet as mx
import pprint
from common.vis.vis_im import draw_bbox, show_text, draw_points
from common.processing.image_roi import  get_roi_images
from common.processing.bbox_transform import bbox_pred, clip_boxes
from common.processing.generate_anchor import generate_base_anchors
from common.processing.image import resize, transform
from common.predict import MutablePredictor
from common.processing.nms import py_nms_wrapper

#from rcnn_symbol_fpn import get_rcnn_symbol
from rpn.rpn_symbol_fpn import get_rpn_symbol

nms = py_nms_wrapper(0.3)

class RPNPredictor(MutablePredictor):
    def __init__(self, network, output_folder, epoch, ctx, config, has_json_symbol=False, fpn=False):
        symbol = get_rpn_symbol(network, config.proposal_type, config.num_classes, config.num_anchors, config)
        self.proposal_type = config.proposal_type
        self.feat_sym = []
        if fpn:
            for i in range(len(config.RPN_FEAT_STRIDE)):
                inter_feat_sym = symbol.get_internals()["rpn_cls_score_p%d_output"%(i+2)]
                (self.feat_sym).append(inter_feat_sym)
             
        if self.proposal_type == 'rpn':
            prefix = '{0}/model/{0}'.format(output_folder)
            input_shapes = [('data', (1, config.input_channel,
                            config.target_size, config.max_size)), ('im_info', (1, 3)), ('feat_shape', (1, len(config.RPN_FEAT_STRIDE), 4))]
        else:
            assert False, 'unsupported proposal type'
        super(RPNPredictor, self).__init__(symbol, prefix, epoch, input_shapes, ctx=ctx)

        #pprint.pprint(symbol.list_outputs())
        self.target_size = config.target_size
        self.max_size = config.max_size
        self.feat_stride = config.feat_stride
        self.image_stride = config.IMAGE_STRIDE
        self.input_mean = config.input_mean
        self.input_scale = config.scale
        if config.rpn_cls_agnostic == False:
            self.classes = ('__background__',
                            'fore_ground')
        elif config.category == 'all':
            if config.eval_tool=='voc':
                self.classes = ('__background__',
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')
            elif config.eval_tool=='coco':
                self.classes = ('__background__','person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush')
        else:
            self.classes = ('__background__',
                            config.category)
        
        
    def rpn_predict(self, im, rois=None):        
        return self._rpn_forward(im, rois)
    
    def vis_det_result(self, im_vis, all_boxes):
        boxes_this_image = [[]] + [all_boxes[j] for j in range(1, len(self.classes))]
        
        for ix, name in enumerate(self.classes):
            draw_bbox(im_vis, boxes_this_image[ix], cls=name)
    
    def save_deploy_symbol(self, name):
        self.symbol.save(name)
    
    def imdb_eval(self, imdb, roidb, rpn_predictor=None, vis=False):
        k = 0
        num_images = imdb.num_images
        imdb_boxes = list()
        for roi_rec in roidb:
            if k % 100 == 0:
                print '{}/{}'.format(k, len(roidb))
            image_name = roi_rec['image']
            assert os.path.exists(image_name), image_name + 'not found'
            im = cv2.imread(image_name)
            results_one_frame = self._rpn_forward(im, thres=1e-3)      
            imdb_boxes.append(results_one_frame)            
        imdb.evaluate_recall(roidb, candidate_boxes=imdb_boxes)
    
    def _rpn_forward(self, im, rois=None, thres=0.5):
        debug = False
        im, im_scale = resize(im, self.target_size, self.max_size, self.image_stride)
        im_tensor = transform(im, self.input_mean, self.input_scale)
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
        
        if len(self.feat_sym) == 0:
            data_shapes = [('data', im_tensor.shape), ('im_info', im_info.shape)]
            data = [mx.nd.array(im_tensor), mx.nd.array(im_info)] 
        else:
            data_shape = {'data': im_tensor.shape}
            feat_shapes = []
            for feat in self.feat_sym:
                _, feat_shape, _ = feat.infer_shape(**data_shape)#get size of feature map for rpn, there are 4 feat_sym in fpn
                feat_shape = [int(i) for i in feat_shape[0]]
                feat_shapes.append(feat_shape)
            feat_shape_ = np.array(feat_shapes[0])
            #print len(feat_shapes)
            for i in range(1, len(feat_shapes)):
                a = np.array(feat_shapes[i])
                feat_shape_=np.vstack((feat_shape_, a))#[5,4]
            #print feat_shape_.shape
            final_feat_shape = feat_shape_[np.newaxis,:,:]#[1,5,4]
            data_shapes = [('data', im_tensor.shape), ('im_info', im_info.shape), ('feat_shape', final_feat_shape.shape)]
            data = [mx.nd.array(im_tensor), mx.nd.array(im_info), mx.nd.array(final_feat_shape)] 
        data_batch = mx.io.DataBatch(data=data, label=None, provide_data=data_shapes,
                                     provide_label=None)
        t = time.time()
        output = self.predict(data_batch)
        
        rois = output['rois_output'].asnumpy()[:, 1:] / im_scale
        scores = output['rois_score'].asnumpy()
        if debug:
            f2 = open('predict.txt', 'w')
            roid_rpn = output['rois_output'].asnumpy()#1200,5
            roid_pred = roid_rpn
            for i in range(roid_pred.shape[0]):
                w = max(0, int(roid_pred[i,3]-roid_pred[i,1]))
                h = max(0, int(roid_pred[i,4]-roid_pred[i,2]))
                s = w*h
                print im.shape
                if w < 50 or h < 300 or w > im.shape[1] or h > im.shape[0] or w > h:
                    continue
                cv2.rectangle(im, (int(roid_pred[i,1]), int(roid_pred[i,2])), (int(roid_pred[i,3]), int(roid_pred[i,4])), (255,0,0), 1)
                f2.write(str(roid_rpn[i,:])+'\n')
            cv2.imwrite('rpn_result.jpg', im)

        # save output
        t1 = time.time() - t
        # post processing
        
        # we used scaled image & roi to train, so it is necessary to transform them back
        dets = np.hstack(rois, scores)
        
        return self._post_process(dets, thres)
    
    def _post_process(self, dets, thres):
        keep = np.where(dets[:,4:] > thres)[0]
        dets = dets[keep, :] 
        return dets
    
