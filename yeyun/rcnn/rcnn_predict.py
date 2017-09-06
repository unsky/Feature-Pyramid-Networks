
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
from rcnn.rcnn_symbol_fpn_corr import get_rcnn_symbol

nms = py_nms_wrapper(0.3)

class RCNNPredictor(MutablePredictor):
    def __init__(self, network, output_folder, epoch, ctx, config, has_json_symbol=False, fpn=False, step='rcnn'):
        symbol = get_rcnn_symbol(network, config.proposal_type, config.num_classes, config.num_anchors, config)
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
        elif self.proposal_type == 'existed_roi':
            prefix = '{0}/model/{0}_rcnn'.format(output_folder)
            input_shapes = [('data', (1, config.input_channel, config.target_size, config.max_size)),
                             ('rois', (1, config.TEST.RPN_POST_NMS_TOP_N, 5))]
        else:
            assert False, 'unsupported proposal type'
        super(RCNNPredictor, self).__init__(symbol, prefix, epoch, input_shapes, ctx=ctx)

        #pprint.pprint(symbol.list_outputs())
        self.target_size = config.target_size
        self.max_size = config.max_size
        self.feat_stride = config.feat_stride
        self.image_stride = config.IMAGE_STRIDE
        self.input_mean = config.input_mean
        self.input_scale = config.scale
        
        if config.category == 'all':
            if config.eval_tool=='voc':
                self.classes = ('__background__',
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')
            if config.eval_tool=='coco':
                self.classes = ('__background__','person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush')
            if config.eval_tool=='kitti':
                self.classes = ('__background__','Car','Truck', 'Tram')
        else:
            self.classes = ('__background__',
                            config.category)
        
        
    def rcnn_predict(self, im, rois=None):        
        return self._rcnn_forward(im, rois)
    
    def vis_det_result(self, im_vis, all_boxes):
        boxes_this_image = [[]] + [all_boxes[j] for j in range(1, len(self.classes))]
        
        for ix, name in enumerate(self.classes):
            draw_bbox(im_vis, boxes_this_image[ix], cls=name)
    
    def save_deploy_symbol(self, name):
        self.symbol.save(name)
    
    def imdb_eval(self, imdb, roidb, rpn_predictor=None, vis=False):
        kps_results = []
        k = 0
        
        num_images = imdb.num_images
        num_classes = len(self.classes)
        #print 'number of images: ', num_images
        #print 'number of classes: ', num_classes
        all_boxes = [[[] for _ in xrange(num_images)]
                     for _ in xrange(num_classes)]
        
        for roi_rec in roidb:
            if k % 100 == 0:
                print '{}/{}'.format(k, len(roidb))
            
            kps_result = []
            image_name = roi_rec['image']
            assert os.path.exists(image_name), image_name + 'not found'
            im = cv2.imread(image_name)
            
            if self.proposal_type == 'rpn':
                results_one_frame = self._rcnn_forward(im, thres=1e-3)
            elif self.proposal_type == 'existed_roi':
                assert rpn_predictor is not None
                rpn_rois_one_frame = rpn_predictor.rpn_predict(im)
                results_one_frame = self._rcnn_forward(im, rpn_rois_one_frame, thres=1e-3)
            #print 'shape of all_boxes:,', all_boxes.shape
            #print 'len of results_one_frame: ', len(results_one_frame)                    
            for j in range(1, num_classes):
                all_boxes[j][k] = results_one_frame[j]
                
            k += 1
            
        imdb.evaluate_detections(all_boxes)
    
    def _rcnn_forward(self, im, rois=None, thres=0.5):
        debug = False
        im, im_scale = resize(im, self.target_size, self.max_size, self.image_stride)
        im_tensor = transform(im, self.input_mean, self.input_scale)
        if self.proposal_type == 'rpn':
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
        elif self.proposal_type == 'existed_roi':
            assert rois is not None
            rois = rois.reshape(1, -1, 5)
            data = [mx.nd.array(im_tensor), mx.nd.array(rois)]
            data_shapes = [('data', im_tensor.shape), ('rois', rois.shape)]
        
        data_batch = mx.io.DataBatch(data=data, label=None, provide_data=data_shapes,
                                     provide_label=None)
        t = time.time()
        output = self.predict(data_batch)
        
        if self.proposal_type == 'rpn':
            rois = output['rois_output'].asnumpy()[:, 1:]
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
        elif self.proposal_type == 'existed_roi':
            rois = rois[0][:, 1:]
        
        # save output
        scores = output['cls_prob_reshape_output'].asnumpy()[0]
        bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]
        t1 = time.time() - t
        # post processing
        print 'predict :{:.4f}s'.format(t1)
        if 0:
            for i in range(rois.shape[0]):
                f2.write('rois: ' + str(i)+'    :   '+str(rois[i,:]) + '#############bbox_deltas: ' +  str(bbox_deltas[i,4:]) + '#################cls_pred: '+ str(scores[i,1])+'\n')
        person_score = scores[:,1]
        max_score = max(person_score)
        print max_score
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
    
