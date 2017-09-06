 #-*- coding: UTF-8 -*- 
'''
Created on 2017年7月23日

@author: YeYun
'''
import os,sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)
import time
import numpy as np
import copy
import mxnet as mx
from mxnet.executor_manager import _split_input_slice

from common.processing.image import tensor_vstack

from rpn_get_batch import get_rpn_batch, fpn_assign_anchor
import multiprocessing

class FpnAnchorLoader(mx.io.DataIter):
    def __init__(self, feat_sym, roidb, config, train_end2end=False, batch_size=1, shuffle=False, ctx=None, work_load_list=None,
                 feat_stride=(4, 8, 16, 32, 64), anchor_scales=(8), anchor_ratios=(0.5, 1, 2), allowed_border=1,
                 aspect_grouping=False):
        """
        This Iter will provide roi data to FPN version of Fast R-CNN network
        :param feat_sym: to infer shape of assign_output
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :param aspect_grouping: group images with similar aspects
        :return: AnchorLoader
        """
        super(FpnAnchorLoader, self).__init__()

        # save parameters as properties
        self.feat_sym = feat_sym
        self.roidb = roidb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list
        self.feat_stride = feat_stride
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.allowed_border = allowed_border
        self.aspect_grouping = aspect_grouping
        
        self.config = config
        self.rpn_depth = len(config.RPN_FEAT_STRIDE)
        self.rcnn_depth = len(config.RCNN_FEAT_STRIDE)
        # infer properties from roidb
        self.size = len(roidb)
        self.index = np.arange(self.size)

        # decide data and label names
        if train_end2end:
            self.data_name = ['data', 'im_info', 'gt_boxes', 'feat_shape']
        else:
            self.data_name = ['data', 'im_info', 'feat_shape']
        self.label_name = []
        label_names = []
        bbox_targets = []
        bbox_weights = []
        for i in range(self.rpn_depth):
            label_names.append('label%d'%(i+1))
            bbox_targets.append('bbox_target%d'%(i+1))
            bbox_weights.append('bbox_weight%d'%(i+1))
        self.label_name = label_names
        for bbox_target in bbox_targets:
            self.label_name.append(bbox_target)
        for bbox_weight in bbox_weights:
            self.label_name.append(bbox_weight)
        print 'label_name of fpnAnchor: ', self.label_name
        #self.label_name = ['label1', 'label2', 'label3','label4','bbox_target1', 'bbox_target2','bbox_target3','bbox_target4', 'bbox_weight1', 'bbox_weight2', 'bbox_weight3', 'bbox_weight4']

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch()

    @property
    def provide_data(self):
        #print self.data_name
        return [(k, v.shape) for k, v in zip(self.data_name, self.data)]

    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label)]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            if self.aspect_grouping:
                widths = np.array([r['width'] for r in self.roidb])
                heights = np.array([r['height'] for r in self.roidb])
                horz = (widths >= heights)
                vert = np.logical_not(horz)
                horz_inds = np.where(horz)[0]
                vert_inds = np.where(vert)[0]
                num_horz = len(horz_inds)
                num_vert = len(vert_inds)
                pad_horz = self.batch_size - num_horz % self.batch_size
                pad_vert = self.batch_size - num_vert % self.batch_size
                horz_per = np.random.permutation(horz_inds)
                vert_per = np.random.permutation(vert_inds)
                inds = np.hstack((horz_per, horz_per[-pad_horz:], vert_per, vert_per[-pad_vert:]))
                self.size = inds.shape[0]
                extra = inds.shape[0] % self.batch_size
                inds_ = np.reshape(inds[:-extra], (-1, self.batch_size))
                row_perm = np.random.permutation(np.arange(inds_.shape[0]))
                inds[:-extra] = np.reshape(inds_[row_perm, :], (-1,))
                self.index = inds
            else:
                np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def infer_shape(self, max_data_shape=None, max_label_shape=None):
        """ Return maximum data and label shape for single gpu """
        if max_data_shape is None:
            max_data_shape = []
        if max_label_shape is None:
            max_label_shape = []
        max_shapes = dict(max_data_shape + max_label_shape)
        input_batch_size = max_shapes['data'][0]
        im_info = [[max_shapes['data'][2], max_shapes['data'][3], 1.0]]
        feat_shapes = []
        for feat_sym in self.feat_sym:
            _, feat_shape, _ = feat_sym.infer_shape(**max_shapes)
            feat_shape = [int(i) for i in feat_shape[0]]
            feat_shapes.append(feat_shape)
        label = fpn_assign_anchor(feat_shapes, np.zeros((0, 5)), im_info, self.config,
                              self.feat_stride, self.anchor_scales, self.anchor_ratios, self.allowed_border)
        label = [label[k] for k in self.label_name]
        label_shape = [(k, tuple([input_batch_size] + list(v.shape[1:]))) for k, v in zip(self.label_name, label)]
        return max_data_shape, label_shape

    def get_batch(self):
        # slice roidb
        #t = time.time()
        debug = True
        
            

        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]

        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)

        # get testing data for multigpu
        data_list = []
        label_list = []
        for islice in slices:
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            data, label = get_rpn_batch(iroidb, self.config.SCALES, self.config.input_mean, stride=self.config.IMAGE_STRIDE)#data:image, label:gt_boxes[n,5]:box+label
            data_list.append(data)
            label_list.append(label)

        # pad data first and then assign anchor (read label)
        data_tensor = tensor_vstack([batch['data'] for batch in data_list])
        for data, data_pad in zip(data_list, data_tensor):
            data['data'] = data_pad[np.newaxis, :]
        new_label_list = []
        for data, label in zip(data_list, label_list):#image and gt_bbox
            # infer label shape
            data_shape = {k: v.shape for k, v in data.items()}
            del data_shape['im_info']
            feat_shapes = []
            #print '###################data_shape:',data_shape
            for feat_sym in self.feat_sym:
                _, feat_shape, _ = feat_sym.infer_shape(**data_shape)#get size of feature map for rpn, there are 4 feat_sym in fpn
                feat_shape = [int(i) for i in feat_shape[0]]
                feat_shapes.append(feat_shape)
            #print feat_shapes
            gt_boxes = copy.deepcopy(label['gt_boxes'])
            # assign anchor for label
            #print 'image_info: ', data['im_info']
            #t = time.time()
            feat_shape_ = np.array(feat_shapes[0])
            #print len(feat_shapes)
            for i in range(1, len(feat_shapes)):
                a = np.array(feat_shapes[i])
                feat_shape_=np.vstack((feat_shape_, a))
            #print feat_shape_.shape
            data['feat_shape'] = feat_shape_[np.newaxis,:,:]
            #t_1 = time.time()
            label = fpn_assign_anchor(feat_shapes, label['gt_boxes'], data['im_info'],
                                  self.config,
                                  self.feat_stride, self.anchor_scales,
                                  self.anchor_ratios, self.allowed_border)


            if gt_boxes.shape[0] == 0:
                gt_boxes = -1 * np.ones((1, 5), dtype=np.float32)
            data['gt_boxes'] = gt_boxes[np.newaxis, :, :]#[1,n,5]
            new_label_list.append(label)

        all_data = dict()
        for key in self.data_name:#['data', 'im_info', 'gt_boxes']
            all_data[key] = tensor_vstack([batch[key] for batch in data_list])
        #all_data: data:[4,H,W,C], im_info:[4,1,3], gt_boxes:[4,n,5]
        all_label = dict()
        for key in self.label_name:#['label', 'bbox_target', 'bbox_weight']
            pad = -1 if (key == 'label1' or key == 'label2' or key == 'label3' or key == 'label4' or key == 'label5') else 0
            all_label[key] = tensor_vstack([batch[key] for batch in new_label_list], pad=pad)

        self.data = [mx.nd.array(all_data[key]) for key in self.data_name]
        self.label = [mx.nd.array(all_label[key]) for key in self.label_name]


