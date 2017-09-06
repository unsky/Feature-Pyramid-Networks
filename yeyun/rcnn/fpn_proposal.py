"""
FPN proposal Operator destribute different size of rois to different depth of layers .
"""

import mxnet as mx
import numpy as np
from distutils.util import strtobool

from rcnn_get_batch import destrib_rois

DEBUG = False

class FPNProposalOperator(mx.operator.CustomOp):
    def __init__(self, num_rois):
        super(FPNProposalOperator, self).__init__()
        self._num_rois = num_rois#1000


        if DEBUG:
            self._count = 0
            self._fg_num = 0
            self._bg_num = 0

    def forward(self, is_train, req, in_data, out_data, aux):

        all_rois = in_data[0].asnumpy()#[1000,5]
        #print 'input shape of fpn_proposal: ', all_rois.shape
        #print 1
        #all_rois[all_rois[:,0]!=0,:] = [0,0,0,0,0] #avoid Proposal bug(64stride feature map has no 400 anchor)
        assert np.all(all_rois[:, 0] == 0), 'Only single item batches are supported'



        #print 3
        rois_return = np.zeros((self._num_rois*4, 5), dtype=all_rois.dtype)#[4000,5]
        layer_indexs = destrib_rois(all_rois)#[1000,1]
        #print 'length of layer_indexs: ', len(layer_indexs)

        for i in range(4):
            index = (layer_indexs == (i + 2))
            rois_return[range(self._num_rois*i, self._num_rois*i+sum(index)), :] = all_rois[index, :]


        if DEBUG:
            print "labels=", labels
            print 'num fg: {}'.format((labels > 0).sum())
            print 'num bg: {}'.format((labels == 0).sum())
            self._count += 1
            self._fg_num += (labels > 0).sum()
            self._bg_num += (labels == 0).sum()
            print "self._count=", self._count
            print 'num fg avg: {}'.format(self._fg_num / self._count)
            print 'num bg avg: {}'.format(self._bg_num / self._count)
            print 'ratio: {:.3f}'.format(float(self._fg_num) / float(self._bg_num))
        #print 7
        for ind, val in enumerate([rois_return]):
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)


@mx.operator.register('fpn_proposal')
class FPNProposalProp(mx.operator.CustomOpProp):
    def __init__(self, num_rois):
        super(FPNProposalProp, self).__init__(need_top_grad=False)
        self._num_rois = int(num_rois)


    def list_arguments(self):
        return ['rois']

    def list_outputs(self):
        return ['rois_output']

    def infer_shape(self, in_shape):
        rpn_rois_shape = in_shape[0]#[1000,5]

        output_rois_shape = (self._num_rois*4, 5)#1000*4,5

        return [rpn_rois_shape], \
               [output_rois_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return FPNProposalOperator(self._num_rois)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []

