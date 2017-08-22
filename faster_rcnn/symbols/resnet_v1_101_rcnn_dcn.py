# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Guodong Zhang
# --------------------------------------------------------

import cPickle
import mxnet as mx
from utils.symbol import Symbol
from operator_py.proposal import *
from operator_py.proposal_target import *
from operator_py.box_annotator_ohem import *


class resnet_v1_101_rcnn_dcn(Symbol):
    def __init__(self):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 1e-5
        self.use_global_stats = True
        self.workspace = 512
        self.units = (3, 4, 23, 3)  # use for 101
        self.filter_list = [256, 512, 1024, 2048]

    def get_resnet_v1_conv2(self, data):
        conv1 = mx.symbol.Convolution(name='conv1', data=data, num_filter=64, pad=(3, 3), kernel=(7, 7), stride=(2, 2),
                                      no_bias=True)
        bn_conv1 = mx.symbol.BatchNorm(name='bn_conv1', data=conv1, use_global_stats=True, fix_gamma=False,
                                       eps=self.eps)
        scale_conv1 = bn_conv1
        conv1_relu = mx.symbol.Activation(name='conv1_relu', data=scale_conv1, act_type='relu')
        pool1 = mx.symbol.Pooling(name='pool1', data=conv1_relu, pooling_convention='full', pad=(0, 0), kernel=(3, 3),
                                  stride=(2, 2), pool_type='max')
        res2a_branch1 = mx.symbol.Convolution(name='res2a_branch1', data=pool1, num_filter=256, pad=(0, 0),
                                              kernel=(1, 1),
                                              stride=(1, 1), no_bias=True)
        bn2a_branch1 = mx.symbol.BatchNorm(name='bn2a_branch1', data=res2a_branch1, use_global_stats=True,
                                           fix_gamma=False, eps=self.eps)
        scale2a_branch1 = bn2a_branch1
        res2a_branch2a = mx.symbol.Convolution(name='res2a_branch2a', data=pool1, num_filter=64, pad=(0, 0),
                                               kernel=(1, 1),
                                               stride=(1, 1), no_bias=True)
        bn2a_branch2a = mx.symbol.BatchNorm(name='bn2a_branch2a', data=res2a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2a_branch2a = bn2a_branch2a
        res2a_branch2a_relu = mx.symbol.Activation(name='res2a_branch2a_relu', data=scale2a_branch2a, act_type='relu')
        res2a_branch2b = mx.symbol.Convolution(name='res2a_branch2b', data=res2a_branch2a_relu, num_filter=64,
                                               pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2a_branch2b = mx.symbol.BatchNorm(name='bn2a_branch2b', data=res2a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2a_branch2b = bn2a_branch2b
        res2a_branch2b_relu = mx.symbol.Activation(name='res2a_branch2b_relu', data=scale2a_branch2b, act_type='relu')
        res2a_branch2c = mx.symbol.Convolution(name='res2a_branch2c', data=res2a_branch2b_relu, num_filter=256,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2a_branch2c = mx.symbol.BatchNorm(name='bn2a_branch2c', data=res2a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2a_branch2c = bn2a_branch2c
        res2a = mx.symbol.broadcast_add(name='res2a', *[scale2a_branch1, scale2a_branch2c])
        res2a_relu = mx.symbol.Activation(name='res2a_relu', data=res2a, act_type='relu')
        res2b_branch2a = mx.symbol.Convolution(name='res2b_branch2a', data=res2a_relu, num_filter=64, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2b_branch2a = mx.symbol.BatchNorm(name='bn2b_branch2a', data=res2b_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2b_branch2a = bn2b_branch2a
        res2b_branch2a_relu = mx.symbol.Activation(name='res2b_branch2a_relu', data=scale2b_branch2a, act_type='relu')
        res2b_branch2b = mx.symbol.Convolution(name='res2b_branch2b', data=res2b_branch2a_relu, num_filter=64,
                                               pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2b_branch2b = mx.symbol.BatchNorm(name='bn2b_branch2b', data=res2b_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2b_branch2b = bn2b_branch2b
        res2b_branch2b_relu = mx.symbol.Activation(name='res2b_branch2b_relu', data=scale2b_branch2b, act_type='relu')
        res2b_branch2c = mx.symbol.Convolution(name='res2b_branch2c', data=res2b_branch2b_relu, num_filter=256,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2b_branch2c = mx.symbol.BatchNorm(name='bn2b_branch2c', data=res2b_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2b_branch2c = bn2b_branch2c
        res2b = mx.symbol.broadcast_add(name='res2b', *[res2a_relu, scale2b_branch2c])
        res2b_relu = mx.symbol.Activation(name='res2b_relu', data=res2b, act_type='relu')
        res2c_branch2a = mx.symbol.Convolution(name='res2c_branch2a', data=res2b_relu, num_filter=64, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2c_branch2a = mx.symbol.BatchNorm(name='bn2c_branch2a', data=res2c_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2c_branch2a = bn2c_branch2a
        res2c_branch2a_relu = mx.symbol.Activation(name='res2c_branch2a_relu', data=scale2c_branch2a, act_type='relu')
        res2c_branch2b = mx.symbol.Convolution(name='res2c_branch2b', data=res2c_branch2a_relu, num_filter=64,
                                               pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2c_branch2b = mx.symbol.BatchNorm(name='bn2c_branch2b', data=res2c_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2c_branch2b = bn2c_branch2b
        res2c_branch2b_relu = mx.symbol.Activation(name='res2c_branch2b_relu', data=scale2c_branch2b, act_type='relu')
        res2c_branch2c = mx.symbol.Convolution(name='res2c_branch2c', data=res2c_branch2b_relu, num_filter=256,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2c_branch2c = mx.symbol.BatchNorm(name='bn2c_branch2c', data=res2c_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2c_branch2c = bn2c_branch2c
        res2c = mx.symbol.broadcast_add(name='res2c', *[res2b_relu, scale2c_branch2c])
        res2c_relu = mx.symbol.Activation(name='res2c_relu', data=res2c, act_type='relu')
        return res2c_relu

    def get_resnet_v1_conv3(self, conv2):
        res3a_branch1 = mx.symbol.Convolution(name='res3a_branch1', data=conv2, num_filter=512, pad=(0, 0),
                                              kernel=(1, 1), stride=(2, 2), no_bias=True) 
        bn3a_branch1 = mx.symbol.BatchNorm(name='bn3a_branch1', data=res3a_branch1, use_global_stats=True,
                                           fix_gamma=False, eps=self.eps)
        scale3a_branch1 = bn3a_branch1
        res3a_branch2a = mx.symbol.Convolution(name='res3a_branch2a', data=conv2, num_filter=128, pad=(0, 0),
                                               kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn3a_branch2a = mx.symbol.BatchNorm(name='bn3a_branch2a', data=res3a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale3a_branch2a = bn3a_branch2a
        res3a_branch2a_relu = mx.symbol.Activation(name='res3a_branch2a_relu', data=scale3a_branch2a, act_type='relu')
        res3a_branch2b = mx.symbol.Convolution(name='res3a_branch2b', data=res3a_branch2a_relu, num_filter=128,
                                               pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3a_branch2b = mx.symbol.BatchNorm(name='bn3a_branch2b', data=res3a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale3a_branch2b = bn3a_branch2b
        res3a_branch2b_relu = mx.symbol.Activation(name='res3a_branch2b_relu', data=scale3a_branch2b, act_type='relu')
        res3a_branch2c = mx.symbol.Convolution(name='res3a_branch2c', data=res3a_branch2b_relu, num_filter=512,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3a_branch2c = mx.symbol.BatchNorm(name='bn3a_branch2c', data=res3a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale3a_branch2c = bn3a_branch2c
        res3a = mx.symbol.broadcast_add(name='res3a', *[scale3a_branch1, scale3a_branch2c])
        res3a_relu = mx.symbol.Activation(name='res3a_relu', data=res3a, act_type='relu')
        res3b1_branch2a = mx.symbol.Convolution(name='res3b1_branch2a', data=res3a_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b1_branch2a = mx.symbol.BatchNorm(name='bn3b1_branch2a', data=res3b1_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b1_branch2a = bn3b1_branch2a
        res3b1_branch2a_relu = mx.symbol.Activation(name='res3b1_branch2a_relu', data=scale3b1_branch2a,
                                                    act_type='relu')
        res3b1_branch2b = mx.symbol.Convolution(name='res3b1_branch2b', data=res3b1_branch2a_relu, num_filter=128,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3b1_branch2b = mx.symbol.BatchNorm(name='bn3b1_branch2b', data=res3b1_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b1_branch2b = bn3b1_branch2b
        res3b1_branch2b_relu = mx.symbol.Activation(name='res3b1_branch2b_relu', data=scale3b1_branch2b,
                                                    act_type='relu')
        res3b1_branch2c = mx.symbol.Convolution(name='res3b1_branch2c', data=res3b1_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b1_branch2c = mx.symbol.BatchNorm(name='bn3b1_branch2c', data=res3b1_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b1_branch2c = bn3b1_branch2c
        res3b1 = mx.symbol.broadcast_add(name='res3b1', *[res3a_relu, scale3b1_branch2c])
        res3b1_relu = mx.symbol.Activation(name='res3b1_relu', data=res3b1, act_type='relu')
        res3b2_branch2a = mx.symbol.Convolution(name='res3b2_branch2a', data=res3b1_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b2_branch2a = mx.symbol.BatchNorm(name='bn3b2_branch2a', data=res3b2_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b2_branch2a = bn3b2_branch2a
        res3b2_branch2a_relu = mx.symbol.Activation(name='res3b2_branch2a_relu', data=scale3b2_branch2a,
                                                    act_type='relu')
        res3b2_branch2b = mx.symbol.Convolution(name='res3b2_branch2b', data=res3b2_branch2a_relu, num_filter=128,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3b2_branch2b = mx.symbol.BatchNorm(name='bn3b2_branch2b', data=res3b2_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b2_branch2b = bn3b2_branch2b
        res3b2_branch2b_relu = mx.symbol.Activation(name='res3b2_branch2b_relu', data=scale3b2_branch2b,
                                                    act_type='relu')
        res3b2_branch2c = mx.symbol.Convolution(name='res3b2_branch2c', data=res3b2_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b2_branch2c = mx.symbol.BatchNorm(name='bn3b2_branch2c', data=res3b2_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b2_branch2c = bn3b2_branch2c
        res3b2 = mx.symbol.broadcast_add(name='res3b2', *[res3b1_relu, scale3b2_branch2c])
        res3b2_relu = mx.symbol.Activation(name='res3b2_relu', data=res3b2, act_type='relu')
        res3b3_branch2a = mx.symbol.Convolution(name='res3b3_branch2a', data=res3b2_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b3_branch2a = mx.symbol.BatchNorm(name='bn3b3_branch2a', data=res3b3_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b3_branch2a = bn3b3_branch2a
        res3b3_branch2a_relu = mx.symbol.Activation(name='res3b3_branch2a_relu', data=scale3b3_branch2a,
                                                    act_type='relu')
        res3b3_branch2b = mx.symbol.Convolution(name='res3b3_branch2b', data=res3b3_branch2a_relu, num_filter=128,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3b3_branch2b = mx.symbol.BatchNorm(name='bn3b3_branch2b', data=res3b3_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b3_branch2b = bn3b3_branch2b
        res3b3_branch2b_relu = mx.symbol.Activation(name='res3b3_branch2b_relu', data=scale3b3_branch2b,
                                                    act_type='relu')
        res3b3_branch2c = mx.symbol.Convolution(name='res3b3_branch2c', data=res3b3_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b3_branch2c = mx.symbol.BatchNorm(name='bn3b3_branch2c', data=res3b3_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b3_branch2c = bn3b3_branch2c
        res3b3 = mx.symbol.broadcast_add(name='res3b3', *[res3b2_relu, scale3b3_branch2c])
        res3b3_relu = mx.symbol.Activation(name='res3b3_relu', data=res3b3, act_type='relu')
        return  res3b3_relu  



    def get_resnet_v1_conv4(self, conv3):
        res4a_branch1 = mx.symbol.Convolution(name='res4a_branch1', data=conv3, num_filter=1024, pad=(0, 0),
                                              kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn4a_branch1 = mx.symbol.BatchNorm(name='bn4a_branch1', data=res4a_branch1, use_global_stats=True,
                                           fix_gamma=False, eps=self.eps)
        scale4a_branch1 = bn4a_branch1
        res4a_branch2a = mx.symbol.Convolution(name='res4a_branch2a', data=conv3, num_filter=256, pad=(0, 0),
                                               kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn4a_branch2a = mx.symbol.BatchNorm(name='bn4a_branch2a', data=res4a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4a_branch2a = bn4a_branch2a
        res4a_branch2a_relu = mx.symbol.Activation(name='res4a_branch2a_relu', data=scale4a_branch2a, act_type='relu')
        res4a_branch2b = mx.symbol.Convolution(name='res4a_branch2b', data=res4a_branch2a_relu, num_filter=256,
                                               pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4a_branch2b = mx.symbol.BatchNorm(name='bn4a_branch2b', data=res4a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4a_branch2b = bn4a_branch2b
        res4a_branch2b_relu = mx.symbol.Activation(name='res4a_branch2b_relu', data=scale4a_branch2b, act_type='relu')
        res4a_branch2c = mx.symbol.Convolution(name='res4a_branch2c', data=res4a_branch2b_relu, num_filter=1024,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4a_branch2c = mx.symbol.BatchNorm(name='bn4a_branch2c', data=res4a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4a_branch2c = bn4a_branch2c
        res4a = mx.symbol.broadcast_add(name='res4a', *[scale4a_branch1, scale4a_branch2c])
        res4a_relu = mx.symbol.Activation(name='res4a_relu', data=res4a, act_type='relu')
        res4b1_branch2a = mx.symbol.Convolution(name='res4b1_branch2a', data=res4a_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b1_branch2a = mx.symbol.BatchNorm(name='bn4b1_branch2a', data=res4b1_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b1_branch2a = bn4b1_branch2a
        res4b1_branch2a_relu = mx.symbol.Activation(name='res4b1_branch2a_relu', data=scale4b1_branch2a,
                                                    act_type='relu')
        res4b1_branch2b = mx.symbol.Convolution(name='res4b1_branch2b', data=res4b1_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b1_branch2b = mx.symbol.BatchNorm(name='bn4b1_branch2b', data=res4b1_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b1_branch2b = bn4b1_branch2b
        res4b1_branch2b_relu = mx.symbol.Activation(name='res4b1_branch2b_relu', data=scale4b1_branch2b,
                                                    act_type='relu')
        res4b1_branch2c = mx.symbol.Convolution(name='res4b1_branch2c', data=res4b1_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b1_branch2c = mx.symbol.BatchNorm(name='bn4b1_branch2c', data=res4b1_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b1_branch2c = bn4b1_branch2c
        res4b1 = mx.symbol.broadcast_add(name='res4b1', *[res4a_relu, scale4b1_branch2c])
        res4b1_relu = mx.symbol.Activation(name='res4b1_relu', data=res4b1, act_type='relu')
        res4b2_branch2a = mx.symbol.Convolution(name='res4b2_branch2a', data=res4b1_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b2_branch2a = mx.symbol.BatchNorm(name='bn4b2_branch2a', data=res4b2_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b2_branch2a = bn4b2_branch2a
        res4b2_branch2a_relu = mx.symbol.Activation(name='res4b2_branch2a_relu', data=scale4b2_branch2a,
                                                    act_type='relu')
        res4b2_branch2b = mx.symbol.Convolution(name='res4b2_branch2b', data=res4b2_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b2_branch2b = mx.symbol.BatchNorm(name='bn4b2_branch2b', data=res4b2_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b2_branch2b = bn4b2_branch2b
        res4b2_branch2b_relu = mx.symbol.Activation(name='res4b2_branch2b_relu', data=scale4b2_branch2b,
                                                    act_type='relu')
        res4b2_branch2c = mx.symbol.Convolution(name='res4b2_branch2c', data=res4b2_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b2_branch2c = mx.symbol.BatchNorm(name='bn4b2_branch2c', data=res4b2_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b2_branch2c = bn4b2_branch2c
        res4b2 = mx.symbol.broadcast_add(name='res4b2', *[res4b1_relu, scale4b2_branch2c])
        res4b2_relu = mx.symbol.Activation(name='res4b2_relu', data=res4b2, act_type='relu')
        res4b3_branch2a = mx.symbol.Convolution(name='res4b3_branch2a', data=res4b2_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b3_branch2a = mx.symbol.BatchNorm(name='bn4b3_branch2a', data=res4b3_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b3_branch2a = bn4b3_branch2a
        res4b3_branch2a_relu = mx.symbol.Activation(name='res4b3_branch2a_relu', data=scale4b3_branch2a,
                                                    act_type='relu')
        res4b3_branch2b = mx.symbol.Convolution(name='res4b3_branch2b', data=res4b3_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b3_branch2b = mx.symbol.BatchNorm(name='bn4b3_branch2b', data=res4b3_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b3_branch2b = bn4b3_branch2b
        res4b3_branch2b_relu = mx.symbol.Activation(name='res4b3_branch2b_relu', data=scale4b3_branch2b,
                                                    act_type='relu')
        res4b3_branch2c = mx.symbol.Convolution(name='res4b3_branch2c', data=res4b3_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b3_branch2c = mx.symbol.BatchNorm(name='bn4b3_branch2c', data=res4b3_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b3_branch2c = bn4b3_branch2c
        res4b3 = mx.symbol.broadcast_add(name='res4b3', *[res4b2_relu, scale4b3_branch2c])
        res4b3_relu = mx.symbol.Activation(name='res4b3_relu', data=res4b3, act_type='relu')
        res4b4_branch2a = mx.symbol.Convolution(name='res4b4_branch2a', data=res4b3_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b4_branch2a = mx.symbol.BatchNorm(name='bn4b4_branch2a', data=res4b4_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b4_branch2a = bn4b4_branch2a
        res4b4_branch2a_relu = mx.symbol.Activation(name='res4b4_branch2a_relu', data=scale4b4_branch2a,
                                                    act_type='relu')
        res4b4_branch2b = mx.symbol.Convolution(name='res4b4_branch2b', data=res4b4_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b4_branch2b = mx.symbol.BatchNorm(name='bn4b4_branch2b', data=res4b4_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b4_branch2b = bn4b4_branch2b
        res4b4_branch2b_relu = mx.symbol.Activation(name='res4b4_branch2b_relu', data=scale4b4_branch2b,
                                                    act_type='relu')
        res4b4_branch2c = mx.symbol.Convolution(name='res4b4_branch2c', data=res4b4_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b4_branch2c = mx.symbol.BatchNorm(name='bn4b4_branch2c', data=res4b4_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b4_branch2c = bn4b4_branch2c
        res4b4 = mx.symbol.broadcast_add(name='res4b4', *[res4b3_relu, scale4b4_branch2c])
        res4b4_relu = mx.symbol.Activation(name='res4b4_relu', data=res4b4, act_type='relu')
        res4b5_branch2a = mx.symbol.Convolution(name='res4b5_branch2a', data=res4b4_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b5_branch2a = mx.symbol.BatchNorm(name='bn4b5_branch2a', data=res4b5_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b5_branch2a = bn4b5_branch2a
        res4b5_branch2a_relu = mx.symbol.Activation(name='res4b5_branch2a_relu', data=scale4b5_branch2a,
                                                    act_type='relu')
        res4b5_branch2b = mx.symbol.Convolution(name='res4b5_branch2b', data=res4b5_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b5_branch2b = mx.symbol.BatchNorm(name='bn4b5_branch2b', data=res4b5_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b5_branch2b = bn4b5_branch2b
        res4b5_branch2b_relu = mx.symbol.Activation(name='res4b5_branch2b_relu', data=scale4b5_branch2b,
                                                    act_type='relu')
        res4b5_branch2c = mx.symbol.Convolution(name='res4b5_branch2c', data=res4b5_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b5_branch2c = mx.symbol.BatchNorm(name='bn4b5_branch2c', data=res4b5_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b5_branch2c = bn4b5_branch2c
        res4b5 = mx.symbol.broadcast_add(name='res4b5', *[res4b4_relu, scale4b5_branch2c])
        res4b5_relu = mx.symbol.Activation(name='res4b5_relu', data=res4b5, act_type='relu')
        res4b6_branch2a = mx.symbol.Convolution(name='res4b6_branch2a', data=res4b5_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b6_branch2a = mx.symbol.BatchNorm(name='bn4b6_branch2a', data=res4b6_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b6_branch2a = bn4b6_branch2a
        res4b6_branch2a_relu = mx.symbol.Activation(name='res4b6_branch2a_relu', data=scale4b6_branch2a,
                                                    act_type='relu')
        res4b6_branch2b = mx.symbol.Convolution(name='res4b6_branch2b', data=res4b6_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b6_branch2b = mx.symbol.BatchNorm(name='bn4b6_branch2b', data=res4b6_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b6_branch2b = bn4b6_branch2b
        res4b6_branch2b_relu = mx.symbol.Activation(name='res4b6_branch2b_relu', data=scale4b6_branch2b,
                                                    act_type='relu')
        res4b6_branch2c = mx.symbol.Convolution(name='res4b6_branch2c', data=res4b6_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b6_branch2c = mx.symbol.BatchNorm(name='bn4b6_branch2c', data=res4b6_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b6_branch2c = bn4b6_branch2c
        res4b6 = mx.symbol.broadcast_add(name='res4b6', *[res4b5_relu, scale4b6_branch2c])
        res4b6_relu = mx.symbol.Activation(name='res4b6_relu', data=res4b6, act_type='relu')
        res4b7_branch2a = mx.symbol.Convolution(name='res4b7_branch2a', data=res4b6_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b7_branch2a = mx.symbol.BatchNorm(name='bn4b7_branch2a', data=res4b7_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b7_branch2a = bn4b7_branch2a
        res4b7_branch2a_relu = mx.symbol.Activation(name='res4b7_branch2a_relu', data=scale4b7_branch2a,
                                                    act_type='relu')
        res4b7_branch2b = mx.symbol.Convolution(name='res4b7_branch2b', data=res4b7_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b7_branch2b = mx.symbol.BatchNorm(name='bn4b7_branch2b', data=res4b7_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b7_branch2b = bn4b7_branch2b
        res4b7_branch2b_relu = mx.symbol.Activation(name='res4b7_branch2b_relu', data=scale4b7_branch2b,
                                                    act_type='relu')
        res4b7_branch2c = mx.symbol.Convolution(name='res4b7_branch2c', data=res4b7_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b7_branch2c = mx.symbol.BatchNorm(name='bn4b7_branch2c', data=res4b7_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b7_branch2c = bn4b7_branch2c
        res4b7 = mx.symbol.broadcast_add(name='res4b7', *[res4b6_relu, scale4b7_branch2c])
        res4b7_relu = mx.symbol.Activation(name='res4b7_relu', data=res4b7, act_type='relu')
        res4b8_branch2a = mx.symbol.Convolution(name='res4b8_branch2a', data=res4b7_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b8_branch2a = mx.symbol.BatchNorm(name='bn4b8_branch2a', data=res4b8_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b8_branch2a = bn4b8_branch2a
        res4b8_branch2a_relu = mx.symbol.Activation(name='res4b8_branch2a_relu', data=scale4b8_branch2a,
                                                    act_type='relu')
        res4b8_branch2b = mx.symbol.Convolution(name='res4b8_branch2b', data=res4b8_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b8_branch2b = mx.symbol.BatchNorm(name='bn4b8_branch2b', data=res4b8_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b8_branch2b = bn4b8_branch2b
        res4b8_branch2b_relu = mx.symbol.Activation(name='res4b8_branch2b_relu', data=scale4b8_branch2b,
                                                    act_type='relu')
        res4b8_branch2c = mx.symbol.Convolution(name='res4b8_branch2c', data=res4b8_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b8_branch2c = mx.symbol.BatchNorm(name='bn4b8_branch2c', data=res4b8_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b8_branch2c = bn4b8_branch2c
        res4b8 = mx.symbol.broadcast_add(name='res4b8', *[res4b7_relu, scale4b8_branch2c])
        res4b8_relu = mx.symbol.Activation(name='res4b8_relu', data=res4b8, act_type='relu')
        res4b9_branch2a = mx.symbol.Convolution(name='res4b9_branch2a', data=res4b8_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b9_branch2a = mx.symbol.BatchNorm(name='bn4b9_branch2a', data=res4b9_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b9_branch2a = bn4b9_branch2a
        res4b9_branch2a_relu = mx.symbol.Activation(name='res4b9_branch2a_relu', data=scale4b9_branch2a,
                                                    act_type='relu')
        res4b9_branch2b = mx.symbol.Convolution(name='res4b9_branch2b', data=res4b9_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b9_branch2b = mx.symbol.BatchNorm(name='bn4b9_branch2b', data=res4b9_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b9_branch2b = bn4b9_branch2b
        res4b9_branch2b_relu = mx.symbol.Activation(name='res4b9_branch2b_relu', data=scale4b9_branch2b,
                                                    act_type='relu')
        res4b9_branch2c = mx.symbol.Convolution(name='res4b9_branch2c', data=res4b9_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b9_branch2c = mx.symbol.BatchNorm(name='bn4b9_branch2c', data=res4b9_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b9_branch2c = bn4b9_branch2c
        res4b9 = mx.symbol.broadcast_add(name='res4b9', *[res4b8_relu, scale4b9_branch2c])
        res4b9_relu = mx.symbol.Activation(name='res4b9_relu', data=res4b9, act_type='relu')
        res4b10_branch2a = mx.symbol.Convolution(name='res4b10_branch2a', data=res4b9_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b10_branch2a = mx.symbol.BatchNorm(name='bn4b10_branch2a', data=res4b10_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b10_branch2a = bn4b10_branch2a
        res4b10_branch2a_relu = mx.symbol.Activation(name='res4b10_branch2a_relu', data=scale4b10_branch2a,
                                                     act_type='relu')
        res4b10_branch2b = mx.symbol.Convolution(name='res4b10_branch2b', data=res4b10_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b10_branch2b = mx.symbol.BatchNorm(name='bn4b10_branch2b', data=res4b10_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b10_branch2b = bn4b10_branch2b
        res4b10_branch2b_relu = mx.symbol.Activation(name='res4b10_branch2b_relu', data=scale4b10_branch2b,
                                                     act_type='relu')
        res4b10_branch2c = mx.symbol.Convolution(name='res4b10_branch2c', data=res4b10_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b10_branch2c = mx.symbol.BatchNorm(name='bn4b10_branch2c', data=res4b10_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b10_branch2c = bn4b10_branch2c
        res4b10 = mx.symbol.broadcast_add(name='res4b10', *[res4b9_relu, scale4b10_branch2c])
        res4b10_relu = mx.symbol.Activation(name='res4b10_relu', data=res4b10, act_type='relu')
        res4b11_branch2a = mx.symbol.Convolution(name='res4b11_branch2a', data=res4b10_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b11_branch2a = mx.symbol.BatchNorm(name='bn4b11_branch2a', data=res4b11_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b11_branch2a = bn4b11_branch2a
        res4b11_branch2a_relu = mx.symbol.Activation(name='res4b11_branch2a_relu', data=scale4b11_branch2a,
                                                     act_type='relu')
        res4b11_branch2b = mx.symbol.Convolution(name='res4b11_branch2b', data=res4b11_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b11_branch2b = mx.symbol.BatchNorm(name='bn4b11_branch2b', data=res4b11_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b11_branch2b = bn4b11_branch2b
        res4b11_branch2b_relu = mx.symbol.Activation(name='res4b11_branch2b_relu', data=scale4b11_branch2b,
                                                     act_type='relu')
        res4b11_branch2c = mx.symbol.Convolution(name='res4b11_branch2c', data=res4b11_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b11_branch2c = mx.symbol.BatchNorm(name='bn4b11_branch2c', data=res4b11_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b11_branch2c = bn4b11_branch2c
        res4b11 = mx.symbol.broadcast_add(name='res4b11', *[res4b10_relu, scale4b11_branch2c])
        res4b11_relu = mx.symbol.Activation(name='res4b11_relu', data=res4b11, act_type='relu')
        res4b12_branch2a = mx.symbol.Convolution(name='res4b12_branch2a', data=res4b11_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b12_branch2a = mx.symbol.BatchNorm(name='bn4b12_branch2a', data=res4b12_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b12_branch2a = bn4b12_branch2a
        res4b12_branch2a_relu = mx.symbol.Activation(name='res4b12_branch2a_relu', data=scale4b12_branch2a,
                                                     act_type='relu')
        res4b12_branch2b = mx.symbol.Convolution(name='res4b12_branch2b', data=res4b12_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b12_branch2b = mx.symbol.BatchNorm(name='bn4b12_branch2b', data=res4b12_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b12_branch2b = bn4b12_branch2b
        res4b12_branch2b_relu = mx.symbol.Activation(name='res4b12_branch2b_relu', data=scale4b12_branch2b,
                                                     act_type='relu')
        res4b12_branch2c = mx.symbol.Convolution(name='res4b12_branch2c', data=res4b12_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b12_branch2c = mx.symbol.BatchNorm(name='bn4b12_branch2c', data=res4b12_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b12_branch2c = bn4b12_branch2c
        res4b12 = mx.symbol.broadcast_add(name='res4b12', *[res4b11_relu, scale4b12_branch2c])
        res4b12_relu = mx.symbol.Activation(name='res4b12_relu', data=res4b12, act_type='relu')
        res4b13_branch2a = mx.symbol.Convolution(name='res4b13_branch2a', data=res4b12_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b13_branch2a = mx.symbol.BatchNorm(name='bn4b13_branch2a', data=res4b13_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b13_branch2a = bn4b13_branch2a
        res4b13_branch2a_relu = mx.symbol.Activation(name='res4b13_branch2a_relu', data=scale4b13_branch2a,
                                                     act_type='relu')
        res4b13_branch2b = mx.symbol.Convolution(name='res4b13_branch2b', data=res4b13_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b13_branch2b = mx.symbol.BatchNorm(name='bn4b13_branch2b', data=res4b13_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b13_branch2b = bn4b13_branch2b
        res4b13_branch2b_relu = mx.symbol.Activation(name='res4b13_branch2b_relu', data=scale4b13_branch2b,
                                                     act_type='relu')
        res4b13_branch2c = mx.symbol.Convolution(name='res4b13_branch2c', data=res4b13_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b13_branch2c = mx.symbol.BatchNorm(name='bn4b13_branch2c', data=res4b13_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b13_branch2c = bn4b13_branch2c
        res4b13 = mx.symbol.broadcast_add(name='res4b13', *[res4b12_relu, scale4b13_branch2c])
        res4b13_relu = mx.symbol.Activation(name='res4b13_relu', data=res4b13, act_type='relu')
        res4b14_branch2a = mx.symbol.Convolution(name='res4b14_branch2a', data=res4b13_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b14_branch2a = mx.symbol.BatchNorm(name='bn4b14_branch2a', data=res4b14_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b14_branch2a = bn4b14_branch2a
        res4b14_branch2a_relu = mx.symbol.Activation(name='res4b14_branch2a_relu', data=scale4b14_branch2a,
                                                     act_type='relu')
        res4b14_branch2b = mx.symbol.Convolution(name='res4b14_branch2b', data=res4b14_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b14_branch2b = mx.symbol.BatchNorm(name='bn4b14_branch2b', data=res4b14_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b14_branch2b = bn4b14_branch2b
        res4b14_branch2b_relu = mx.symbol.Activation(name='res4b14_branch2b_relu', data=scale4b14_branch2b,
                                                     act_type='relu')
        res4b14_branch2c = mx.symbol.Convolution(name='res4b14_branch2c', data=res4b14_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b14_branch2c = mx.symbol.BatchNorm(name='bn4b14_branch2c', data=res4b14_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b14_branch2c = bn4b14_branch2c
        res4b14 = mx.symbol.broadcast_add(name='res4b14', *[res4b13_relu, scale4b14_branch2c])
        res4b14_relu = mx.symbol.Activation(name='res4b14_relu', data=res4b14, act_type='relu')
        res4b15_branch2a = mx.symbol.Convolution(name='res4b15_branch2a', data=res4b14_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b15_branch2a = mx.symbol.BatchNorm(name='bn4b15_branch2a', data=res4b15_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b15_branch2a = bn4b15_branch2a
        res4b15_branch2a_relu = mx.symbol.Activation(name='res4b15_branch2a_relu', data=scale4b15_branch2a,
                                                     act_type='relu')
        res4b15_branch2b = mx.symbol.Convolution(name='res4b15_branch2b', data=res4b15_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b15_branch2b = mx.symbol.BatchNorm(name='bn4b15_branch2b', data=res4b15_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b15_branch2b = bn4b15_branch2b
        res4b15_branch2b_relu = mx.symbol.Activation(name='res4b15_branch2b_relu', data=scale4b15_branch2b,
                                                     act_type='relu')
        res4b15_branch2c = mx.symbol.Convolution(name='res4b15_branch2c', data=res4b15_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b15_branch2c = mx.symbol.BatchNorm(name='bn4b15_branch2c', data=res4b15_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b15_branch2c = bn4b15_branch2c
        res4b15 = mx.symbol.broadcast_add(name='res4b15', *[res4b14_relu, scale4b15_branch2c])
        res4b15_relu = mx.symbol.Activation(name='res4b15_relu', data=res4b15, act_type='relu')
        res4b16_branch2a = mx.symbol.Convolution(name='res4b16_branch2a', data=res4b15_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b16_branch2a = mx.symbol.BatchNorm(name='bn4b16_branch2a', data=res4b16_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b16_branch2a = bn4b16_branch2a
        res4b16_branch2a_relu = mx.symbol.Activation(name='res4b16_branch2a_relu', data=scale4b16_branch2a,
                                                     act_type='relu')
        res4b16_branch2b = mx.symbol.Convolution(name='res4b16_branch2b', data=res4b16_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b16_branch2b = mx.symbol.BatchNorm(name='bn4b16_branch2b', data=res4b16_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b16_branch2b = bn4b16_branch2b
        res4b16_branch2b_relu = mx.symbol.Activation(name='res4b16_branch2b_relu', data=scale4b16_branch2b,
                                                     act_type='relu')
        res4b16_branch2c = mx.symbol.Convolution(name='res4b16_branch2c', data=res4b16_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b16_branch2c = mx.symbol.BatchNorm(name='bn4b16_branch2c', data=res4b16_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b16_branch2c = bn4b16_branch2c
        res4b16 = mx.symbol.broadcast_add(name='res4b16', *[res4b15_relu, scale4b16_branch2c])
        res4b16_relu = mx.symbol.Activation(name='res4b16_relu', data=res4b16, act_type='relu')
        res4b17_branch2a = mx.symbol.Convolution(name='res4b17_branch2a', data=res4b16_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b17_branch2a = mx.symbol.BatchNorm(name='bn4b17_branch2a', data=res4b17_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b17_branch2a = bn4b17_branch2a
        res4b17_branch2a_relu = mx.symbol.Activation(name='res4b17_branch2a_relu', data=scale4b17_branch2a,
                                                     act_type='relu')
        res4b17_branch2b = mx.symbol.Convolution(name='res4b17_branch2b', data=res4b17_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b17_branch2b = mx.symbol.BatchNorm(name='bn4b17_branch2b', data=res4b17_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b17_branch2b = bn4b17_branch2b
        res4b17_branch2b_relu = mx.symbol.Activation(name='res4b17_branch2b_relu', data=scale4b17_branch2b,
                                                     act_type='relu')
        res4b17_branch2c = mx.symbol.Convolution(name='res4b17_branch2c', data=res4b17_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b17_branch2c = mx.symbol.BatchNorm(name='bn4b17_branch2c', data=res4b17_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b17_branch2c = bn4b17_branch2c
        res4b17 = mx.symbol.broadcast_add(name='res4b17', *[res4b16_relu, scale4b17_branch2c])
        res4b17_relu = mx.symbol.Activation(name='res4b17_relu', data=res4b17, act_type='relu')
        res4b18_branch2a = mx.symbol.Convolution(name='res4b18_branch2a', data=res4b17_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b18_branch2a = mx.symbol.BatchNorm(name='bn4b18_branch2a', data=res4b18_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b18_branch2a = bn4b18_branch2a
        res4b18_branch2a_relu = mx.symbol.Activation(name='res4b18_branch2a_relu', data=scale4b18_branch2a,
                                                     act_type='relu')
        res4b18_branch2b = mx.symbol.Convolution(name='res4b18_branch2b', data=res4b18_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b18_branch2b = mx.symbol.BatchNorm(name='bn4b18_branch2b', data=res4b18_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b18_branch2b = bn4b18_branch2b
        res4b18_branch2b_relu = mx.symbol.Activation(name='res4b18_branch2b_relu', data=scale4b18_branch2b,
                                                     act_type='relu')
        res4b18_branch2c = mx.symbol.Convolution(name='res4b18_branch2c', data=res4b18_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b18_branch2c = mx.symbol.BatchNorm(name='bn4b18_branch2c', data=res4b18_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b18_branch2c = bn4b18_branch2c
        res4b18 = mx.symbol.broadcast_add(name='res4b18', *[res4b17_relu, scale4b18_branch2c])
        res4b18_relu = mx.symbol.Activation(name='res4b18_relu', data=res4b18, act_type='relu')
        res4b19_branch2a = mx.symbol.Convolution(name='res4b19_branch2a', data=res4b18_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b19_branch2a = mx.symbol.BatchNorm(name='bn4b19_branch2a', data=res4b19_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b19_branch2a = bn4b19_branch2a
        res4b19_branch2a_relu = mx.symbol.Activation(name='res4b19_branch2a_relu', data=scale4b19_branch2a,
                                                     act_type='relu')
        res4b19_branch2b = mx.symbol.Convolution(name='res4b19_branch2b', data=res4b19_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b19_branch2b = mx.symbol.BatchNorm(name='bn4b19_branch2b', data=res4b19_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b19_branch2b = bn4b19_branch2b
        res4b19_branch2b_relu = mx.symbol.Activation(name='res4b19_branch2b_relu', data=scale4b19_branch2b,
                                                     act_type='relu')
        res4b19_branch2c = mx.symbol.Convolution(name='res4b19_branch2c', data=res4b19_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b19_branch2c = mx.symbol.BatchNorm(name='bn4b19_branch2c', data=res4b19_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b19_branch2c = bn4b19_branch2c
        res4b19 = mx.symbol.broadcast_add(name='res4b19', *[res4b18_relu, scale4b19_branch2c])
        res4b19_relu = mx.symbol.Activation(name='res4b19_relu', data=res4b19, act_type='relu')
        res4b20_branch2a = mx.symbol.Convolution(name='res4b20_branch2a', data=res4b19_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b20_branch2a = mx.symbol.BatchNorm(name='bn4b20_branch2a', data=res4b20_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b20_branch2a = bn4b20_branch2a
        res4b20_branch2a_relu = mx.symbol.Activation(name='res4b20_branch2a_relu', data=scale4b20_branch2a,
                                                     act_type='relu')
        res4b20_branch2b = mx.symbol.Convolution(name='res4b20_branch2b', data=res4b20_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b20_branch2b = mx.symbol.BatchNorm(name='bn4b20_branch2b', data=res4b20_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b20_branch2b = bn4b20_branch2b
        res4b20_branch2b_relu = mx.symbol.Activation(name='res4b20_branch2b_relu', data=scale4b20_branch2b,
                                                     act_type='relu')
        res4b20_branch2c = mx.symbol.Convolution(name='res4b20_branch2c', data=res4b20_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b20_branch2c = mx.symbol.BatchNorm(name='bn4b20_branch2c', data=res4b20_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b20_branch2c = bn4b20_branch2c
        res4b20 = mx.symbol.broadcast_add(name='res4b20', *[res4b19_relu, scale4b20_branch2c])
        res4b20_relu = mx.symbol.Activation(name='res4b20_relu', data=res4b20, act_type='relu')
        res4b21_branch2a = mx.symbol.Convolution(name='res4b21_branch2a', data=res4b20_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b21_branch2a = mx.symbol.BatchNorm(name='bn4b21_branch2a', data=res4b21_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b21_branch2a = bn4b21_branch2a
        res4b21_branch2a_relu = mx.symbol.Activation(name='res4b21_branch2a_relu', data=scale4b21_branch2a,
                                                     act_type='relu')
        res4b21_branch2b = mx.symbol.Convolution(name='res4b21_branch2b', data=res4b21_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b21_branch2b = mx.symbol.BatchNorm(name='bn4b21_branch2b', data=res4b21_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b21_branch2b = bn4b21_branch2b
        res4b21_branch2b_relu = mx.symbol.Activation(name='res4b21_branch2b_relu', data=scale4b21_branch2b,
                                                     act_type='relu')
        res4b21_branch2c = mx.symbol.Convolution(name='res4b21_branch2c', data=res4b21_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b21_branch2c = mx.symbol.BatchNorm(name='bn4b21_branch2c', data=res4b21_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b21_branch2c = bn4b21_branch2c
        res4b21 = mx.symbol.broadcast_add(name='res4b21', *[res4b20_relu, scale4b21_branch2c])
        res4b21_relu = mx.symbol.Activation(name='res4b21_relu', data=res4b21, act_type='relu')
        res4b22_branch2a = mx.symbol.Convolution(name='res4b22_branch2a', data=res4b21_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b22_branch2a = mx.symbol.BatchNorm(name='bn4b22_branch2a', data=res4b22_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b22_branch2a = bn4b22_branch2a
        res4b22_branch2a_relu = mx.symbol.Activation(name='res4b22_branch2a_relu', data=scale4b22_branch2a,
                                                     act_type='relu')
        res4b22_branch2b = mx.symbol.Convolution(name='res4b22_branch2b', data=res4b22_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b22_branch2b = mx.symbol.BatchNorm(name='bn4b22_branch2b', data=res4b22_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b22_branch2b = bn4b22_branch2b
        res4b22_branch2b_relu = mx.symbol.Activation(name='res4b22_branch2b_relu', data=scale4b22_branch2b,
                                                     act_type='relu')
        res4b22_branch2c = mx.symbol.Convolution(name='res4b22_branch2c', data=res4b22_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b22_branch2c = mx.symbol.BatchNorm(name='bn4b22_branch2c', data=res4b22_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b22_branch2c = bn4b22_branch2c
        res4b22 = mx.symbol.broadcast_add(name='res4b22', *[res4b21_relu, scale4b22_branch2c])
        res4b22_relu = mx.symbol.Activation(name='res4b22_relu', data=res4b22, act_type='relu')
        return res4b22_relu

    def get_resnet_v1_conv5(self, conv4):
        res5a_branch1 = mx.symbol.Convolution(name='res5a_branch1', data=conv4, num_filter=2048, pad=(0, 0),
                                              kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn5a_branch1 = mx.symbol.BatchNorm(name='bn5a_branch1', data=res5a_branch1, use_global_stats=True, fix_gamma=False, eps=self.eps)
        scale5a_branch1 = bn5a_branch1
        res5a_branch2a = mx.symbol.Convolution(name='res5a_branch2a', data=conv4, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn5a_branch2a = mx.symbol.BatchNorm(name='bn5a_branch2a', data=res5a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2a = bn5a_branch2a

        res5a_branch2a_relu = mx.symbol.Activation(name='res5a_branch2a_relu', data=scale5a_branch2a, act_type='relu')


        # res5a_branch2b_offset = mx.symbol.Convolution(name='res5a_branch2b_offset', data = res5a_branch2a_relu,
        #                                               num_filter=72, pad=(2, 2), kernel=(3, 3), stride=(1, 1), dilate=(2, 2), cudnn_off=True)
        # res5a_branch2b = mx.contrib.symbol.DeformableConvolution(name='res5a_branch2b', data=res5a_branch2a_relu, offset=res5a_branch2b_offset,
        #                                                          num_filter=512, pad=(2, 2), kernel=(3, 3), num_deformable_group=4,
        #                                                          stride=(1, 1), dilate=(2, 2), no_bias=True)

        res5a_branch2b = mx.symbol.Convolution(name='res5a_branch2b', data=res5a_branch2a_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)

        bn5a_branch2b = mx.symbol.BatchNorm(name='bn5a_branch2b', data=res5a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2b = bn5a_branch2b
        res5a_branch2b_relu = mx.symbol.Activation(name='res5a_branch2b_relu', data=scale5a_branch2b, act_type='relu')
        res5a_branch2c = mx.symbol.Convolution(name='res5a_branch2c', data=res5a_branch2b_relu, num_filter=2048, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5a_branch2c = mx.symbol.BatchNorm(name='bn5a_branch2c', data=res5a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2c = bn5a_branch2c
        res5a = mx.symbol.broadcast_add(name='res5a', *[scale5a_branch1, scale5a_branch2c])
        res5a_relu = mx.symbol.Activation(name='res5a_relu', data=res5a, act_type='relu')
        res5b_branch2a = mx.symbol.Convolution(name='res5b_branch2a', data=res5a_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5b_branch2a = mx.symbol.BatchNorm(name='bn5b_branch2a', data=res5b_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2a = bn5b_branch2a
        res5b_branch2a_relu = mx.symbol.Activation(name='res5b_branch2a_relu', data=scale5b_branch2a, act_type='relu')

        # res5b_branch2b_offset = mx.symbol.Convolution(name='res5b_branch2b_offset', data = res5b_branch2a_relu,
        #                                               num_filter=72, pad=(2, 2), kernel=(3, 3), stride=(1, 1), dilate=(2, 2), cudnn_off=True)
        # res5b_branch2b = mx.contrib.symbol.DeformableConvolution(name='res5b_branch2b', data=res5b_branch2a_relu, offset=res5b_branch2b_offset,
        #                                                          num_filter=512, pad=(2, 2), kernel=(3, 3), num_deformable_group=4,
        #                                                          stride=(1, 1), dilate=(2, 2), no_bias=True)
        res5b_branch2b = mx.symbol.Convolution(name='res5b_branch2b', data=res5b_branch2a_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5b_branch2b = mx.symbol.BatchNorm(name='bn5b_branch2b', data=res5b_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2b = bn5b_branch2b
        res5b_branch2b_relu = mx.symbol.Activation(name='res5b_branch2b_relu', data=scale5b_branch2b, act_type='relu')
        res5b_branch2c = mx.symbol.Convolution(name='res5b_branch2c', data=res5b_branch2b_relu, num_filter=2048, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5b_branch2c = mx.symbol.BatchNorm(name='bn5b_branch2c', data=res5b_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2c = bn5b_branch2c
        res5b = mx.symbol.broadcast_add(name='res5b', *[res5a_relu, scale5b_branch2c])
        res5b_relu = mx.symbol.Activation(name='res5b_relu', data=res5b, act_type='relu')
        res5c_branch2a = mx.symbol.Convolution(name='res5c_branch2a', data=res5b_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5c_branch2a = mx.symbol.BatchNorm(name='bn5c_branch2a', data=res5c_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2a = bn5c_branch2a
        res5c_branch2a_relu = mx.symbol.Activation(name='res5c_branch2a_relu', data=scale5c_branch2a, act_type='relu')

        # res5c_branch2b_offset = mx.symbol.Convolution(name='res5c_branch2b_offset', data = res5c_branch2a_relu,
        #                                               num_filter=72, pad=(2, 2), kernel=(3, 3), stride=(1, 1), dilate=(2, 2), cudnn_off=True)
        # res5c_branch2b = mx.contrib.symbol.DeformableConvolution(name='res5c_branch2b', data=res5c_branch2a_relu, offset=res5c_branch2b_offset,
        #                                                          num_filter=512, pad=(2, 2), kernel=(3, 3), num_deformable_group=4,
        #                                                          stride=(1, 1), dilate=(2, 2), no_bias=True)

        res5c_branch2b = mx.symbol.Convolution(name='res5c_branch2b', data=res5c_branch2a_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)

        bn5c_branch2b = mx.symbol.BatchNorm(name='bn5c_branch2b', data=res5c_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2b = bn5c_branch2b
        res5c_branch2b_relu = mx.symbol.Activation(name='res5c_branch2b_relu', data=scale5c_branch2b, act_type='relu')
        res5c_branch2c = mx.symbol.Convolution(name='res5c_branch2c', data=res5c_branch2b_relu, num_filter=2048, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5c_branch2c = mx.symbol.BatchNorm(name='bn5c_branch2c', data=res5c_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2c = bn5c_branch2c
        res5c = mx.symbol.broadcast_add(name='res5c', *[res5b_relu, scale5c_branch2c])
        res5c_relu = mx.symbol.Activation(name='res5c_relu', data=res5c, act_type='relu')
        return res5c_relu


    def get_rpn(self, conv_feat, num_anchors,p_str):
        rpn_conv = mx.sym.Convolution(
            data=conv_feat, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3"+p_str)
        rpn_relu = mx.sym.Activation(data=rpn_conv, act_type="relu", name="rpn_relu"+p_str)
        rpn_cls_score = mx.sym.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score"+p_str)
        rpn_bbox_pred = mx.sym.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred"+p_str)
        return rpn_cls_score, rpn_bbox_pred

    def get_symbol(self, cfg, is_train=True):

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors_p3 = cfg.network.p3_NUM_ANCHORS
        num_anchors_p4 = cfg.network.p4_NUM_ANCHORS
        num_anchors_p5 = cfg.network.p5_NUM_ANCHORS
        num_anchors_p6 = cfg.network.p6_NUM_ANCHORS

        # input init
        if is_train:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name="im_info")
            gt_boxes = mx.sym.Variable(name="gt_boxes")
            
            rpn_label_p3 = mx.sym.Variable(name='label/p3')
            rpn_label_p4 = mx.sym.Variable(name='label/p4')
            rpn_label_p5 = mx.sym.Variable(name='label/p5')
            rpn_label_p6 = mx.sym.Variable(name='label/p6')


            rpn_bbox_target_p3 = mx.sym.Variable(name='bbox_target/p3')           
            rpn_bbox_target_p4 = mx.sym.Variable(name='bbox_target/p4')
            rpn_bbox_target_p5 = mx.sym.Variable(name='bbox_target/p5')
            rpn_bbox_target_p6 = mx.sym.Variable(name='bbox_target/p6')

            rpn_bbox_weight_p3 = mx.sym.Variable(name='bbox_weight/p3')            
            rpn_bbox_weight_p4 = mx.sym.Variable(name='bbox_weight/p4')
            rpn_bbox_weight_p5 = mx.sym.Variable(name='bbox_weight/p5')
            rpn_bbox_weight_p6 = mx.sym.Variable(name='bbox_weight/p6')
        else:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name="im_info")

        # shared convolutional layers
        conv2 = self.get_resnet_v1_conv2(data)#4
        conv3 = self.get_resnet_v1_conv3(conv2)#8
        conv4 = self.get_resnet_v1_conv4(conv3)#16
        conv5 = self.get_resnet_v1_conv5(conv4)#32

        ####################################P4 AND P5###########################
        c3 = mx.sym.Convolution(data = conv2, kernel = (1,1), pad = (0,0),stride=(1, 1), num_filter = 256, name = 'c3' )
        c4 = mx.sym.Convolution(data = conv3, kernel = (1,1), pad = (0,0), stride=(1, 1),num_filter = 256, name = 'c4' )
    	c5 = mx.sym.Convolution(data = conv4, kernel = (1,1), pad = (0,0), stride=(1, 1),num_filter = 256,name = 'c5')
     	c6 = mx.sym.Convolution(data = conv5, kernel = (1,1), pad = (0,0), stride=(1, 1),num_filter = 256,name = 'c6')       
        #######p6
        p6 = c6
        ######p5->newp5
        p6Upx2 = mx.sym.UpSampling(p6, scale =2 , sample_type = 'nearest')
        p6UpCrop = mx.sym.Crop(p6Upx2,c5)
        p5 = mx.sym.ElementWiseSum(name = 'p5', *[c5, p6UpCrop])
      	newp5 = mx.sym.Convolution(data = p5, kernel = (1,1), pad = (0,0), stride=(1, 1), num_filter = 256, name = 'newp5')      
        ####p4->newp4
        p5Upx2 = mx.sym.UpSampling(p5, scale =2 , sample_type = 'nearest')
        p5UpCrop = mx.sym.Crop(p5Upx2,c4)
        p4 = mx.sym.ElementWiseSum(name = 'p4', *[c4, p5UpCrop])
      	newp4 = mx.sym.Convolution(data = p4, kernel = (1,1), pad = (0,0), stride=(1, 1),num_filter = 256,name = 'newp4')         
        ####p3->newp3
        p4Upx2 = mx.sym.UpSampling(p4, scale =2 , sample_type = 'nearest')
        p4UpCrop = mx.sym.Crop(p4Upx2,c3)
        p3 = mx.sym.ElementWiseSum(name = 'p3', *[c3, p4UpCrop])
      	newp3 = mx.sym.Convolution(data = p3, kernel = (1,1), pad = (0,0), stride=(1, 1),num_filter = 256,name = 'newp3')    
 
        #####################fpn-rpn###############
        rpn_cls_score_p3, rpn_bbox_pred_p3 = self.get_rpn(newp3, num_anchors_p3,'/p3')
        rpn_cls_score_p4, rpn_bbox_pred_p4 = self.get_rpn(newp4, num_anchors_p4,'/p4')
        rpn_cls_score_p5, rpn_bbox_pred_p5 = self.get_rpn(newp5, num_anchors_p5,'/p5')
        rpn_cls_score_p6, rpn_bbox_pred_p6 = self.get_rpn(p6, num_anchors_p6,'/p6')


        if is_train:

            rpn_cls_score_reshape_p3 = mx.sym.Reshape(
                 data=rpn_cls_score_p3, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape/p3")    
            
            rpn_cls_score_reshape_p4 = mx.sym.Reshape(
                 data=rpn_cls_score_p4, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape/p4")
    
            rpn_cls_score_reshape_p5 = mx.sym.Reshape(
                 data=rpn_cls_score_p5, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape/p5")
    
            rpn_cls_score_reshape_p6 = mx.sym.Reshape(
                 data=rpn_cls_score_p6, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape/p6")



            rpn_cls_prob_p3 = mx.sym.SoftmaxOutput(data=rpn_cls_score_reshape_p3, label=rpn_label_p3, multi_output=True,
                                                 normalization='valid', use_ignore=True, ignore_label=-1,
                                                 name="rpn_cls_prob/p3")    
            rpn_cls_prob_p4 = mx.sym.SoftmaxOutput(data=rpn_cls_score_reshape_p4, label=rpn_label_p4, multi_output=True,
                                                 normalization='valid', use_ignore=True, ignore_label=-1,
                                                 name="rpn_cls_prob/p4")            
            rpn_cls_prob_p5 = mx.sym.SoftmaxOutput(data=rpn_cls_score_reshape_p5, label=rpn_label_p5, multi_output=True,
                                                 normalization='valid', use_ignore=True, ignore_label=-1,
                                                 name="rpn_cls_prob/p5")
            rpn_cls_prob_p6 = mx.sym.SoftmaxOutput(data=rpn_cls_score_reshape_p6, label=rpn_label_p6, multi_output=True,
                                                 normalization='valid', use_ignore=True, ignore_label=-1,
                                                 name="rpn_cls_prob/p6")

            p3_rpn_bbox_loss_ = rpn_bbox_weight_p3 * mx.sym.smooth_l1(name='rpn_bbox_loss_/p3', scalar=3.0,
                                                                data=(rpn_bbox_pred_p3 - rpn_bbox_target_p3))
            rpn_bbox_loss_p3 = mx.sym.MakeLoss(name='rpn_bbox_loss/p3', data=p3_rpn_bbox_loss_,
                                            grad_scale=1.0 / cfg.TRAIN.RPN_BATCH_SIZE)

            p4_rpn_bbox_loss_ = rpn_bbox_weight_p4 * mx.sym.smooth_l1(name='rpn_bbox_loss_/p4', scalar=3.0,
                                                                data=(rpn_bbox_pred_p4 - rpn_bbox_target_p4))
            rpn_bbox_loss_p4 = mx.sym.MakeLoss(name='rpn_bbox_loss/p4', data=p4_rpn_bbox_loss_,
                                            grad_scale=1.0 / cfg.TRAIN.RPN_BATCH_SIZE)

            p5_rpn_bbox_loss_ = rpn_bbox_weight_p5 * mx.sym.smooth_l1(name='rpn_bbox_loss_/p5', scalar=3.0,
                                                                 data=(rpn_bbox_pred_p5 - rpn_bbox_target_p5))
            
            rpn_bbox_loss_p5 = mx.sym.MakeLoss(name='rpn_bbox_loss/p5', data=p5_rpn_bbox_loss_,
                                             grad_scale=1.0 / cfg.TRAIN.RPN_BATCH_SIZE)

            p6_rpn_bbox_loss_ = rpn_bbox_weight_p6 * mx.sym.smooth_l1(name='rpn_bbox_loss_/p6', scalar=3.0,
                                                                 data=(rpn_bbox_pred_p6 - rpn_bbox_target_p6))
            rpn_bbox_loss_p6 = mx.sym.MakeLoss(name='rpn_bbox_loss/p6', data=p6_rpn_bbox_loss_,
                                             grad_scale=1.0 / cfg.TRAIN.RPN_BATCH_SIZE)


            # ROI proposal
            rpn_cls_act_p3 = mx.sym.SoftmaxActivation(
                data=rpn_cls_score_reshape_p3, mode="channel", name="rpn_cls_act/p3")
            rpn_cls_act_reshape_p3 = mx.sym.Reshape(
                data=rpn_cls_act_p3, shape=(0, 2 * num_anchors_p3, -1, 0), name='rpn_cls_act_reshape/p3')

            rpn_cls_act_p4 = mx.sym.SoftmaxActivation(
                data=rpn_cls_score_reshape_p4, mode="channel", name="rpn_cls_act/p4")
            rpn_cls_act_reshape_p4 = mx.sym.Reshape(
                data=rpn_cls_act_p4, shape=(0, 2 * num_anchors_p4, -1, 0), name='rpn_cls_act_reshape/p4')

            rpn_cls_act_p5 = mx.sym.SoftmaxActivation(
                 data=rpn_cls_score_reshape_p5, mode="channel", name="rpn_cls_act/p5")
            rpn_cls_act_reshape_p5 = mx.sym.Reshape(
                 data=rpn_cls_act_p5, shape=(0, 2 * num_anchors_p5, -1, 0), name='rpn_cls_act_reshape/p5')

            rpn_cls_act_p6 = mx.sym.SoftmaxActivation(
                 data=rpn_cls_score_reshape_p6, mode="channel", name="rpn_cls_act/p6")
            rpn_cls_act_reshape_p6 = mx.sym.Reshape(
                 data=rpn_cls_act_p6, shape=(0, 2 * num_anchors_p6, -1, 0), name='rpn_cls_act_reshape/p6')
            
            if cfg.TRAIN.CXX_PROPOSAL:  
                rois_p3 = mx.contrib.sym.Proposal(
                    cls_prob=rpn_cls_act_reshape_p3, bbox_pred=rpn_bbox_pred_p3, im_info=im_info, name='rois/p3',
                    feature_stride=cfg.network.p3_RPN_FEAT_STRIDE, scales=tuple(cfg.network.p3_ANCHOR_SCALES),
                    ratios=tuple(cfg.network.p3_ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)

                rois_p4 = mx.contrib.sym.Proposal(
                    cls_prob=rpn_cls_act_reshape_p4, bbox_pred=rpn_bbox_pred_p4, im_info=im_info, name='rois/p4',
                    feature_stride=cfg.network.p4_RPN_FEAT_STRIDE, scales=tuple(cfg.network.p4_ANCHOR_SCALES),
                    ratios=tuple(cfg.network.p4_ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)


                rois_p5 = mx.contrib.sym.Proposal(
                     cls_prob=rpn_cls_act_reshape_p5, bbox_pred=rpn_bbox_pred_p5, im_info=im_info, name='rois/p5',
                     feature_stride=cfg.network.p5_RPN_FEAT_STRIDE, scales=tuple(cfg.network.p5_ANCHOR_SCALES),
                     ratios=tuple(cfg.network.p5_ANCHOR_RATIOS),
                     rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                     threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)

                rois_p6 = mx.contrib.sym.Proposal(
                     cls_prob=rpn_cls_act_reshape_p6, bbox_pred=rpn_bbox_pred_p6, im_info=im_info, name='rois/p6',
                     feature_stride=cfg.network.p6_RPN_FEAT_STRIDE, scales=tuple(cfg.network.p6_ANCHOR_SCALES),
                     ratios=tuple(cfg.network.p6_ANCHOR_RATIOS),
                     rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                     threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
            else:
                rois_p3 = mx.sym.Custom(
                    cls_prob=rpn_cls_act_reshape_p3, bbox_pred=rpn_bbox_pred_p3, im_info=im_info, name='rois/p3',
                    op_type='proposal', feat_stride=cfg.network.p3_RPN_FEAT_STRIDE,
                    px_shape = newp3,
                    scales=tuple(cfg.network.p3_ANCHOR_SCALES), ratios=tuple(cfg.network.p3_ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)

                rois_p4 = mx.sym.Custom(
                    cls_prob=rpn_cls_act_reshape_p4, bbox_pred=rpn_bbox_pred_p4, im_info=im_info, name='rois/p4',
                    op_type='proposal', feat_stride=cfg.network.p4_RPN_FEAT_STRIDE,
                    px_shape = newp4,
                    scales=tuple(cfg.network.p4_ANCHOR_SCALES), ratios=tuple(cfg.network.p4_ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
                rois_p5 = mx.sym.Custom(
                     cls_prob=rpn_cls_act_reshape_p5, bbox_pred=rpn_bbox_pred_p5, im_info=im_info, name='rois/p5',
                     op_type='proposal',feat_stride=cfg.network.p5_RPN_FEAT_STRIDE,
                     px_shape = newp5,
                     scales=tuple(cfg.network.p5_ANCHOR_SCALES), ratios=tuple(cfg.network.p5_ANCHOR_RATIOS),
                     rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
                rois_p6 = mx.sym.Custom(
                     cls_prob=rpn_cls_act_reshape_p6, bbox_pred=rpn_bbox_pred_p6, im_info=im_info, name='rois/p6',
                     op_type='proposal',feat_stride=cfg.network.p6_RPN_FEAT_STRIDE,
                     px_shape = p6,
                     scales=tuple(cfg.network.p6_ANCHOR_SCALES), ratios=tuple(cfg.network.p6_ANCHOR_RATIOS),
                     rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
            # ROI proposal target
            gt_boxes_reshape = mx.sym.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
            
            rois_p3, label_p3, bbox_target_p3, bbox_weight_p3 = mx.sym.Custom(rois=rois_p3, gt_boxes=gt_boxes_reshape,
                                                                  op_type='proposal_target',
                                                                  num_classes=num_reg_classes,
                                                                  batch_images=cfg.TRAIN.BATCH_IMAGES,
                                                                  batch_rois=cfg.TRAIN.BATCH_ROIS,
                                                                  cfg=cPickle.dumps(cfg),
                                                                  fg_fraction=cfg.TRAIN.FG_FRACTION)
            rois_p4, label_p4, bbox_target_p4, bbox_weight_p4 = mx.sym.Custom(rois=rois_p4, gt_boxes=gt_boxes_reshape,
                                                                  op_type='proposal_target',
                                                                  num_classes=num_reg_classes,
                                                                  batch_images=cfg.TRAIN.BATCH_IMAGES,
                                                                  batch_rois=cfg.TRAIN.BATCH_ROIS,
                                                                  cfg=cPickle.dumps(cfg),
                                                                  fg_fraction=cfg.TRAIN.FG_FRACTION)

            rois_p5, label_p5, bbox_target_p5, bbox_weight_p5 = mx.sym.Custom(rois=rois_p5, gt_boxes=gt_boxes_reshape,
                                                                   op_type='proposal_target',
                                                                   num_classes=num_reg_classes,
                                                                   batch_images=cfg.TRAIN.BATCH_IMAGES,
                                                                   batch_rois=cfg.TRAIN.BATCH_ROIS,   
                                                                   cfg=cPickle.dumps(cfg),
                                                                   fg_fraction=cfg.TRAIN.FG_FRACTION)

            rois_p6, label_p6, bbox_target_p6, bbox_weight_p6 = mx.sym.Custom(rois=rois_p6, gt_boxes=gt_boxes_reshape,
                                                                   op_type='proposal_target',
                                                                   num_classes=num_reg_classes,
                                                                   batch_images=cfg.TRAIN.BATCH_IMAGES,
                                                                   batch_rois=cfg.TRAIN.BATCH_ROIS,   
                                                                   cfg=cPickle.dumps(cfg),
                                                                   fg_fraction=cfg.TRAIN.FG_FRACTION)
           # rois = mx.sym.Concat(rois_p4,rois_p5,dim = 0)
####################################################

            label = mx.sym.Concat(label_p3,label_p4,label_p5,label_p6,dim =0)
            bbox_target = mx.sym.Concat(bbox_target_p3,bbox_target_p4,bbox_target_p5,bbox_target_p6,dim=0)
            bbox_weight = mx.sym.Concat(bbox_weight_p3,bbox_weight_p4,bbox_weight_p5,bbox_weight_p6,dim=0)
 
        else:
            # ROI Proposal
            rpn_cls_score_reshape_p3 = mx.sym.Reshape(
                data=rpn_cls_score_p3, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape/p3")
            rpn_cls_prob_p3 = mx.sym.SoftmaxActivation(
                data=rpn_cls_score_reshape_p3, mode="channel", name="rpn_cls_prob/p3")
            rpn_cls_prob_reshape_p3 = mx.sym.Reshape(
                data=rpn_cls_prob_p3, shape=(0, 2 * num_anchors_p3, -1, 0), name='rpn_cls_prob_reshape/p3')

            rpn_cls_score_reshape_p4 = mx.sym.Reshape(
                data=rpn_cls_score_p4, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape/p4")
            rpn_cls_prob_p4 = mx.sym.SoftmaxActivation(
                data=rpn_cls_score_reshape_p4, mode="channel", name="rpn_cls_prob/p4")
            rpn_cls_prob_reshape_p4 = mx.sym.Reshape(
                data=rpn_cls_prob_p4, shape=(0, 2 * num_anchors_p4, -1, 0), name='rpn_cls_prob_reshape/p4')


            rpn_cls_score_reshape_p5 = mx.sym.Reshape(
                 data=rpn_cls_score_p5, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape/p5")
            rpn_cls_prob_p5 = mx.sym.SoftmaxActivation(
                 data=rpn_cls_score_reshape_p5, mode="channel", name="rpn_cls_prob/p5")
            rpn_cls_prob_reshape_p5 = mx.sym.Reshape(
                  data=rpn_cls_prob_p5, shape=(0, 2 * num_anchors_p5, -1, 0), name='rpn_cls_prob_reshape/p5')
    

            rpn_cls_score_reshape_p6 = mx.sym.Reshape(
                 data=rpn_cls_score_p6, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape/p6")
            rpn_cls_prob_p5 = mx.sym.SoftmaxActivation(
                 data=rpn_cls_score_reshape_p6, mode="channel", name="rpn_cls_prob/p6")
            rpn_cls_prob_reshape_p6 = mx.sym.Reshape(
                  data=rpn_cls_prob_p5, shape=(0, 2 * num_anchors_p6, -1, 0), name='rpn_cls_prob_reshape/p6')

            if cfg.TEST.CXX_PROPOSAL:
                rois_p3 = mx.contrib.sym.Proposal(
                    cls_prob=rpn_cls_prob_reshape_p3, bbox_pred=rpn_bbox_pred_p3, im_info=im_info, name='rois/p3',
                    feature_stride=cfg.network.p3_RPN_FEAT_STRIDE, scales=tuple(cfg.network.p3_ANCHOR_SCALES),
                    ratios=tuple(cfg.network.p3_ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)

                rois_p4 = mx.contrib.sym.Proposal(
                    cls_prob=rpn_cls_prob_reshape_p4, bbox_pred=rpn_bbox_pred_p4, im_info=im_info, name='rois/p4',
                    feature_stride=cfg.network.p4_RPN_FEAT_STRIDE, scales=tuple(cfg.network.p4_ANCHOR_SCALES),
                    ratios=tuple(cfg.network.p4_ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
                
                rois_p5 = mx.contrib.sym.Proposal(
                     cls_prob = rpn_cls_prob_reshape_p5, bbox_pred=rpn_bbox_pred_p5, im_info=im_info, name='rois/p5',
                     feature_stride=cfg.network.p5_RPN_FEAT_STRIDE, scales=tuple(cfg.network.p5_ANCHOR_SCALES),
                     ratios=tuple(cfg.network.p5_ANCHOR_RATIOS),
                     rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                     threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
          
                rois_p6 = mx.contrib.sym.Proposal(
                     cls_prob = rpn_cls_prob_reshape_p6, bbox_pred=rpn_bbox_pred_p6, im_info=im_info, name='rois/p6',
                     feature_stride=cfg.network.p6_RPN_FEAT_STRIDE, scales=tuple(cfg.network.p6_ANCHOR_SCALES),
                     ratios=tuple(cfg.network.p6_ANCHOR_RATIOS),
                     rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                     threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
            else:
                rois_p3 = mx.sym.Custom(
                    cls_prob=rpn_cls_prob_reshape_p3, bbox_pred=rpn_bbox_pred_p3, im_info=im_info, name='rois/p3',
                    op_type='proposal', feat_stride=cfg.network.p3_RPN_FEAT_STRIDE,
                    px_shape = newp3,
                    scales=tuple(cfg.network.p3_ANCHOR_SCALES), ratios=tuple(cfg.network.p3_ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)

                rois_p4 = mx.sym.Custom(
                    cls_prob=rpn_cls_prob_reshape_p4, bbox_pred=rpn_bbox_pred_p4, im_info=im_info, name='rois/p4',
                    op_type='proposal', feat_stride=cfg.network.p4_RPN_FEAT_STRIDE,
                    px_shape = newp4,
                    scales=tuple(cfg.network.p4_ANCHOR_SCALES), ratios=tuple(cfg.network.p4_ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
                rois_p5 = mx.sym.Custom(
                    cls_prob=rpn_cls_prob_reshape_p5, bbox_pred=rpn_bbox_pred_p5, im_info=im_info, name='rois/p5',
                    op_type='proposal', feat_stride=cfg.network.p5_RPN_FEAT_STRIDE,
                    px_shape = newp5,                   
                    scales=tuple(cfg.network.p5_ANCHOR_SCALES), ratios=tuple(cfg.network.p5_ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)

                rois_p6 = mx.sym.Custom(
                    cls_prob=rpn_cls_prob_reshape_p6, bbox_pred=rpn_bbox_pred_p6, im_info=im_info, name='rois/p6',
                    op_type='proposal', feat_stride=cfg.network.p6_RPN_FEAT_STRIDE,
                    px_shape = p6,
                    scales=tuple(cfg.network.p6_ANCHOR_SCALES), ratios=tuple(cfg.network.p6_ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)

            rois = mx.sym.Concat(rois_p3,rois_p4,rois_p5,rois_p6,dim=0,name = 'rois')
 
        #####################################roi pooling p#################################

    
        offset_t_p3 = mx.contrib.sym.DeformablePSROIPooling(name='offset_t/p3', data=newp3, rois=rois_p3, group_size=1, pooled_size=7,
                                                         sample_per_part=4, no_trans=True, part_size=7, output_dim=256, spatial_scale=0.25)
        offset_p3 = mx.sym.FullyConnected(name='offset/p3', data=offset_t_p3, num_hidden=7 * 7 * 2, lr_mult=0.01)
        offset_reshape_p3 = mx.sym.Reshape(data=offset_p3, shape=(-1, 2, 7, 7), name="offset_reshape/p3")

        deformable_roi_pool_p3 = mx.contrib.sym.DeformablePSROIPooling(name='deformable_roi_pool/p3', data=newp3, rois=rois_p3,
                                                                    trans=offset_reshape_p3, group_size=1, pooled_size=7, sample_per_part=4,
                                                                    no_trans=False, part_size=7, output_dim=256, spatial_scale=0.25, trans_std=0.1)


        offset_t_p4 = mx.contrib.sym.DeformablePSROIPooling(name='offset_t/p4', data=newp4, rois=rois_p4, group_size=1, pooled_size=7,
                                                         sample_per_part=4, no_trans=True, part_size=7, output_dim=256, spatial_scale=0.125)
        offset_p4 = mx.sym.FullyConnected(name='offset/p4', data=offset_t_p4, num_hidden=7 * 7 * 2, lr_mult=0.01)
        offset_reshape_p4 = mx.sym.Reshape(data=offset_p4, shape=(-1, 2, 7, 7), name="offset_reshape/p4")

        deformable_roi_pool_p4 = mx.contrib.sym.DeformablePSROIPooling(name='deformable_roi_pool/p4', data=newp4, rois=rois_p4,
                                                                    trans=offset_reshape_p4, group_size=1, pooled_size=7, sample_per_part=4,
                                                                    no_trans=False, part_size=7, output_dim=256, spatial_scale=0.125, trans_std=0.1)
       

        offset_t_p5 = mx.contrib.sym.DeformablePSROIPooling(name='offset_t/p5', data=newp5, rois=rois_p5, group_size=1, pooled_size=7,
                                                          sample_per_part=4, no_trans=True, part_size=7, output_dim=256, spatial_scale=0.0625)
        offset_p5 = mx.sym.FullyConnected(name='offset/p5', data=offset_t_p5, num_hidden=7 * 7 * 2, lr_mult=0.01)
        offset_reshape_p5 = mx.sym.Reshape(data=offset_p5, shape=(-1, 2, 7, 7), name="offset_reshape/p5")
 
        deformable_roi_pool_p5 = mx.contrib.sym.DeformablePSROIPooling(name='deformable_roi_pool/p5', data=newp5, rois=rois_p5,
                                                                      trans=offset_reshape_p5, group_size=1, pooled_size=7, sample_per_part=4,
                                                                      no_trans=False, part_size=7, output_dim=256, spatial_scale=0.0625, trans_std=0.1)
        
        
        offset_t_p6 = mx.contrib.sym.DeformablePSROIPooling(name='offset_t/p6', data=p6, rois=rois_p6, group_size=1, pooled_size=7,
                                                          sample_per_part=4, no_trans=True, part_size=7, output_dim=256, spatial_scale=0.03125)
        offset_p6 = mx.sym.FullyConnected(name='offset/p6', data=offset_t_p6, num_hidden=7 * 7 * 2, lr_mult=0.01)
        offset_reshape_p6 = mx.sym.Reshape(data=offset_p6, shape=(-1, 2, 7, 7), name="offset_reshape/p6")
 
        deformable_roi_pool_p6 = mx.contrib.sym.DeformablePSROIPooling(name='deformable_roi_pool/p6', data=p6, rois=rois_p6,
                                                                      trans=offset_reshape_p6, group_size=1, pooled_size=7, sample_per_part=4,
                                                                      no_trans=False, part_size=7, output_dim=256, spatial_scale=0.03125, trans_std=0.1)        
        
        
        
        
        #######################concat all rois#####################################################
        deformable_roi_pool = mx.sym.Concat(deformable_roi_pool_p3,deformable_roi_pool_p4,deformable_roi_pool_p5,deformable_roi_pool_p6,dim =0,name='deformable_roi_pool')
       
        # 2 fc
        fc_new_1 = mx.sym.FullyConnected(name='fc_new_1', data=deformable_roi_pool, num_hidden=1024)
        fc_new_1_relu = mx.sym.Activation(data=fc_new_1, act_type='relu', name='fc_new_1_relu')

        fc_new_2 = mx.sym.FullyConnected(name='fc_new_2', data=fc_new_1_relu, num_hidden=1024)
        fc_new_2_relu = mx.sym.Activation(data=fc_new_2, act_type='relu', name='fc_new_2_relu')

        # cls_score/bbox_pred
        cls_score = mx.sym.FullyConnected(name='cls_score', data=fc_new_2_relu, num_hidden=num_classes)
        bbox_pred = mx.sym.FullyConnected(name='bbox_pred', data=fc_new_2_relu, num_hidden=num_reg_classes * 4)

        if is_train:
            if cfg.TRAIN.ENABLE_OHEM:
                labels_ohem, bbox_weights_ohem = mx.sym.Custom(op_type='BoxAnnotatorOHEM', num_classes=num_classes,
                                                               num_reg_classes=num_reg_classes,
                                                               roi_per_img=cfg.TRAIN.BATCH_ROIS_OHEM,
                                                               cls_score=cls_score, bbox_pred=bbox_pred, labels=label,
                                                               bbox_targets=bbox_target, bbox_weights=bbox_weight)
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=labels_ohem,
                                                normalization='valid', use_ignore=True, ignore_label=-1)
                bbox_loss_ = bbox_weights_ohem * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0,
                                                                  data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_,
                                            grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS_OHEM)
                rcnn_label = labels_ohem
            else:
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='valid')
                bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0,
                                                            data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS)
                rcnn_label = label

            # reshape output
            rcnn_label = mx.sym.Reshape(data=rcnn_label, shape=(cfg.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                       name='bbox_loss_reshape')
            group = mx.sym.Group([rpn_cls_prob_p3,rpn_bbox_loss_p3,rpn_cls_prob_p4, rpn_bbox_loss_p4, rpn_cls_prob_p5,rpn_bbox_loss_p5,rpn_cls_prob_p6,rpn_bbox_loss_p6,cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label)])
        else:
            cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                       name='bbox_pred_reshape')
            group = mx.sym.Group([rois, cls_prob, bbox_pred])

        self.sym = group
        return group

    def init_weight(self, cfg, arg_params, aux_params):

        arg_params['rpn_conv_3x3/p3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_conv_3x3/p3_weight'])
        arg_params['rpn_conv_3x3/p3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_conv_3x3/p3_bias'])
        arg_params['rpn_cls_score/p3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_cls_score/p3_weight'])
        arg_params['rpn_cls_score/p3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_cls_score/p3_bias'])
        arg_params['rpn_bbox_pred/p3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_bbox_pred/p3_weight'])
        arg_params['rpn_bbox_pred/p3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_bbox_pred/p3_bias'])

        arg_params['rpn_conv_3x3/p4_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_conv_3x3/p4_weight'])
        arg_params['rpn_conv_3x3/p4_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_conv_3x3/p4_bias'])
        arg_params['rpn_cls_score/p4_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_cls_score/p4_weight'])
        arg_params['rpn_cls_score/p4_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_cls_score/p4_bias'])
        arg_params['rpn_bbox_pred/p4_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_bbox_pred/p4_weight'])
        arg_params['rpn_bbox_pred/p4_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_bbox_pred/p4_bias'])
     
        arg_params['rpn_conv_3x3/p5_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_conv_3x3/p5_weight'])
        arg_params['rpn_conv_3x3/p5_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_conv_3x3/p5_bias'])
        arg_params['rpn_cls_score/p5_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_cls_score/p5_weight'])
        arg_params['rpn_cls_score/p5_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_cls_score/p5_bias'])
        arg_params['rpn_bbox_pred/p5_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_bbox_pred/p5_weight'])
        arg_params['rpn_bbox_pred/p5_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_bbox_pred/p5_bias'])
 
        arg_params['rpn_conv_3x3/p6_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_conv_3x3/p6_weight'])
        arg_params['rpn_conv_3x3/p6_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_conv_3x3/p6_bias'])
        arg_params['rpn_cls_score/p6_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_cls_score/p6_weight'])
        arg_params['rpn_cls_score/p6_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_cls_score/p6_bias'])
        arg_params['rpn_bbox_pred/p6_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_bbox_pred/p6_weight'])
        arg_params['rpn_bbox_pred/p6_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_bbox_pred/p6_bias'])


        arg_params['c3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['c3_weight'])
        arg_params['c3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['c3_bias'])
        arg_params['c4_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['c4_weight'])
        arg_params['c4_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['c4_bias'])
        arg_params['c5_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['c5_weight'])
        arg_params['c5_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['c5_bias'])
        arg_params['c6_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['c6_weight'])
        arg_params['c6_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['c6_bias'])


        arg_params['newp3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['newp3_weight'])
        arg_params['newp3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['newp3_bias'])
        arg_params['newp4_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['newp4_weight'])
        arg_params['newp4_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['newp4_bias'])
        arg_params['newp5_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['newp5_weight'])
        arg_params['newp5_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['newp5_bias'])

        arg_params['res5a_branch2b_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['res5a_branch2b_weight'])
      

        arg_params['res5b_branch2b_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['res5b_branch2b_weight'])
 

        arg_params['res5c_branch2b_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['res5c_branch2b_weight'])


        arg_params['offset/p3_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['offset/p3_weight'])
        arg_params['offset/p3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['offset/p3_bias'])       
        
        arg_params['offset/p4_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['offset/p4_weight'])
        arg_params['offset/p4_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['offset/p4_bias'])

   
        arg_params['offset/p5_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['offset/p5_weight'])
        arg_params['offset/p5_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['offset/p5_bias'])

        arg_params['offset/p6_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['offset/p6_weight'])
        arg_params['offset/p6_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['offset/p6_bias'])        
       
        arg_params['fc_new_1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc_new_1_weight'])
        arg_params['fc_new_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc_new_1_bias'])
        arg_params['fc_new_2_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc_new_2_weight'])
        arg_params['fc_new_2_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc_new_2_bias'])
        arg_params['cls_score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['cls_score_weight'])
        arg_params['cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['cls_score_bias'])
        arg_params['bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bbox_pred_weight'])
        arg_params['bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['bbox_pred_bias'])
