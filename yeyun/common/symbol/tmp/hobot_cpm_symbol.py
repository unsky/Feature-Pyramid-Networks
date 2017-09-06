#-*- coding: UTF-8 -*- 
'''
Created on 2017年3月9日

@author: Crown
'''

import mxnet as mx

def conv_factory(data, num_filter, kernel, stride, pad, act_type = 'relu', conv_type = 0):
    if conv_type == 0:
        conv = mx.symbol.Convolution(data = data, num_filter = num_filter, 
                                     kernel = kernel, stride = stride, pad = pad)
 
        act = mx.symbol.Activation(data = conv, act_type=act_type)
        return act
    elif conv_type == 1:
        conv = mx.symbol.Convolution(data = data, num_filter = num_filter, 
                                     kernel = kernel, stride = stride, pad = pad)
        return conv

def plus_conv(conv_data, add_data, num_filter, kernel, stride, pad, act_type = 'relu'):
    conv = mx.symbol.Convolution(data = conv_data, num_filter = num_filter, 
                                     kernel = kernel, stride = stride, pad = pad)
    res_data = add_data + conv
    act = mx.symbol.Activation(data = res_data, act_type=act_type)
    return act

def homo_conv(data, num_filters, kernel, stride, pad, act_type = 'relu', 
              conv_type = 0, repeat_n = 1):
    assert(repeat_n > 0);
    assert(len(num_filters) == repeat_n);
    for i in range(repeat_n):
        data = conv_factory(data=data, num_filter=num_filters[i], kernel=kernel, 
                            stride=stride, pad=pad, act_type=act_type, conv_type=conv_type)
    return data
 

def output_conv_block(data, num_filters):
    assert(len(num_filters) == 3)
    conv1 = conv_factory(data=data, num_filter=num_filters[0], kernel=(3,3), 
                        stride=(1,1), pad=(1,1), act_type='relu', conv_type=0)
    conv2 = conv_factory(data=conv1, num_filter=num_filters[1], kernel=(3,3), 
                        stride=(1,1), pad=(1,1), act_type='relu', conv_type=0)
    conv3 = conv_factory(data=conv2, num_filter=num_filters[2], kernel=(1,1), 
                        stride=(1,1), pad=(0,0), act_type='relu', conv_type=1)
    return conv3


def get_hobot_conv(data):
    conv1 = homo_conv(data=data, num_filters = [16,16], \
                      kernel = (3,3), stride = (2,2), pad = (0,0), act_type='relu', 
                      conv_type = 0, repeat_n = 2)
     
    conv2 = conv_factory(conv1, num_filter = 32, kernel = (3,3), stride=(2,2), 
                         pad = (0,0), act_type = 'relu', conv_type = 0)
    
    conv3 = conv_factory(conv2, num_filter = 32, kernel = (3,3), stride=(1,1), 
                         pad = (1,1), act_type = 'relu', conv_type = 0) 
    
    res_conv4 = plus_conv(conv3 , conv2, num_filter = 32, kernel = (3,3), stride = (1,1), 
              pad=(1,1), act_type = 'relu');
              
    conv5 = conv_factory(res_conv4, num_filter = 48, kernel = (3,3), stride=(2,2), 
                         pad = (1,1), act_type = 'relu', conv_type = 0) 
    
    conv6_1 = conv_factory(conv5, num_filter = 32, kernel = (3,3), stride=(1,1), 
                         pad = (1,1), act_type = 'relu', conv_type = 1) 
    
    res_conv6_1 = plus_conv(res_conv4 , conv6_1, num_filter = 32, 
                            kernel = (1,1), stride = (2,2), pad=(0,0), act_type = 'relu');
                            
    conv6_2 = conv_factory(res_conv6_1, num_filter = 48, kernel = (3,3), stride=(1,1), 
                           pad = (1,1), act_type = 'relu', conv_type = 0) 
    
    return conv6_2

def get_hobot_conv2(data):
    conv1 = homo_conv(data=data, num_filters = [16,16], \
                      kernel = (3,3), stride = (2,2), pad = (1,1), act_type='relu', 
                      conv_type = 0, repeat_n = 2)
     
    conv2 = conv_factory(conv1, num_filter = 32, kernel = (3,3), stride=(2,2), 
                         pad = (1,1), act_type = 'relu', conv_type = 0)
    
    conv3 = conv_factory(conv2, num_filter = 32, kernel = (3,3), stride=(1,1), 
                         pad = (1,1), act_type = 'relu', conv_type = 0) 
    
    res_conv4 = plus_conv(conv3 , conv2, num_filter = 32, kernel = (3,3), stride = (1,1), 
              pad=(1,1), act_type = 'relu');
              
    conv5 = conv_factory(res_conv4, num_filter = 48, kernel = (3,3), stride=(2,2), 
                         pad = (1,1), act_type = 'relu', conv_type = 0) 
    
    conv6_1 = conv_factory(conv5, num_filter = 32, kernel = (3,3), stride=(1,1), 
                         pad = (1,1), act_type = 'relu', conv_type = 1) 
    
    res_conv6_1 = plus_conv(res_conv4 , conv6_1, num_filter = 32, 
                            kernel = (1,1), stride = (2,2), pad=(0,0), act_type = 'relu');
                            
    conv6_2 = conv_factory(res_conv6_1, num_filter = 48, kernel = (3,3), stride=(1,1), 
                           pad = (1,1), act_type = 'relu', conv_type = 0)
    
    conv7_1 = mx.symbol.UpSampling(conv6_2, scale = 2, sample_type="nearest")
    
    conv7 = mx.symbol.Concat(*[res_conv4, conv7_1])
    
    return conv7

def get_hobot_cpm_symbol(point_num, is_train=True, with_attribute=False):
    """ CPM standard symbol """
    data = mx.symbol.Variable(name="data")

    conv_feat = get_hobot_conv2(data)
   
    fc6 = conv_factory(conv_feat, num_filter = 64, kernel = (1,1), stride=(1,1), 
                       pad = (0,0), act_type = 'relu', conv_type = 0) 
    
    hobot_cpm_cls_score = mx.symbol.Convolution(
        data=fc6, kernel=(1, 1), pad=(0, 0), num_filter=point_num*2, name='hobot_cpm_cls_score')
    
    hobot_cpm_point_pred = mx.symbol.Convolution(
        data=fc6, kernel=(1, 1), pad=(0, 0), num_filter=point_num*2, name='hobot_cpm_point_pred')

    # prepare scf data
    hobot_cpm_cls_score_reshape = mx.symbol.Reshape(
        data=hobot_cpm_cls_score, shape=(0, 2, -1, 0), name='hobot_cpm_cls_score_reshape')

         
    if is_train:
        hobot_cpm_label = mx.symbol.Variable(name='hobot_cpm_label')
        hobot_cpm_point_target = mx.symbol.Variable(name='hobot_cpm_point_target')
        hobot_cpm_point_weight = mx.symbol.Variable(name='hobot_cpm_point_weight')
        
        
        # classification
        hobot_cpm_cls_loss = mx.symbol.SoftmaxOutput(data=hobot_cpm_cls_score_reshape, label=hobot_cpm_label, multi_output=True,
                                           normalization='valid', use_ignore=True, ignore_label=-1, name='scf_cls_loss')  
        # point regression
        hobot_cpm_point_loss_ = hobot_cpm_point_weight * mx.symbol.smooth_l1(name='hobot_cpm_point_loss_', scalar=3.0, 
                                                                             data=(hobot_cpm_point_pred - hobot_cpm_point_target))
        hobot_cpm_point_loss = mx.sym.MakeLoss(name='hobot_cpm_point_loss', data=hobot_cpm_point_loss_, grad_scale=1.0 / 5)
        
        group = mx.symbol.Group([hobot_cpm_cls_loss, hobot_cpm_point_loss])
        
        return group
    else:
        hobot_cpm_cls_prob = mx.symbol.SoftmaxActivation(data=hobot_cpm_cls_score_reshape, mode='channel', name='hobot_cpm_cls_prob')
        group = mx.symbol.Group([hobot_cpm_cls_prob, hobot_cpm_point_pred])
        
        return group
    
if __name__ == "__main__":
    batch_size = 1 
    
    deploy_symbol = get_hobot_cpm_symbol(False)
    deploy_symbol.save('./scf-hand-deploy.json')
    print(deploy_symbol.list_arguments())
    
    data_shape = (batch_size, 3, 128, 128);
#     label1_shape = (batch_size, 6, 4,4); 
      
    dot = mx.viz.plot_network(deploy_symbol)
    dot.render('test-output/round-table.gv', view=True)
    
        
        
        
        
        
        
        
    
    