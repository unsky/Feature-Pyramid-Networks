import mxnet as mx

eps = 2e-5
USE_GLOBAL_STATS = True
workspace = 512
res_deps = {'18': (2, 2, 2, 2), '34': (3, 4, 6, 3), '50': (3, 4, 6, 3), '101': (3, 4, 23, 3), '152': (3, 8, 36, 3), '200': (3, 24, 36, 3)}
# units = res_deps['50']
# filter_list = [256, 512, 1024, 2048]


def residual_unit(data, num_filter, stride, dim_match, name, use_global_stats=USE_GLOBAL_STATS, bn_mom=0.9, bottle_neck=True, dilate=(1, 1)):
    
    if bottle_neck:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1), pad=(0, 0), dilate=dilate, 
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride, pad=(1, 1), dilate=dilate, 
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, momentum=bn_mom, name=name + '_bn3')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True, dilate=dilate, 
                                   workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True, dilate=dilate, 
                                          workspace=workspace, name=name + '_sc')
        sum = mx.sym.ElementWiseSum(*[conv3, shortcut], name=name + '_plus')
        return sum
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, use_global_stats=use_global_stats, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1), dilate=dilate, 
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, use_global_stats=use_global_stats, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), dilate=dilate, 
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True, dilate=dilate, 
                                            workspace=workspace, name=name+'_sc')
            
        sum = mx.sym.ElementWiseSum(*[conv2, shortcut], name=name + '_plus')
        return sum


def residual_unit_alphacnn(data, num_filter, stride, dim_match, name, bottle_neck=True):
    
    if bottle_neck:
        act1 = mx.sym.Activation(data=data, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.AlphaConvolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        act2 = mx.sym.Activation(data=conv1, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.AlphaConvolution(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride, pad=(1, 1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        act3 = mx.sym.Activation(data=conv2, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.AlphaConvolution(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.AlphaConvolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                          workspace=workspace, name=name + '_sc')
        sum = mx.sym.ElementWiseSum(*[conv3, shortcut], name=name + '_plus')
        return sum
    else:
        act1 = mx.sym.Activation(data=data, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.AlphaConvolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        act2 = mx.sym.Activation(data=conv1, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.AlphaConvolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.AlphaConvolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                               workspace=workspace, name=name+'_sc')
            
        return conv2 + shortcut

def get_resnet_alphacnn_conv(data, depth):    
    units = res_deps[str(depth)]
    filter_list = [256, 512, 1024, 2048] if depth >= 50 else [64, 128, 256, 512]
    bottle_neck = True if depth >= 50 else False
    
    # res1
    conv0 = mx.sym.Convolution(data=data, num_filter=64, kernel=(5, 5), stride=(2, 2), pad=(2, 2),
                               no_bias=True, name="conv0", workspace=workspace)
    bn0 = mx.sym.BatchNorm(data=conv0, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn0')
    relu0 = mx.sym.Activation(data=bn0, act_type='relu', name='relu0')
    pool0 = mx.symbol.Pooling(data=relu0, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool0')

    # res2
    unit = residual_unit(data=pool0, num_filter=filter_list[0], stride=(1, 1), dim_match=False, name='stage1_unit1',
                         bottle_neck=bottle_neck)
    for i in range(2, units[0] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[0], stride=(1, 1), dim_match=True, name='stage1_unit%s' % i,
                             bottle_neck=bottle_neck)

    # res3
    unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(2, 2), dim_match=False, name='stage2_unit1',
                         bottle_neck=bottle_neck)
    for i in range(2, units[1] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(1, 1), dim_match=True, name='stage2_unit%s' % i,
                             bottle_neck=bottle_neck)

    # res4
    unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(2, 2), dim_match=False, name='stage3_unit1',
                         bottle_neck=bottle_neck)
    for i in range(2, units[2] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(1, 1), dim_match=True, name='stage3_unit%s' % i,
                             bottle_neck=bottle_neck)
    return unit

def get_resnet_conv(data, depth):    
    units = res_deps[str(depth)]
    filter_list = [256, 512, 1024, 2048] if depth >= 50 else [64, 128, 256, 512]
    bottle_neck = True if depth >= 50 else False
    
    # res1
    data_bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=eps, use_global_stats=USE_GLOBAL_STATS, name='bn_data')
    conv0 = mx.sym.Convolution(data=data_bn, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                               no_bias=True, name="conv0", workspace=workspace)
    bn0 = mx.sym.BatchNorm(data=conv0, fix_gamma=False, eps=eps, use_global_stats=USE_GLOBAL_STATS, name='bn0')
    relu0 = mx.sym.Activation(data=bn0, act_type='relu', name='relu0')
    pool0 = mx.symbol.Pooling(data=relu0, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool0')

    # res2
    unit = residual_unit(data=pool0, num_filter=filter_list[0], stride=(1, 1), dim_match=False, name='stage1_unit1',
                         bottle_neck=bottle_neck)
    for i in range(2, units[0] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[0], stride=(1, 1), dim_match=True, name='stage1_unit%s' % i,
                             bottle_neck=bottle_neck)

    # res3
    unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(2, 2), dim_match=False, name='stage2_unit1',
                         bottle_neck=bottle_neck)
    for i in range(2, units[1] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(1, 1), dim_match=True, name='stage2_unit%s' % i,
                             bottle_neck=bottle_neck)

    # res4
    unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(2, 2), dim_match=False, name='stage3_unit1',
                         bottle_neck=bottle_neck)
    for i in range(2, units[2] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(1, 1), dim_match=True, name='stage3_unit%s' % i,
                             bottle_neck=bottle_neck)
    return unit


def get_resnet_conv5(data, depth): 
    units = res_deps[str(depth)]
    filter_list = [256, 512, 1024, 2048] if depth >= 50 else [64, 128, 256, 512]
    bottle_neck = True if depth >= 50 else False
    dilate = (1,1)
    
    # res1
    data_bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=eps, use_global_stats=USE_GLOBAL_STATS, name='bn_data')
    conv0 = mx.sym.Convolution(data=data_bn, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                               no_bias=True, name="conv0", workspace=workspace)
    bn0 = mx.sym.BatchNorm(data=conv0, fix_gamma=False, eps=eps, use_global_stats=USE_GLOBAL_STATS, name='bn0')
    relu0 = mx.sym.Activation(data=bn0, act_type='relu', name='relu0')
    pool0 = mx.symbol.Pooling(data=relu0, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool0')

    # res2
    unit = residual_unit(data=pool0, num_filter=filter_list[0], stride=(1, 1), dim_match=False, name='stage1_unit1',
                         bottle_neck=bottle_neck)
    for i in range(2, units[0] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[0], stride=(1, 1), dim_match=True, name='stage1_unit%s' % i,
                             bottle_neck=bottle_neck)

    # res3
    unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(2, 2), dim_match=False, name='stage2_unit1',
                         bottle_neck=bottle_neck)
    for i in range(2, units[1] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(1, 1), dim_match=True, name='stage2_unit%s' % i,
                             bottle_neck=bottle_neck)

    # res4
    unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(2, 2), dim_match=False, name='stage3_unit1',
                         bottle_neck=bottle_neck)
    for i in range(2, units[2] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(1, 1), dim_match=True, name='stage3_unit%s' % i,
                             bottle_neck=bottle_neck)
    # res5    
    unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(1, 1), dim_match=False, name='stage4_unit1', dilate=dilate)
    for i in range(2, units[3] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(1, 1), dim_match=True, name='stage4_unit%s' % i, dilate=dilate)

    return unit

def get_fpn_resnet_conv(data, depth): #add bn to fpn layer:2017-08-01
    units = res_deps[str(depth)]
    filter_list = [256, 512, 1024, 2048, 256] if depth >= 50 else [64, 128, 256, 512, 256]

    bottle_neck = True if depth >= 50 else False
    
    # res1
    data_bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=eps, use_global_stats=USE_GLOBAL_STATS, name='bn_data')
    conv0 = mx.sym.Convolution(data=data_bn, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                               no_bias=True, name="conv0", workspace=workspace)
    bn0 = mx.sym.BatchNorm(data=conv0, fix_gamma=False, eps=eps, use_global_stats=USE_GLOBAL_STATS, name='bn0')
    relu0 = mx.sym.Activation(data=bn0, act_type='relu', name='relu0')
    pool0 = mx.symbol.Pooling(data=relu0, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool0')

    # res2
    conv1 = residual_unit(data=pool0, num_filter=filter_list[0], stride=(1, 1), dim_match=False, name='stage1_unit1',
                         bottle_neck=bottle_neck)
    for i in range(2, units[0] + 1):
        conv1 = residual_unit(data=conv1, num_filter=filter_list[0], stride=(1, 1), dim_match=True, name='stage1_unit%s' % i,
                             bottle_neck=bottle_neck)
    #stride = 4

    # res3
    conv2 = residual_unit(data=conv1, num_filter=filter_list[1], stride=(2, 2), dim_match=False, name='stage2_unit1',
                         bottle_neck=bottle_neck)
    for i in range(2, units[1] + 1):
        conv2 = residual_unit(data=conv2, num_filter=filter_list[1], stride=(1, 1), dim_match=True, name='stage2_unit%s' % i,
                             bottle_neck=bottle_neck)
    # stride = 8
    # res4
    conv3 = residual_unit(data=conv2, num_filter=filter_list[2], stride=(2, 2), dim_match=False, name='stage3_unit1',
                         bottle_neck=bottle_neck)
    for i in range(2, units[2] + 1):
        conv3 = residual_unit(data=conv3, num_filter=filter_list[2], stride=(1, 1), dim_match=True, name='stage3_unit%s' % i,
                             bottle_neck=bottle_neck)
    #stride = 16
    # res5
    conv4 = residual_unit(data=conv3, num_filter=filter_list[3], stride=(2, 2), dim_match=False, name='stage4_unit1',
                         bottle_neck=bottle_neck)
    for i in range(2, units[3] + 1):
        conv4 = residual_unit(data=conv4, num_filter=filter_list[3], stride=(1, 1), dim_match=True, name='stage4_unit%s' % i,
                             bottle_neck=bottle_neck)
    # bn4 = mx.sym.BatchNorm(data=conv4, fix_gamma=False, eps=eps, use_global_stats=USE_GLOBAL_STATS, momentum=0.9, name='stage5_bn1')
    # act4 = mx.sym.Activation(data=bn4, act_type='relu', name='stage5_relu1')
    #stride = 32

    #stride = 64
    # de-res5
    up_conv5_out = mx.symbol.Convolution(data=conv4, kernel=(1, 1), pad=(0, 0), num_filter=filter_list[4], name='stage5_conv_1x1')

    up_conv4 = mx.symbol.UpSampling(up_conv5_out, scale=2, sample_type="nearest")
    #bn_up_conv4 = mx.sym.BatchNorm(data=up_conv4, fix_gamma=False, eps=eps, use_global_stats=USE_GLOBAL_STATS, momentum=0.9, name='up_conv4_bn1')
    conv3_1 = mx.symbol.Convolution(
        data=conv3, kernel=(1, 1), pad=(0, 0), num_filter=filter_list[4], name='up_stage4_conv_1x1')
    #bn_conv3_1 = mx.sym.BatchNorm(data=conv3_1, fix_gamma=False, eps=eps, use_global_stats=USE_GLOBAL_STATS, momentum=0.9, name='conv3_1_bn1')
    up_conv4_Crop = mx.sym.Crop(up_conv4,conv3_1)  
    up_conv4_ = up_conv4_Crop + conv3_1
    up_conv4_out = mx.symbol.Convolution(
        data=up_conv4_, kernel=(3, 3), pad=(1, 1), num_filter=filter_list[4], name='up_stage4_conv_3x3')
     
    # de-res4
    up_conv3 = mx.symbol.UpSampling(up_conv4_out, scale=2, sample_type="nearest")
    #bn_up_conv3 = mx.sym.BatchNorm(data=up_conv3, fix_gamma=False, eps=eps, use_global_stats=USE_GLOBAL_STATS, momentum=0.9, name='up_conv3_bn1')
    conv2_1 = mx.symbol.Convolution(
        data=conv2, kernel=(1, 1), pad=(0, 0), num_filter=filter_list[4], name='up_stage3_conv_1x1')
    #bn_conv2_1 = mx.sym.BatchNorm(data=conv2_1, fix_gamma=False, eps=eps, use_global_stats=USE_GLOBAL_STATS, momentum=0.9, name='conv2_1_bn1')
    up_conv3_Crop = mx.sym.Crop(up_conv3,conv2_1)  
    up_conv3_ = up_conv3_Crop + conv2_1
    up_conv3_out = mx.symbol.Convolution(
        data=up_conv3_, kernel=(3, 3), pad=(1, 1), num_filter=filter_list[4], name='up_stage3_conv_3x3')
    
    # de-res3
    up_conv2 = mx.symbol.UpSampling(up_conv3_out, scale=2, sample_type="nearest") 
    #bn_up_conv2 = mx.sym.BatchNorm(data=up_conv2, fix_gamma=False, eps=eps, use_global_stats=USE_GLOBAL_STATS, momentum=0.9, name='up_conv2_bn1')
    conv1_1 = mx.symbol.Convolution(
        data=conv1, kernel=(1, 1), pad=(0, 0), num_filter=filter_list[4], name='up_stage2_conv_1x1')
    #bn_conv1_1 = mx.sym.BatchNorm(data=conv1_1, fix_gamma=False, eps=eps, use_global_stats=USE_GLOBAL_STATS, momentum=0.9, name='conv1_1_bn1')
    up_conv2_Crop = mx.sym.Crop(up_conv2,conv1_1)  

    up_conv2_ = up_conv2_Crop + conv1_1
    up_conv2_out = mx.symbol.Convolution(
        data=up_conv2_, kernel=(3, 3), pad=(1, 1), num_filter=filter_list[4], name='up_stage2_conv_3x3')
    
    output = []
    output.append(up_conv2_out)#stride:4
    output.append(up_conv3_out)#stride:8
    output.append(up_conv4_out)#stride:16
    output.append(up_conv5_out)#stride:32

    return output



