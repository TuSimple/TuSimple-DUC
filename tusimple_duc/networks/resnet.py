import mxnet as mx
no_bias = True
use_global_stats = True
fix_gamma = False
bn_momentum = 0.9995
eps = 1e-6


def Conv(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), dilate=(1, 1), name=None):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad,
                                 dilate=dilate, no_bias=no_bias, name=('%s' % name), workspace=4096)
    return conv


def ReLU(data, name):
    return mx.symbol.Activation(data=data, act_type='relu', name=name)


def Conv_AC(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), dilate=(1, 1), name=None):
    conv = Conv(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, dilate=dilate, name=name)
    act = ReLU(data=conv, name=('%s_relu' % name))
    return act


def Conv_BN(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), dilate=(1, 1), name=None, suffix=''):
    conv = Conv(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, dilate=dilate, name=name)
    bn = mx.symbol.BatchNorm(data=conv, name=('%s/bn' % suffix), eps=eps, use_global_stats=use_global_stats,
                             momentum=bn_momentum, fix_gamma=fix_gamma)
    return bn


def Conv_BN_AC(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), dilate=(1, 1), name=None, suffix=''):
    conv = Conv_BN(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, dilate=dilate,
                   name=name, suffix=suffix)
    act = ReLU(data=conv, name=('%s/relu' % name))
    return act


def ResidualFactory_o(data, num_1x1_a, num_3x3_b, num_1x1_c, dilate, suffix):
    branch1 = Conv_BN(data=data,   num_filter=num_1x1_c, kernel=(1, 1), name=('conv%s_1x1_proj' % suffix),
                      suffix=('conv%s_1x1_proj' % suffix), pad=(0, 0))
    branch2a = Conv_BN_AC(data=data,   num_filter=num_1x1_a, kernel=(1, 1), name=('conv%s_1x1_reduce' % suffix),
                          suffix=('conv%s_1x1_reduce' % suffix), pad=(0, 0))
    branch2b = Conv_BN_AC(data=branch2a, num_filter=num_3x3_b, kernel=(3, 3), name=('conv%s_3x3' % suffix),
                          suffix=('conv%s_3x3' % suffix), pad=dilate, dilate=dilate)
    branch2c = Conv_BN(data=branch2b, num_filter=num_1x1_c, kernel=(1, 1), name=('conv%s_1x1_increase' % suffix),
                       suffix=('conv%s_1x1_increase' % suffix), pad=(0, 0))
    summ = mx.symbol.ElementWiseSum(*[branch2c, branch1], name=('conv%s' % suffix))
    summ_ac = ReLU(data=summ, name=('res%s_relu' % suffix))
    return summ_ac


def ResidualFactory_x(data, num_1x1_a, num_3x3_b, num_1x1_c, dilate, suffix):
    branch2a = Conv_BN_AC(data=data,   num_filter=num_1x1_a, kernel=(1, 1), name=('conv%s_1x1_reduce' % suffix),
                          suffix=('conv%s_1x1_reduce' % suffix), pad=(0, 0))
    branch2b = Conv_BN_AC(data=branch2a, num_filter=num_3x3_b, kernel=(3, 3), name=('conv%s_3x3' % suffix),
                          suffix=('conv%s_3x3' % suffix), pad=dilate, dilate=dilate)
    branch2c = Conv_BN(data=branch2b, num_filter=num_1x1_c, kernel=(1, 1), name=('conv%s_1x1_increase' % suffix),
                       suffix=('conv%s_1x1_increase' % suffix), pad=(0, 0))
    summ = mx.symbol.ElementWiseSum(*[data, branch2c], name=('res%s' % suffix))
    summ_ac = ReLU(data=summ, name=('res%s_relu' % suffix))
    return summ_ac


def ResidualFactory_d(data, num_1x1_a, num_3x3_b, num_1x1_c, suffix):
    branch1 = Conv_BN(data=data, num_filter=num_1x1_c, kernel=(1, 1), name=('conv%s_1x1_proj' % suffix),
                      suffix=('conv%s_1x1_proj' % suffix), pad=(0, 0), stride=(2, 2))
    branch2a = Conv_BN_AC(data=data, num_filter=num_1x1_a, kernel=(1, 1), name=('conv%s_1x1_reduce' % suffix),
                          suffix=('conv%s_1x1_reduce' % suffix), pad=(0, 0), stride=(2, 2))
    branch2b = Conv_BN_AC(data=branch2a, num_filter=num_3x3_b, kernel=(3, 3), name=('conv%s_3x3' % suffix),
                          suffix=('conv%s_3x3' % suffix), pad=(1, 1))
    branch2c = Conv_BN(data=branch2b, num_filter=num_1x1_c, kernel=(1, 1), name=('conv%s_1x1_increase' % suffix),
                       suffix=('conv%s_1x1_increase' % suffix), pad=(0, 0))
    summ = mx.symbol.ElementWiseSum(*[branch2c, branch1], name=('res%s' % suffix))
    summ_ac = ReLU(data=summ, name=('res%s_relu' % suffix))
    return summ_ac


def get_resnet_hdc(bn_use_global_stats=True):
    """
    Get resnet with hybrid dilated convolutions
    Parameters
    ----------
    bn_use_global_stats: whether the batch normalization layers should use global stats

    Returns the symbol generated
    -------

    """
    global use_global_stats
    use_global_stats = bn_use_global_stats

    data = mx.symbol.Variable(name="data")

    # group 1
    res1_1 = Conv_BN_AC(data=data, num_filter=64, kernel=(3, 3), name='conv1_1_3x3_s2', suffix='conv1_1_3x3_s2', pad=(1, 1), stride=(2, 2))
    res1_2 = Conv_BN_AC(data=res1_1, num_filter=64, kernel=(3, 3), name='conv1_2_3x3', suffix='conv1_2_3x3', pad=(1, 1), stride=(1, 1))
    res1_3 = Conv_BN_AC(data=res1_2, num_filter=128, kernel=(3, 3), name='conv1_3_3x3', suffix='conv1_3_3x3', pad=(1, 1), stride=(1, 1))
    pool1 = mx.symbol.Pooling(data=res1_3, pool_type="max", kernel=(3, 3), stride=(2, 2), name="pool1_3x3_s2")

    # group 2
    res2a = ResidualFactory_o(pool1, 64, 64, 256, (1, 1), '2_1')
    res2b = ResidualFactory_x(res2a, 64, 64, 256, (1, 1), '2_2')
    res2c = ResidualFactory_x(res2b, 64, 64, 256, (1, 1), '2_3')

    # group 3
    res3a = ResidualFactory_d(res2c, 128, 128, 512, '3_1')
    res3b1 = ResidualFactory_x(res3a, 128, 128, 512, (1, 1), '3_2')
    res3b2 = ResidualFactory_x(res3b1, 128, 128, 512, (1, 1), '3_3')
    res3b3 = ResidualFactory_x(res3b2, 128, 128, 512, (1, 1), '3_4')

    # group 4
    res4a = ResidualFactory_o(res3b3, 256, 256, 1024, (2, 2), '4_1')
    res4b1 = ResidualFactory_x(res4a, 256, 256, 1024, (2, 2), '4_2')
    res4b2 = ResidualFactory_x(res4b1, 256, 256, 1024, (5, 5), '4_3')
    res4b3 = ResidualFactory_x(res4b2, 256, 256, 1024, (9, 9), '4_4')
    res4b4 = ResidualFactory_x(res4b3, 256, 256, 1024, (1, 1), '4_5')
    res4b5 = ResidualFactory_x(res4b4, 256, 256, 1024, (2, 2), '4_6')
    res4b6 = ResidualFactory_x(res4b5, 256, 256, 1024, (5, 5), '4_7')
    res4b7 = ResidualFactory_x(res4b6, 256, 256, 1024, (9, 9), '4_8')
    res4b8 = ResidualFactory_x(res4b7, 256, 256, 1024, (1, 1), '4_9')
    res4b9 = ResidualFactory_x(res4b8, 256, 256, 1024, (2, 2), '4_10')
    res4b10 = ResidualFactory_x(res4b9, 256, 256, 1024, (5, 5), '4_11')
    res4b11 = ResidualFactory_x(res4b10, 256, 256, 1024, (9, 9), '4_12')
    res4b12 = ResidualFactory_x(res4b11, 256, 256, 1024, (1, 1), '4_13')
    res4b13 = ResidualFactory_x(res4b12, 256, 256, 1024, (2, 2), '4_14')
    res4b14 = ResidualFactory_x(res4b13, 256, 256, 1024, (5, 5), '4_15')
    res4b15 = ResidualFactory_x(res4b14, 256, 256, 1024, (9, 9), '4_16')
    res4b16 = ResidualFactory_x(res4b15, 256, 256, 1024, (1, 1), '4_17')
    res4b17 = ResidualFactory_x(res4b16, 256, 256, 1024, (2, 2), '4_18')
    res4b18 = ResidualFactory_x(res4b17, 256, 256, 1024, (5, 5), '4_19')
    res4b19 = ResidualFactory_x(res4b18, 256, 256, 1024, (9, 9), '4_20')
    res4b20 = ResidualFactory_x(res4b19, 256, 256, 1024, (1, 1), '4_21')
    res4b21 = ResidualFactory_x(res4b20, 256, 256, 1024, (2, 2), '4_22')
    res4b22 = ResidualFactory_x(res4b21, 256, 256, 1024, (5, 5), '4_23')
    # group 5
    res5a = ResidualFactory_o(res4b22, 512, 512, 2048, (5, 5), '5_1')
    res5b = ResidualFactory_x(res5a, 512, 512, 2048, (9, 9), '5_2')
    res5c = ResidualFactory_x(res5b, 512, 512, 2048, (17, 17), '5_3')
    return res5c
