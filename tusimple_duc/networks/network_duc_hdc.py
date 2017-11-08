import mxnet as mx
from resnet import get_resnet_hdc


def get_symbol_duc_hdc(label_num=19, ignore_label=255, bn_use_global_stats=True,
                       aspp_num=4, aspp_stride=6, cell_cap=64, exp="cityscapes"):
    """
    Get
    Parameters
    ----------
    label_num: the number of labels
    ignore_label: id for ignore label
    bn_use_global_stats: whether batch normalizations should use global_stats
    aspp_num: number of ASPPs
    aspp_stride: stride of ASPPs
    cell_cap: capacity of a cell in dense upsampling convolutions
    exp: expression

    Returns
    -------

    """
    # Base Network
    res = get_resnet_hdc(bn_use_global_stats=bn_use_global_stats)

    # ASPP
    aspp_list = list()
    for i in range(aspp_num):
        pad = ((i + 1) * aspp_stride, (i + 1) * aspp_stride)
        dilate = pad
        conv_aspp=mx.symbol.Convolution(data=res, num_filter=cell_cap * label_num, kernel=(3, 3), pad=pad,
                                        dilate=dilate, name=('fc1_%s_c%d' % (exp, i)), workspace=8192)
        aspp_list.append(conv_aspp)

    summ = mx.symbol.ElementWiseSum(*aspp_list, name=('fc1_%s' % exp))

    cls_score_reshape = mx.symbol.Reshape(data=summ, shape=(0, label_num, -1), name='cls_score_reshape')
    cls = mx.symbol.SoftmaxOutput(data=cls_score_reshape, multi_output=True,
                                  normalization='valid', use_ignore=True, ignore_label=ignore_label, name='seg_loss')
    return cls

if __name__ == '__main__':
    symbol = get_symbol_duc_hdc(label_num=19, cell_cap=16)

    t = mx.viz.plot_network(symbol, shape={'data': (3, 3, 480, 480)})
    t.render()
