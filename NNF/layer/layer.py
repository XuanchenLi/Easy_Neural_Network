from NNF.core.node import *
from operators.ops import *


def fc(input, input_size, size, activation):
    """

    :param input_size: 输入维数
    :param size: 神经元数
    :param activation: 激活函数
    """
    weights = Variable((size, input_size), init=True, trainable=True)
    bias = Variable((size, 1), init=True, trainable=True)
    affine = Add(MatMul(weights, input), bias)

    if activation == "ReLU":
        return LeakyReLU(affine)
    elif activation == "Logistic":
        return Logistic(affine)
    else:
        return affine


def conv(feature_maps, input_shape, kernels, kernel_shape, activation):
    """

    :param feature_maps: 输入特征图组（多个同形状的值节点）
    :param input_shape: tuple，特征图形状
    :param kernels: 卷积核数
    :param kernel_shape: 卷积核尺寸
    :param activation: 激活函数
    :return: 输出特征图组
    """
    bias_aux = Variable(input_shape, init=False, trainable=False)
    bias_aux.set_value(np.mat(np.ones(input_shape)))
    outputs = []
    for i in range(kernels):
        channels = []
        for fm in feature_maps:
            kernel = Variable(kernel_shape, init=True, trainable=True)
            conv_ = Convolve(fm, kernel)
            channels.append(conv_)

        chs = Add(*channels)
        bias = ScalarMultiply(
            Variable((1, 1), init=True, trainable=True), bias_aux
        )
        affine = Add(chs, bias)

        if activation == "ReLU":
            outputs.append(LeakyReLU(0.0, affine))
        elif activation == "Logistic":
            outputs.append(Logistic(affine))
        else:
            outputs.append(affine)

    return outputs


def max_pooling(feature_maps, kernel_shape, stride):
    outputs = []
    for fm in feature_maps:
        outputs.append(MaxPooling(fm, size=kernel_shape, stride=stride))
