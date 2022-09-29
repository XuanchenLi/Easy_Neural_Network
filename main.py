from neuralNetwork import NeuralNetwork
from sklearn.datasets import load_iris
import random
import os
import struct
import numpy as np

PATH = './'


def get_data(dataset, kind, seed=0):
    if dataset == "MNIST":
        labels_p = os.path.join(PATH, 'MNIST', '%s-labels.idx1-ubyte'%kind)
        images_p = os.path.join(PATH, 'MNIST', '%s-images.idx3-ubyte'%kind)
        with open(labels_p, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)

        ls = []
        for i in range(labels.size):
            temp = np.zeros(10)
            temp[labels[i]] = 1
            ls.append(temp.T)
        labels = np.array(ls)

        with open(images_p, 'rb') as imgpath:
            magic, n, rows, cols = struct.unpack(">IIII", imgpath.read(16))
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

            images = images / 2550
        return images, labels
    elif dataset == "Iris":
        iris = load_iris()
        N, _ = iris.data.shape
        index = [i for i in range(len(iris.data))]
        random.seed(seed)
        random.shuffle(index)
        attributes = np.array(iris.data[index])
        target = np.array(iris.target[index])
        if kind == "train":
            attributes = attributes[:int(N*0.7)]
            target = target[:int(N*0.7)]
        elif kind == "valid":
            attributes = attributes[int(N * 0.7 + 1):]
            target = target[int(N * 0.7 + 1):]
        else:
            raise ValueError("dataset not exist")
        ls = []
        for i in range(target.size):
            temp = np.zeros(3)
            temp[target[i]] = 1
            ls.append(temp.T)
        target = np.array(ls)
        return attributes, target
    else:
        raise ValueError("dataset not exist")


if __name__ == '__main__':
    import sys

    sys.path.append('../..')
    from sklearn.datasets import load_iris
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    import NNF as nf
    from NNF.core.node import *
    from NNF.core.graph import *
    from NNF.operators.ops import *
    from NNF.operators.loss import *
    from NNF.optimizer.optimizer import *
    from NNF.layer.layer import *
    import matplotlib.pyplot as plt

    # 读取鸢尾花数据集，去掉第一列Id
    data = pd.read_csv("iris.csv").drop("Id", axis=1)
    iris = load_iris()  # 载入鸢尾花数据集
    datas = iris.data  # (150,4)：150 行 5 列
    x1 = [x[0] for x in datas]  # 第1列：花萼长度 sepal_length
    x2 = [x[1] for x in datas]  # 第2列：花萼宽度 sepal_width
    x3 = [x[2] for x in datas]  # 第3列：花瓣长度 petal_length
    x4 = [x[3] for x in datas]  # 第4列：花瓣宽度 petal_width
    plt.scatter(x1[:50], x2[:50], color='red', marker='o', label='setosa')  # 前50个样本
    plt.scatter(x1[50:100], x2[50:100], color='blue', marker='x', label='vericolor')  # 中间50个样本
    plt.scatter(x1[100:150], x2[100:150], color='green', marker='+', label='Virginica')  # 后50个样本
    plt.legend(loc=1)  # loc=1，2，3，4分别表示label在右上角，左上角，左下角，右下角
    plt.show()

    # print(datas)
    # print(iris.target)

    # 随机打乱样本顺序
    data = data.sample(len(data), replace=False)

    # 将字符串形式的类别标签转换成整数0，1，2
    le = LabelEncoder()
    number_label = le.fit_transform(data["Species"])

    # 将整数形式的标签转换成One-Hot编码
    oh = OneHotEncoder(sparse=False)
    one_hot_label = oh.fit_transform(number_label.reshape(-1, 1))

    # 特征列
    features = data[['SepalLengthCm',
                     'SepalWidthCm',
                     'PetalLengthCm',
                     'PetalWidthCm']].values

    # 构造计算图：输入向量，是一个4x1矩阵，不需要初始化，不参与训练
    x = Variable(dim=(4, 1), init=False, trainable=False)

    # One-Hot类别标签，是3x1矩阵
    one_hot = Variable(dim=(3, 1), init=False, trainable=False)

    # 第一隐藏层，10个神经元，激活函数为ReLU
    hidden_1 = fc(x, 4, 10, "ReLU")

    # 第二隐藏层，10个神经元，激活函数为ReLU
    hidden_2 = fc(hidden_1, 10, 10, "ReLU")

    # 输出层，3个神经元，无激活函数
    output = fc(hidden_2, 10, 3, None)

    # 模型输出概率
    predict = SoftMax(output)

    # 交叉熵损失函数
    loss = CrossEntropyWithSoftMax(output, one_hot)

    # 学习率
    learning_rate = 0.02

    # 构造Adam优化器
    optimizer = Adam(default_graph, loss, learning_rate)

    # 批大小为16
    batch_size = 16

    end = []
    # 训练执行10个epoch
    for epoch in range(30):

        # 批计数器清零
        batch_count = 0

        # 遍历训练集中的样本
        for i in range(len(features)):

            # 取第i个样本，构造4x1矩阵对象
            feature = np.mat(features[i, :]).T

            # 取第i个样本的One-Hot标签，3x1矩阵
            label = np.mat(one_hot_label[i, :]).T

            # 将特征赋给x节点，将标签赋给one_hot节点
            x.set_value(feature)
            one_hot.set_value(label)

            # 调用优化器的one_step方法，执行一次前向传播和反向传播
            optimizer.step()

            # 批计数器加1
            batch_count += 1

            # 若批计数器大于等于批大小，则执行一次更新，并清零计数器
            if batch_count >= batch_size:
                optimizer.update()
                batch_count = 0

        # 每个epoch结束后评估模型的正确率
        pred = []

        # 遍历训练集，计算当前模型对每个样本的预测概率
        for i in range(len(features)):
            feature = np.mat(features[i, :]).T
            x.set_value(feature)

            # 在模型的predict节点上执行前向传播
            predict.forward()
            pred.append(predict.value.A.ravel())  # 模型的预测结果：3个概率值

        pred = np.array(pred).argmax(axis=1)  # 取最大概率对应的类别为预测类别

        # 判断预测结果与样本标签相同的数量与训练集总数量之比，即模型预测的正确率
        accuracy = (number_label == pred).astype(np.int).sum() / len(data)
        end.append(accuracy)
        # 打印当前epoch数和模型在训练集上的正确率
        print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))

        # x0 = pred[pred == 0]
        # x1 = pred[pred == 1]
        # x2 = pred[pred == 2]
        # plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
        # plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
        # plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
    plt.plot(end)

    plt.show();