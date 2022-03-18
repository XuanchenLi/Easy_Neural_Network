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
    train_x, train_y = get_data("Iris", "train", 100)
    valid_x, valid_y = get_data("Iris", "valid", 100)
    node_num = [4, 30, 3]
    network = NeuralNetwork(2, node_num, 0.3, 50)
    network.set_activate("sigmoid")
    network.gaussian_weights()
    network.fit(train_x, train_y)
    acc = network.test(valid_x, valid_y)
    print(acc)
    '''
    train_x, train_y = get_data("MNIST", "train")
    valid_x, valid_y = get_data("MNIST", "t10k")
    node_num = [784, 30, 10]
    network = NeuralNetwork(2, node_num, 0.3, 5)
    network.set_activate("sigmoid")
    network.gaussian_weights()
    network.fit(train_x, train_y)
    acc = network.test(valid_x, valid_y)
    print(acc)
    '''

