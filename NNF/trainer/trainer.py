import abc
import numpy as np
from NNF.core import *


class Trainer(object):
    """
    训练器就基类
    """
    def __init__(self, input_x, input_y,
                 loss, optimizer, epoches,
                 batch_size=8, eval_on_train=False,
                 metrics=None, *args, **kargs):
        self.input_x = input_x  # 输入节点, list类型
        self.input_y = input_y  # 标签节点
        self.loss = loss
        self.optimizer = optimizer
        self.epoches = epoches
        self.epoch = 0
        self.batch_size = batch_size
        self.eval_on_train = eval_on_train
        self.metrics = metrics

    def train_and_eval(self, train_x, train_y, test_x=None, test_y=None):
        self.variable_weights_init()
        self.main_loop(train_x, train_y, test_x, test_y)

    @abc.abstractmethod
    def variable_weights_init(self):
        raise NotImplementedError()

    def main_loop(self, train_x, train_y, test_x, test_y):
        for self.epoch in range(self.epoches):
            self.train(train_x, train_y)
            if self.eval_on_train and test_x is not None and test_y is not None:
                self.eval(test_x, test_y)

    def train(self, train_x, train_y):
        for i in range(len(list(train_x.values())[0])):
            self.step(self.get_input_values(train_x, i), train_y[i])
            if (i + 1) % self.batch_size == 0:
                self.optimizer_update()

    def get_input_values(self, data_x, index):
        """
        :param index: 索引
        :param data_x: dict类型数据集
        """
        input_values = dict()
        for input_name in data_x.keys():
            input_values[input_name] = data_x[input_name][index]
        return input_values

    def step(self, data_x, data_y, is_eval=False):
        for i in range(len(self.input_x)):
            input_value = data_x.get(self.input_x[i].name)
            self.input_x[i].set_value(np.mat(input_value).T)
        self.input_y.set_value(np.mat(data_y).T)
        if not is_eval:
            self.optimizer.step()

    @abc.abstractmethod
    def optimizer_update(self):
        raise NotImplementedError()

    def eval(self, test_x, test_y):
        for metric in self.metrics:
            metric.clear_value()

        for i in range(len(list(test_x)[0])):
            self.step(self.get_input_values(test_x, i),
                      test_y[i], is_eval=True)
            for metric in self.metrics:
                metric.forward()
        metric_log = 'Epoch [{}] metrics '.format(self.epoch + 1)
        for metric in self.metrics:
            metric_log += metric.value_str()
        print(metric_log)
