import numpy as np
from graph import  Graph, default_graph
import abc

class Node(object):
    """
    计算图节点基类
    """

    def __init__(self, *parents, **kargs):
        self.graph = kargs.get('graph', default_graph)
        self.need_save = kargs.get('need_save', True)
        self.mangle_name(**kargs)
        self.parents = list(parents)
        self.children = []
        self.value = None  # 本节点的值
        self.jacobi = None  # 对本节点参数的梯度矩阵

        # 建立双向链接
        for parent in parents:
            parent.children.append(self)

        self.graph.add_node(self)

    def get_parents(self):
        return self.parents

    def mangle_name(self, **kargs):
        self.name = kargs.get('name', "{}:{}".format(
            self.__class__.__name__, self.graph.node_count()))
        if self.graph.name_scope:
            self.name = "{}/{}".format(self.graph.name_scope, self.name)

    def forward(self):
        for parent in self.parents:
            if parent.value is None:
                parent.forward()

        self.compute()

    @abc.abstractmethod
    def compute(self):
        """
        抽象方法
        """

    def backward(self, res):
        if self.jacobi is None:
            if self is res:
                self.jacobi = np.mat(
                    np.eye(self.dimension())
                )
            else:
                self.jacobi = np.mat(
                    np.zeros(res.dimension(), self.dimension())
                )
                for child in self.children:
                    self.jacobi += child.backward(res) * child.get_jacobi(self)

        return self.jacobi

    @abc.abstractmethod
    def get_jacobi(self, parent):
        """
        计算对父节点梯度矩阵
        """

    def dimension(self):
        return self.value.shape[0] * self.value.shape[1]

    def shape(self):
        return self.value.shape

    def clear_jacobi(self):
        self.jacobi = None

    def clear_value(self, recursive=True):
        self.value = None
        if recursive:
            for child in self.children:
                child.clear_value(recursive)


class Variable(Node):
    """
    变量，继承基类节点
    """

    def __init__(self, dim, init=False, trainable=True, **kargs):
        """

        :param dim: tuple, 包含高和宽
        :param init:
        :param trainable:
        :param kargs:
        """
        Node.__init__(self, **kargs)
        self.dim = dim
        if init:
            self.value = np.mat(
                np.random.normal(0, 0.001, self.dim)
            )
        self.trainable = trainable

    def set_value(self, value):
        self.clear_value()
        self.value = value
