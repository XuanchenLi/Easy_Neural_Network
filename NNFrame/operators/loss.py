import numpy as np
from ..core.node import Node
from .ops import SoftMax


class LossFunction(Node):
    pass


class LogLoss(LossFunction):
    """
    对数损失函数
    """
    def compute(self):
        x = self.parents[0].value
        self.value = np.log(1 + np.where(-x > 1e2, 1e2, -x))  # 避免溢出

    def get_jacobi(self, parent):
        x = parent.value
        dx = -1 / (1 + np.power(np.e, np.where(x > 1e2, 1e2, x)))
        return np.diag(dx.ravel())


class CrossEntropyWithSoftMax(LossFunction):
    """
    对第一个父节点施加SoftMax，以第二个父节点为One-Hot
    """
    def compute(self):
        prob = SoftMax.softmax(self.parents[0].value)
        self.value = np.mat(
            -np.sum(np.multiply(self.parents[1].value, np.log(prob + 1e-10)))
        )

    def get_jacobi(self, parent):
        prob = SoftMax.softmax(self.parents[0].value)
        if parent is parent[0]:
            return (prob - self.parents[1].value).T
        else:
            return (-np.log(prob)).T
