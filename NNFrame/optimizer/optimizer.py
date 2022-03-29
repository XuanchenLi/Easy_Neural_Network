import abc
import numpy as np
from ..core.node import *
from ..core.graph import *
from ..core.core import *


class Optimizer(object):
    def __init__(self, graph, target, learning_rate=0.01):
        self.graph = graph
        self.target = target
        self.learning_rate = learning_rate
        self.acc_grad = dict()
        self.acc_t = 0

    def step(self):
        self.forward_backward()
        self.acc_t += 1

    def get_grad(self, node):
        return self.acc_grad[node] / self.acc_t

    @abc.abstractmethod
    def update_aux(self):
        pass

    def update(self, var_grad):
        if var_grad is not None:
            self.apply_grad(var_grad)

        self.update_aux()
        self.acc_grad.clear()
        self.acc_t = 0

    def apply_grad(self, var_grad, use_sum=False, acc_t=None):
        for node, grad in self.acc_grad.items():
            if isinstance(node, Node):
                pass
            else:
                target_node = get_node_from_graph(node)
                if use_sum:
                    self.acc_grad[target_node] += grad
                else:
                    self.acc_grad[target_node] = grad

        if use_sum:
            self.acc_t += acc_t
        else:
            if acc_t is None:
                self.acc_t = 1
            else:
                self.acc_t = acc_t

    def forward_backward(self):
        self.graph.clear_jacobi()
        self.target.forward()
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                node.backward(self.target)
                grad = node.jacobi.T.reshape(node.shape())
                if node not in self.acc_grad:
                    self.acc_grad[node] = grad
                else:
                    self.acc_grad[node] += grad


class GradientDescent(Optimizer):
    def __init__(self, graph, target, learning_rate=0.01):
        Optimizer.__init__(self, graph, target, learning_rate)

    def update_aux(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                grad = self.get_grad(node)
                node.set_value(node.value - self.learning_rate * grad)


class Momentum(Optimizer):
    def __init__(self, graph, target, learning_rate=0.01, momentum=0.9):
        Optimizer.__init__(self, graph, target, learning_rate)
        self.momentum = momentum
        self.v = dict()

    def update_aux(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                grad = self.get_grad(node)

                if node not in self.v:
                    self.v[node] = - self.learning_rate * grad
                else:
                    self.v[node] = self.momentum * self.v[node] \
                                   - self.learning_rate * grad
                node.set_value(node.value + self.v[node])


class AdaGrad(Optimizer):
    def __init__(self, graph, target, learning_rate=0.01):
        Optimizer.__init__(self, graph, target, learning_rate)

        self.s = dict()

    def update_aux(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                grad = self.get_grad(node)

                if node not in self.s:
                    self.s[node] = np.power(grad, 2)
                else:
                    self.s[node] = self.s[node] + np.power(grad, 2)
                node.set_value(node.value -
                               self.learning_rate * grad / (np.sqrt(self.s[node]) + 1e-7))


class RMSProp(Optimizer):
    def __init__(self, graph, target, learning_rate=0.01, decay=0.9):
        Optimizer.__init__(self, graph, target, learning_rate)

        self.s = dict()
        self.decay = decay

    def update_aux(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                grad = self.get_grad(node)

                if node not in self.s:
                    self.s[node] = np.power(grad, 2)
                else:
                    self.s[node] = self.decay * self.s[node] + \
                        (1 - self.decay) * np.power(grad, 2)
                node.set_value(node.value -
                               self.learning_rate * grad / (np.sqrt(self.s[node] + 1e-6)))


class Adam(Optimizer):
    def __init__(self, graph, target, learning_rate=0.01, decay_1=0.9, decay_2=0.99):
        Optimizer.__init__(self, graph, target, learning_rate)

        self.s = dict()
        self.v = dict()
        self.decay_1 = decay_1
        self.decay_2 = decay_2

    def update_aux(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                grad = self.get_grad(node)

                if node not in self.s:
                    self.s[node] = grad
                    self.v[node] = np.power(grad, 2)
                else:
                    self.s[node] = self.decay_1 * self.s[node] + \
                        (1 - self.decay_1) * grad

                    self.v[node] = self.decay_2 * self.v[node] + \
                        (1 - self.decay_2) * np.power(grad, 2)

                node.set_value(node.value -
                               self.learning_rate * self.s[node] / (np.sqrt(self.v[node]) + 1e-8))

