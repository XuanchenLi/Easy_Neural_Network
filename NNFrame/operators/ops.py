import numpy as np
from ..core.node import Node


def fill_diagonal(pos, filler):
    """
    将某矩阵填充在对角线上
    """
    n = int(pos.shape[0] / filler.shape[0])

    r, c = filler.shape
    for i in range(n):
        pos[i * r:(i + 1) * r, i * c:(i + 1) * c] = filler

    return pos


class Operator(Node):
    """
    trivial class
    """
    pass


class Add(Operator):
    def compute(self):
        self.value = np.mat(
            np.zeros(self.parents[0].shape())
        )
        for parent in self.parents:
            self.value += parent.value

    def get_jacobi(self, parent):
        return np.mat(np.eye(self.dimension()))


class Multiply(Operator):
    def compute(self):
        self.value = np.multiply(self.parents[0].value, self.parents[1].value)

    def get_jacobi(self, parent):
        if parent is self.parents[0]:
            return np.diag(self.parents[1].value.A1)
        else:
            return np.diag(self.parents[0].value.A1)


class MatMul(Operator):
    def compute(self):
        assert len(self.parents) == 2 and \
               self.parents[0].shape()[1] == self.parents[1].shape()[0]
        self.value = self.parents[0].value * self.parents[1].value

    def get_jacobi(self, parent):
        zeros = np.mat(
            np.zeros(self.dimension(), parent.dimension())
        )
        if parent is self.parents[0]:
            return fill_diagonal(zeros, self.parents[1].value.T)
        else:
            jacobi = fill_diagonal(zeros, parent[0].value)
            r_sort = np.arange(self.dimension()).reshape(
                self.shape()[::-1]
            ).T.ravel()
            c_sort = np.arange(parent.dimension()).reshape(
                parent.shape()[::-1]
            ).T.ravel()
            return jacobi[r_sort, :][:, c_sort]


class Logistic(Operator):
    def compute(self):
        x = self.parents[0].value
        self.value = np.mat(
            1.0 / (1.0 + np.power(np.e, np.where(-x > 1e2, 1e2, -x)))
        )

    def get_jacobi(self, parent):
        return np.diag(
            np.mat(
                np.multiply(self.value, 1 - self.value)
            ).A1
        )


class SoftMax(Operator):
    @staticmethod
    def softmax(x):
        x[x > 1e2] = 1e2
        ep = np.power(np.e, x)
        return ep / np.sum(ep)

    def compute(self):
        self.value = SoftMax.softmax(self.parents[0].value)

    def get_jacobi(self, parent):
        """
        trivial function
        与交叉熵配合使用，无需实现
        """


class LeakyReLU(Operator):
    def __init__(self, nslope=0, **kargs):
        Operator.__init__(**kargs)
        self.nslope = nslope

    def compute(self):
        self.value = np.mat(
            np.where(self.parents[0].value > 0.0,
                     self.parents[0].value,
                     self.parents[0].value * self.nslope
                     )
        )

    def get_jacobi(self, parent):
        return np.diag(
            np.where(self.parents[0].value.A1 > 0.0,
                     1.0, self.nslope)
        )

    def set_leakey(self, nslope):
        self.nslope = nslope


class Reshape(Operator):
    """
    改变矩阵形状
    """
    def __init__(self, *parent, **kargs):
        Operator.__init__(self, *parent, **kargs)
        self.to_shape = kargs.get('shape')

    def compute(self):
        self.value = self.parents[0].value.reshape(self.to_shape)

    def get_jacobi(self, parent):
        return np.mat(
             np.eye(self.dimension())
        )


class Concat(Operator):
    """
    拼接父节点向量
    """
    def compute(self):
        self.value = np.concatenate(
            [parent.value.flatten() for parent in self.parents],
            axis=1
        ).T

    def get_jacobi(self, parent):
        dims = [parent.dimension() for parent in self.parents]
        pos = self.parents.index(parent)
        dimension = parent.dimension

        jacobi = np.mat(
            np.zeros(self.dimension(), dimension)
        )
        s_row = int(np.sum(dims[:pos]))
        jacobi[s_row:s_row + dimension, 0:dimension] = np.eye(dimension)
        return jacobi


class Welding(Operator):
    """
    挂钩节点
    可用于灵活建图
    """
    def compute(self):
        self.value = self.parents[0].value

    def get_jacobi(self, parent):
        return np.mat(
            np.eye(self.dimension())
        )

    def weld(self, new_parent):
        if len(self.parents) == 1 and self.parents[0] is not None:
            self.parents[0].children.remove(self)
        self.parents.clear()
        self.parents.append(new_parent)
        new_parent.children.append(self)


class Convolve(Operator):
    def __init__(self, *parents, **kargs):
        Operator.__init__(*parents, **kargs)
        self.padded = None

    def compute(self):
        data = self.parents[0].value
        kernel = self.parents[1].value
        w, h = data.shape
        kw, kh = kernel.shape
        half_kw, half_kh = int(kw/2), int(kh/2)
        pw, ph = tuple(
            np.add(data.shape,
                   np.multiply((half_kw, half_kh), 2)
                   )
        )
        self.padded = np.mat(np.zeros((pw, ph)))
        self.padded[half_kw:half_kw + w, half_kh:half_kh + h] = data
        self.value = np.mat(
            np.zeros((w, h))
        )
        for i in range(half_kw, half_kw + w):
            for j in range(half_kh, half_kh + h):
                self.value[i - half_kw, j-half_kw] = np.sum(
                    np.multiply(
                        self.padded[i - half_kw:i - half_kw + kw, j - half_kh:j - half_kh + kh],
                        kernel
                    )
                )

    def get_jacobi(self, parent):
        data = self.parents[0].value
        kernel = self.parents[1].value
        w, h = data.shape
        kw, kh = kernel.shape
        half_kw, half_kh = int(kw / 2), int(kh / 2)
        pw, ph = tuple(
            np.add(data.shape,
                   np.multiply((half_kw, half_kh), 2)
                   )
        )
        jacobi = []
        if parent is self.parents[0]:
            for i in np.arange(half_kw, half_kw + w):
                for j in np.arange(half_kh, half_kh + w):
                    mask = np.mat(np.zeros(pw, ph))
                    mask[i - half_kw:i - half_kw + kw, j - half_kh:j - half_kh + kh] = kernel
                    jacobi.append(mask[half_kw:half_kw + w, half_kh:half_kh + h].A1)
        elif parent is self.parents[1]:
            for i in np.arange(half_kw, half_kw + w):
                for j in np.arange(half_kh, half_kh + w):
                    jacobi.append(
                        self.padded[i - half_kw:i - half_kw + kw, j - half_kh:j - half_kh + kh].A1
                    )
        return np.mat(jacobi)


class ScalarMultiply(Operator):
    """
     标量 × 矩阵
    """
    def compute(self):
        self.value = np.multiply(self.parents[0].value, self.parents[1].value)

    def get_jacobi(self, parent):
        if parent is self.parents[0]:
            return self.parents[1].value.flatten().T
        else:
            return np.mat(
                np.eye(self.parents[1].dimension())
            ) * self.parents[0].value[0, 0]


class MaxPooling(Operator):
    def __init__(self, *parent, **kargs):
        Operator.__init__(self, *parent, **kargs)
        self.stride = kargs.get("stride")
        self.size = kargs.get("size")
        self.flag = None

    def compute(self):
        data = self.parents[0].value
        w, h = data.shape
        dim = w * h
        sw, sh = self.stride
        kw, kh = self.size
        hkw, hkh = int(kw/2), int(kh/2)
        res = []
        flag = []

        for i in np.arange(0, w, sw):
            row = []
            for j in np.arange(0, h, sh):
                top, btm = max(0, i - hkw), min(w, i + hkw + 1)
                left, right = max(0, j - hkh), min(h, j + hkh + 1)
                window = data[top:btm, left:right]
                row.append(np.max(window))

                pos = np.argmax(window)
                w_width = right - left
                offset_w, offset_h = top + pos // w_width, left + pos % w_width
                offset = offset_w * w + offset_h
                tmp = np.zeros(dim)
                tmp[offset] = 1
                flag.append(tmp)
            res.append(row)
        self.flag = np.mat(flag)
        self.value = np.mat(row)

    def get_jacobi(self, parent):
        return self.flag
