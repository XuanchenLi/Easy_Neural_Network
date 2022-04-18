from NNF.core.node import *


class Metrics(Node):
    """
    父节点为模型输出节点和标签节点
    """
    def __init__(self, *parents, **kargs):
        kargs['need_save'] = kargs.get('need_save', False)
        Node.__init__(self, *parents, **kargs)

        self.init()

    def reset(self):
        self.clear_value()
        self.init()

    @abc.abstractmethod
    def init(self):
        pass

    def get_jacobi(self, parent):
        """
        不允许调用
        """
        raise NotImplementedError()

    @staticmethod
    def prob_to_label(prob, threshold=0.5):
        if prob.shape[0] > 1:
            labels = np.argmax(prob, axis=0)
        else:
            labels = np.where(prob < threshold, 0, 1)
        return labels

    def value_str(self):
        return "{}: {:.4f} ".format(self.name, self.value)


class Accuracy(Metrics):
    def __init__(self, *parents, **kargs):
        Metrics.__init__(*parents, **kargs)

    def init(self):
        self.correct_num = 0
        self.tot_num = 0

    def compute(self):
        pred = Metrics.prob_to_label(self.parents[0].value)
        gt = self.parents[1].value
        self.correct_num += np.sum(pred == gt)
        self.tot_num += len(pred)
        self.value = 0
        if self.tot_num != 0:
            self.value = float(self.correct_num) / self.tot_num


class Precision(Metrics):
    """
    查准率 TP/(TP + FP)
    """
    def __init__(self, *parents, **kargs):
        Metrics.__init__(*parents, **kargs)

    def init(self):
        self.true_p = 0
        self.pred_p = 0

    def compute(self):
        pred = Metrics.prob_to_label(self.parents[0].value)
        gt = self.parents[1].value
        self.pred_p += np.sum(pred == 1)
        self.true_p += np.sum(gt == pred and pred == 1)
        self.value = 0
        if self.pred_p != 0:
            self.value = float(self.true_p) / self.pred_p


class Recall(Metrics):
    """
    查全率 TP/(TP + FN)
    """
    def __init__(self, *parents, **kargs):
        Metrics.__init__(*parents, **kargs)

    def init(self):
        self.true_p = 0
        self.gt_p = 0

    def compute(self):
        pred = Metrics.prob_to_label(self.parents[0].value)
        gt = self.parents[1].value
        self.gt_p += np.sum(gt == 1)
        self.true_p += np.sum(gt == pred and pred == 1)
        self.value = 0
        if self.gt_p != 0:
            self.value = float(self.true_p) / self.gt_p


class ROC(Metrics):
    def __init__(self, *parents, **kargs):
        Metrics.__init__(*parents, **kargs)

    def init(self):
        self.count = 100
        self.gt_p = 0
        self.gt_n = 0
        self.true_p = np.array([0] * self.count)
        self.false_p = np.array([0] * self.count)
        self.tpr = np.array([0] * self.count)
        self.fpr = np.array([0] * self.count)

    def compute(self):
        prob = self.parents[0].value
        gt = self.parents[1].value
        self.gt_p += np.sum(gt == 1)
        self.gt_n += np.sum(gt == -1)
        thresholds = list(np.arange(0.01, 1.00, 0.01))
        for idx in range(0, len(thresholds)):
            pred = Metrics.prob_to_label(prob, thresholds[idx])
            self.true_p[idx] += np.sum(pred == gt and gt == 1)
            self.false_p[idx] += np.sum(pred != gt and gt == 1)
        if self.gt_p != 0 and self.gt_n != 0:
            self.tpr = self.true_p / self.gt_p
            self.fpr = self.false_p / self.gt_n


class ROC_AUC(Metrics):
    def __init__(self, *parents, **kargs):
        Metrics.__init__(*parents, **kargs)

    def init(self):
        self.gt_p = []
        self.gt_n = []

    def compute(self):
        prob = self.parents[0].value
        gt = self.parents[1].value
        if gt[0, 0] == 1:
            self.gt_p.append(prob)
        else:
            self.gt_n.append(prob)
        self.total = len(self.gt_p) * len(self.gt_n)

    def value_str(self):
        count = 0
        for gtp in self.gt_p:
            for gtn in self.gt_n:
                if gtp > gtn:
                    count += 1
        self.value = float(count) / self.total
        return "{}: {:.4f} ".format(self.name, self.value)
