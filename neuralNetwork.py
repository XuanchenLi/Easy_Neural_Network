import numpy as np


class NeuralNetwork:
    def __init__(self, layerNumber, neuronNumberPerLayer, learningRate, epochs):
        self.outputs = []
        self.errors = []
        self.layerNumber = layerNumber
        self.neuronNumberPerLayer = neuronNumberPerLayer
        self.learningRate = learningRate
        self.weight = []
        self.activation_function = None
        self.epochs = epochs

    # initialize weights
    def gaussian_weights(self):
        for i in range(self.layerNumber):
            self.weight.append(np.random.normal(
                0.0, pow(self.neuronNumberPerLayer[i+1], -0.5),
                (self.neuronNumberPerLayer[i+1], self.neuronNumberPerLayer[i])
            ))

    def zero_weights(self):
        for i in range(self.layerNumber):
            self.weight.append(np.zeros(self.neuronNumberPerLayer[i+1] * self.neuronNumberPerLayer[i]))

    # set activation function
    def set_activate(self, function):
        if function == "sigmoid":
            self.activation_function = lambda x: 1.0/(1.0+np.exp(-x))
        else:
            raise ValueError("not implemented")

    # forward propagation
    def forward(self, inputs, labels):
        inputs_t = np.array(inputs, ndmin=2).T
        labels_t = np.array(labels, ndmin=2).T
        self.outputs = []
        self.outputs.append(inputs_t)
        self.errors = []
        for i in range(self.layerNumber):
            tmp_inputs = np.dot(self.weight[i], inputs_t)
            tmp_outputs = self.activation_function(tmp_inputs)
            inputs_t = tmp_outputs
            self.outputs.append(tmp_outputs)
        for i in range(self.layerNumber):
            if i == 0:
                self.errors.append(labels_t - self.outputs[-1])
            else:  # 计算右层残差加权求和
                self.errors.append(
                    np.dot((self.weight[self.layerNumber-i]).T, self.errors[i-1])
                )

    # back propagation
    def back_propagation(self):
        for i in range(self.layerNumber):
            # 更新权重
            self.weight[self.layerNumber-i-1] += self.learningRate * \
                                                 np.dot(
                                                     (self.errors[i] *
                                                      self.outputs[-1-i] * (1.0 - self.outputs[-1-i])),
                                                     np.transpose(self.outputs[-1-i-1])
                                                 )

    # fit
    def fit(self, inputs, targets):
        for i in range(self.epochs):
            for k in range(len(inputs)):
                self.forward(inputs[k], targets[k])
                self.back_propagation()

    # predict
    def predict(self, inputs):
        inputs_t = np.array(inputs, ndmin=2).T
        for i in range(self.layerNumber):
            tmp_inputs = np.dot(self.weight[i], inputs_t)
            tmp_outputs = self.activation_function(tmp_inputs)
            inputs_t = tmp_outputs
        return inputs_t

    # test
    def test(self, valid_x, valid_y):

        cnt = 0
        for i in range(len(valid_x)):
            pre = self.predict(valid_x[i])
            print(list(pre).index(max(list(pre))), valid_y[i])
            if list(pre).index(max(list(pre))) == list(valid_y[i]).index(1):
                cnt += 1
        return cnt/len(valid_x)



    # set loss function

    # calculate error

