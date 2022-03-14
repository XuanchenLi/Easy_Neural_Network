import numpy as np


class NeuralNetwork:
    def __init__(self, layerNumber, neuronNumberPerLayer, learningRate):
        self.layerNumber = layerNumber
        self.neuronNumberPerLayer = neuronNumberPerLayer
        self.learningRate = learningRate
        self.weight = []

    # initialize weights
    def gaussian_weights(self, loc, scale):
        for i in range(self.layerNumber):
            self.weight.append(np.random.normal(
                loc, scale,
                (self.neuronNumberPerLayer[i+1], self.neuronNumberPerLayer[i])
            ))

    def zero_weights(self):
        for i in range(self.layerNumber):
            self.weight.append(np.zeros(self.neuronNumberPerLayer[i+1]*self.neuronNumberPerLayer[i]))

    # set activation function

    # forward propagation

    # back propagation

    # calculate error


