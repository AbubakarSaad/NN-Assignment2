import numpy as np

class Neuron(object):

    def __init__(self, l3max, l3min, sizeofWeightVector):
        self.l3max = l3max
        self.l3min = l3min
        self.sizeofWeightVector = sizeofWeightVector
        self.weights = np.random.uniform(low=self.l3min, high=self.l3max, size=(self.sizeofWeightVector,))
        self.classification = None


    def updateWeights(self, weights):
        self.weights = weights

    def initialWeights(self):
        # np.random.seed(10)
        return self.weights

    
    def EucDistance(self, inputVector, Weights):
        # print(inputVector, Weights)
        return np.sqrt(np.sum((inputVector - Weights)**2))