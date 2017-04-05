import numpy as np

class Neuron(object):

    def __init__(self, l3max, l3min, sizeofWeightVector):
        self.l3max = l3max
        self.l3min = l3min
        self.sizeofWeightVector = sizeofWeightVector
        self.weights = np.random.uniform(low=self.l3min, high=self.l3max, size=(self.sizeofWeightVector,))

    def initialWeights(self):
        return self.weights

    
    def EucDistance():
        pass