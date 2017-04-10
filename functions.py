import math
import numpy as np

class Functions():

    def neighbourhood(self, radius, numIteration, timeConstant):
        return radius * np.exp(-(numIteration / timeConstant))

    def guassin(self, radius, dist):
        return np.exp(-(dist**2)/(2*(radius**2)))
    
    def updateLR(self, learningRate, numIteration, timeConstant):
        return learningRate * np.exp(-(numIteration / timeConstant))

    def mexicanhat(self, radius, dist):
        return (1-(dist**2/radius**2)) * np.exp(-(dist**2)/(2*(radius**2)))