import math
import numpy as np

class Functions():

    def neighbourhood(self, radius, numIteration, timeConstant):
        return radius * np.exp(-(numIteration / timeConstant))

    def guassin(self, radius, dist):
        # dist is the parathe therom
        # print(bmux, bmuy, ni, nj)
        return np.exp(-((dist)/(2*(radius**2))))
    
    def updateLR(self, learningRate, numIteration, timeConstant):
        return learningRate * np.exp(-(numIteration / timeConstant))