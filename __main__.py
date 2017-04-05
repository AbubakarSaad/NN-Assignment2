import math
import numpy as np
import os as os
from neuron import Neuron

def main():
    # opening a file
    # os.path.dirname gets the dir path for this file
    filepath = "c:\\Users\\Abu\\Documents\\ANN\\Assignment2\\"
    first_line = []
    # Get the first line in python
    with open('L30fft16.out') as f:
        first_line = f.readline().split(' ')
    
    
    cols = int(first_line[1]) + 1
    # print(cols)
    
    data = np.loadtxt('L30fft16.out', delimiter=' ', dtype='float', skiprows=1, usecols=range(1, cols))
    print(data)

    l3File_max = np.amax(data)
    l3File_min = np.amin(data)
    # print(l3File_max, l3File_min)

    neuron = Neuron(l3File_max, l3File_min, int(first_line[1]))
    print(neuron.initialWeights())

    topology = np.full((15, 15), neuron, dtype=object)
    print(topology[1][1].initialWeights())


if __name__ == "__main__":
    main()