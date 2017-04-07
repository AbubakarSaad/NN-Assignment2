import math
import numpy as np
import os as os
from neuron import Neuron
from functions import Functions
from random import randint

def main():

    learningRate = 0.9
    decayRate = 1
    initRadius = 7.5
    epoch = 3

    # opening a file
    # os.path.dirname gets the dir path for this file
    filepath = "c:\\Users\\Abu\\Documents\\ANN\\Assignment2\\"
    first_line = []
    # Get the first line in python
    with open('L30fft16.out') as f:
        first_line = f.readline().split(' ')
    
    
    cols = int(first_line[1]) + 1
    # print(cols)
    
    data = np.loadtxt('L30fft16.out', delimiter=' ', dtype='float', skiprows=1, usecols=range(0, cols))
    # print(data[0])

    l3File_max = np.amax(data)
    l3File_min = np.amin(data)

    # create an array for topolgy
    topolgy = np.empty([15, 15], dtype=object)
    for i in range(len(topolgy)):
        for j in range(len(topolgy)):
            topolgy[i][j] = Neuron(l3File_max, l3File_min, int(first_line[1]))

    # print(topolgy)
    # print(topolgy[0][0].EucDistance(data[0], topolgy[0][0].initialWeights()))

    
    # print(l3File_max, l3File_min)
    for e in range(epoch):
        rand = randint(1, 52)
        print('------------------epoch:---------------', e)
        distancesBMU = np.empty([15, 15], dtype=float)
    
        new_data = np.delete(data[rand], 0)
        # print(new_data)
        # distance 
        # create an array that will hold all the distances from a vector the neuron, min === BMU
        for i in range(len(distancesBMU)):
            for j in range(len(distancesBMU)):
                distancesBMU[i][j] = topolgy[i][j].EucDistance(new_data, topolgy[i][j].initialWeights())

        # print(distancesBMU)
        BMU = np.amin(distancesBMU)
        # print('BMU: ', BMU)
        coordBMU = np.where( distancesBMU == BMU )
        BMUx = coordBMU[0]
        BMUy = coordBMU[1]
        # print(coordBMU)

        print('coords', BMUx[0], BMUy[0])
        # print(topolgy[BMUx[0]][BMUy[0]].initialWeights())
        
        lamda = epoch / initRadius
        #print('Lambda', lamda)
        
        updateLR = learningRate
        radius = Functions().neighbourhood(initRadius, e, lamda)
        print('Radius: ', radius)
        # print('guassin: ', Functions().guassin(radius, BMUx[0], BMUy[0], BMUx[0], BMUy[0]))


        # double for loops to update all the map nodes
        for i in range(len(topolgy)):
            for j in range(len(topolgy)):
                guass = Functions().guassin(radius, BMUx[0], BMUy[0], i, j)
                # print(guass)
                update_weight = topolgy[i][j].initialWeights() + updateLR * guass * (new_data - topolgy[i][j].initialWeights()) 
                topolgy[i][j].updateWeights(update_weight)

                # if the neuron is activeated
                # check the data classifier 
                # set that to all the neighbouring as that classifier
                

        if e == 1:
            print('data number: ', rand)
            print('data: ', data[rand])
            print('updated weights?', topolgy[BMUx[0]][BMUy[0]].initialWeights())
        updateLR = Functions().updateLR(learningRate, e, lamda)





if __name__ == "__main__":
    main()