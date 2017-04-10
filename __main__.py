import math
import numpy as np
import os as os
from neuron import Neuron
from functions import Functions
from random import randint
from graphics import *

"""
author: Abubakar Saad
"""

def main():

    learningRate = 4.0
    initRadius = 7.5
    epoch = 55
    filename = 'L30fft_64.out'

    # Creating the graphic
    win = GraphWin('SOM', 690, 690)
    win.setCoords(0.0, 0.0, 15.0, 15.0)
    win.setBackground("black")

    # opening a file
    # os.path.dirname gets the dir path for this file
    filepath = "c:\\Users\\Abu\\Documents\\ANN\\Assignment2\\"
    first_line = []

    # Get the first line in python
    with open(filename) as f:
        first_line = f.readline().split(' ')
    
    # cols
    cols = int(first_line[1]) + 1
    
    # To pick random samples from data set
    dataNum = np.arange(0, 53)
    np.random.shuffle(dataNum)
    print(dataNum, len(dataNum))

    # holds all the samples from 1 file
    data = np.loadtxt(filename, delimiter=' ', dtype='float', skiprows=1, usecols=range(0, cols))

    indexDel = np.array([])
    testDataArr = np.array([])
    
    # Hold out method of selecting random items and separating them
    for dn in range(0, 8):
        indexDel = np.append(indexDel, dataNum[dn])
        ts = int(indexDel[dn])
        testDataArr = np.append(testDataArr, data[ts])
        dataNum = np.delete(dataNum, dn)

    # holds 15% of the data for testing
    testDataArr = np.split(testDataArr, 8)
    print(len(testDataArr))

    # find the min and max to generate the weights of nodes
    l3File_max = np.amax(data)
    l3File_min = np.amin(data)

    # create an array for topolgy
    topolgy = np.empty([15, 15], dtype=object)
    for i in range(len(topolgy)):
        for j in range(len(topolgy)):
            topolgy[i][j] = Neuron(l3File_max, l3File_min, int(first_line[1]))


    accuracy = 0
    # learning rate to be updated
    updateLR = learningRate
    lamda = epoch / initRadius
    
    
    for e in range(epoch):
        # rand = randint(0, 52)
        rand = randint(0, 44)
        print('------------------epoch:---------------', e)
        # store the distances of BMU
        distancesBMU = np.empty([15, 15], dtype=float)
        
        # from dataNum === random samples
        rs = dataNum[rand]
        # print(rs)
        new_data = np.delete(data[rs], 0)

        # distance 
        # create an array that will hold all the distances from a vector the neuron, min === BMU
        for i in range(len(distancesBMU)):
            for j in range(len(distancesBMU)):
                distancesBMU[i][j] = topolgy[i][j].EucDistance(new_data, topolgy[i][j].initialWeights())

        # BMU 
        BMU = np.amin(distancesBMU)
        coordBMU = np.argwhere( distancesBMU == BMU )
        coordBMU = coordBMU.flatten()
        # BMU coordss
        BMUx = coordBMU[0]
        BMUy = coordBMU[1]

        radius = Functions().neighbourhood(initRadius, e, lamda)
        print('Radius: ', radius)
        print('Learning Rate: ', updateLR)

        # double for loops to update all the map nodes weights
        for i in range(len(topolgy)):
            for j in range(len(topolgy)):
                # dist from BMU to current Node
                dist = np.sqrt((BMUx - i)**2 + (BMUy - j)**2)
                guass = Functions().guassin(radius, dist)
                mexicanHat = Functions().mexicanhat(radius, dist)
                # update the weight
                update_weight = topolgy[i][j].initialWeights() + updateLR * mexicanHat * (new_data - topolgy[i][j].initialWeights()) 
                topolgy[i][j].updateWeights(update_weight)
                topolgy[BMUx][BMUy].updateClassify(data[rs][0])
                
                # update the neighbours with the same classifier as the BMU
                if (dist < radius): 
                    topolgy[i][j].updateClassify(data[rs][0])
        
        updateLR = Functions().updateLR(learningRate, e, lamda)
        # test during the training also, if the optimial configuration is found, break out the loop
        if (accuracy/len(data) > 0.89):
            break
        else:
            accuracy = testing(data, topolgy)
            print('accuracy: ', accuracy/len(data), accuracy)
    
    ####################################################### TESTING ####################################################
    accuracyT = testing(testDataArr, topolgy)
    print('accuracy Test: ', accuracyT/len(testDataArr))
    for i in range(len(topolgy)):
        for j in range(len(topolgy)):
            if (topolgy[i][j].getClassify() == None):
                print(topolgy[i][j].getClassify(), end=" ")
            else:
                print(int(topolgy[i][j].getClassify()), end=" ")
        print()

    print("------------------------------------- Testing ---------------------------------")
    # draws the topolgy on the GUI
    for i in range(len(topolgy)):
        for j in range(len(topolgy)):
            # 1 is bad motor
            if (topolgy[i][j].getClassify() == 1):
                square = Rectangle(Point(i,j), Point(i+1,j+1))
                square.draw(win)
                square.setFill("yellow")
                
            # 0 is good motor
            elif(topolgy[i][j].getClassify() == 0):
                square = Rectangle(Point(i,j), Point(i+1,j+1))
                square.draw(win)
                square.setFill("green")
            
            elif(topolgy[i][j].getClassify() == None):
                square = Rectangle(Point(i,j), Point(i+1,j+1))
                square.draw(win)
                square.setFill("purple")
    
    win.getMouse()
    win.close()


# testing method 
def testing(data, topolgy):
    BMUarray = np.array([])
    accuracy = 0
    ones = 0
    zeros = 0
    for d in range(len(data)):
        
        distancesBMUT = np.empty([15, 15], dtype=float)
        
        new_dataT = np.delete(data[d], 0)

        for i in range(len(distancesBMUT)):
            for j in range(len(distancesBMUT)):
                distancesBMUT[i][j] = topolgy[i][j].EucDistance(new_dataT, topolgy[i][j].initialWeights())
         
        BMUT = np.amin(distancesBMUT)
        coordBMUT = np.argwhere( distancesBMUT == BMUT )
        coordBMUT = coordBMUT.flatten()
        BMUxT = coordBMUT[0]
        BMUyT = coordBMUT[1]

        idenfiyer = topolgy[BMUxT][BMUyT].getClassify()

        if (idenfiyer == None):
            idenfiyer = str(idenfiyer)
        else: 
            idenfiyer = str(int(topolgy[BMUxT][BMUyT].getClassify()))
        
        codBMU = str(BMUxT) +',' + str(BMUyT) + ',' + idenfiyer
        # print('codBMU, ', codBMU)
        if (len(BMUarray) == 0): 
            BMUarray = np.append(BMUarray, codBMU)
            if(data[d][0] == topolgy[BMUxT][BMUyT].getClassify()):
                    accuracy += 1
        else:
            # increase the zeros or ones or the accuracy if the data vector has the same classifier as the activated BMU
            if(data[d][0] == topolgy[BMUxT][BMUyT].getClassify()):
                if(codBMU in BMUarray):
                    accuracy += 1
                    if (data[d][0] == 0):
                        zeros += 1
                    elif (data[d][0] == 1):
                        ones += 1
                else:
                    if (data[d][0] == 0):
                        zeros += 1
                    elif (data[d][0] == 1):
                        ones += 1
                    BMUarray = np.append(BMUarray, codBMU)
                    accuracy += 1

    BMUarray = np.array([])
    print('Good Motors: ', zeros)
    print('Bad Motors', ones)
    return accuracy

if __name__ == "__main__":
    main()