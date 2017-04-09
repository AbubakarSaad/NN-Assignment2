import math
import numpy as np
import os as os
from neuron import Neuron
from functions import Functions
from random import randint
from graphics import *

def main():

    learningRate = 1.5
    decayRate = 1
    initRadius = 7.5
    epoch = 1000

    win = GraphWin('Floor', 730, 701)

    win.setCoords(0.0, 0.0, 14.0, 14.0)
    win.setBackground("black")

    #draw grid
    for x in range(15):
        for y in range(15):
            win.plotPixel(x*50, y*50, "white")
    
    


    # opening a file
    # os.path.dirname gets the dir path for this file
    filepath = "c:\\Users\\Abu\\Documents\\ANN\\Assignment2\\"
    first_line = []
    # Get the first line in python
    with open('L30fft16.out') as f:
        first_line = f.readline().split(' ')
    
    # cols
    cols = int(first_line[1]) + 1
    # print(cols)
    dataNum = np.arange(0, 53)
    np.random.shuffle(dataNum)
    print(dataNum, len(dataNum))
    data = np.loadtxt('L30fft_64.out', delimiter=' ', dtype='float', skiprows=1, usecols=range(0, cols))

    indexDel = np.array([])
    testDataArr = np.array([])
    # from 0-52 select 8 random elements
    # store those in an array 
    # and delete it from data array 
    for dn in range(0, 8):
        indexDel = np.append(indexDel, dataNum[dn])
        ts = int(indexDel[dn])
        testDataArr = np.append(testDataArr, data[ts])
        dataNum = np.delete(dataNum, dn)

    
    testDataArr = np.split(testDataArr, 8)
    #print(testDataArr)
    print(indexDel)
    print(dataNum, len(dataNum))
    #print(data)

    l3File_max = np.amax(data)
    l3File_min = np.amin(data)

    # create an array for topolgy
    topolgy = np.empty([15, 15], dtype=object)
    for i in range(len(topolgy)):
        for j in range(len(topolgy)):
            topolgy[i][j] = Neuron(l3File_max, l3File_min, int(first_line[1]))

    # print(topolgy)
    # print(topolgy[0][0].EucDistance(data[0], topolgy[0][0].initialWeights()))
    BMUarray = np.array([])
    accuracy = 0
    # print(l3File_max, l3File_min)
    for e in range(epoch):
        rand = randint(0, 44)
        print('------------------epoch:---------------', e)
        distancesBMU = np.empty([15, 15], dtype=float)
        
        # from dataNum === random samples

        rs = dataNum[rand]
        new_data = np.delete(data[rs], 0)

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
        # BMUarray = np.append(BMUarray, BMU)

        # just get the min of the BMU.x and currentNode.x
        # then compare that to currentNode.x +- 15
        # and do the same for y again too

        # print('coords', BMUx[0], BMUy[0])
        # print(topolgy[BMUx[0]][BMUy[0]].initialWeights())
        
        lamda = epoch / initRadius
        #print('Lambda', lamda)
        
        updateLR = learningRate
        radius = Functions().neighbourhood(initRadius, e, lamda)
        print('Radius: ', radius)
        # print('guassin: ', Functions().guassin(radius, BMUx[0], BMUy[0], BMUx[0], BMUy[0]))

        # print(data[rand])
        # double for loops to update all the map nodes
        for i in range(len(topolgy)):
            for j in range(len(topolgy)):
                dist = (BMUx[0] - i)**2 + (BMUy[0] - j)**2
                guass = Functions().guassin(radius, dist)
                # print(guass)
                update_weight = topolgy[i][j].initialWeights() + updateLR * guass * (new_data - topolgy[i][j].initialWeights()) 
                topolgy[i][j].updateWeights(update_weight)
                topolgy[BMUx[0]][BMUy[0]].updateClassify(data[rs][0])
                # if the neuron (BMU) is activeated
                # check the data classifier 
                # set that to all the neighbouring as that classifier
                if (dist <= radius): 
                    topolgy[i][j].updateClassify(data[rs][0])
                # after the training test 
                # finding the BMU and check the classifier is the same as the data classifier, Make sure there isnt same BMU
                # if it is increase accuracy
        
        updateLR = Functions().updateLR(learningRate, e, lamda)
        if (accuracy/len(data) > 0.86):
            break
        else:
            accuracy = testing(data, topolgy, BMUarray)
            print('accuracy: ', accuracy/len(data))
            BMUarray = np.array([])
        # if e == epoch - 1:
        #     print('data number: ', rand)
        #     print('data: ', data[rand])
        #     print('updated weights?', topolgy[BMUx[0]][BMUy[0]].initialWeights())
        #     print('BMUarray', BMUarray)
        #     for i in range(len(topolgy)):
        #         for j in range(len(topolgy)):
        #             print(topolgy[i][j].getClassify(), end=" ")
        #         print()
            # print('accuracy: ', accuracy)
    
    ####################################################### TESTING ####################################################
    accuracyT = testing(testDataArr, topolgy, BMUarray)
    print('accuracy Test: ', accuracyT/len(testDataArr))
    for i in range(len(topolgy)):
        for j in range(len(topolgy)):
            print(int(topolgy[i][j].getClassify()), end=" ")
        print()

    print("------------------------------------- Testing ---------------------------------")
    for i in range(len(topolgy)-1, 0, -1):
        for j in range(len(topolgy)):
            if (topolgy[i][j].getClassify() == 1):
                square = Rectangle(Point(i,j), Point(i+1,j+1))
                square.draw(win)
                square.setFill("yellow")
            print(int(topolgy[i][j].getClassify()), end=" ")
        print()

    

    win.getMouse()
    win.close()

    
    

def testing(data, topolgy, BMUarray):
    accuracy = 0
    for d in range(len(data)):
        # print('------------------testing:---------------', d)
        # print('data: ', data[d][0])
        distancesBMUT = np.empty([15, 15], dtype=float)
        
        new_dataT = np.delete(data[d], 0)

        for i in range(len(distancesBMUT)):
            for j in range(len(distancesBMUT)):
                distancesBMUT[i][j] = topolgy[i][j].EucDistance(new_dataT, topolgy[i][j].initialWeights())
         
        BMUT = np.amin(distancesBMUT)
        coordBMUT = np.where( distancesBMUT == BMUT )
        BMUxT = coordBMUT[0]
        BMUyT = coordBMUT[1]
        #print('testing bmu', BMU)
        codBMU = str(BMUxT[0]) +',' + str(BMUyT[0])
        # print(codBMU)
        if (len(BMUarray) == 0): 
            BMUarray = np.append(BMUarray, codBMU)
        else:
            same = np.where(BMUarray == codBMU)
            if(len(same[0]) == 0):
                BMUarray = np.append(BMUarray, codBMU)
                if(data[d][0] == topolgy[BMUxT[0]][BMUyT[0]].getClassify()):
                    accuracy += 1
        # if d == len(data) - 1:
            # print('data: ', data[d])
            # print('updated weights?', topolgy[BMUx[0]][BMUy[0]].initialWeights())
            # print('BMUarray', BMUarray)
            # for i in range(len(topolgy)):
            #     for j in range(len(topolgy)):
            #         print(topolgy[i][j].getClassify(), end=" ")
            #     print()
    return accuracy

if __name__ == "__main__":
    main()