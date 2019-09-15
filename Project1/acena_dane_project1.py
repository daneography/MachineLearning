#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 09:14:27 2019

@author: Dane Acena

Project 1: k-Nearest Neighbor

Problem Description: Clemson fisheries habve recetly discovered two new species
of fish in Lake Hartwell. Species TigerFish1 is a delicious fish that tastes 
Bluefin Tuna. TigerFish0 looks almost identical to TigerFish1 but is sloightly 
poisonous and usually makes the person who consumes it very ill. The Clemson 
Wildlife and Fisheries Biology graduate students have captured, measured and 
tested hundred of each species and created a file that contains measurements of
the body length and dorsal fin length of each fish, along with its species. You
have been hired to create a k-Nearest Neighbot program that, given the body
length and dorsal fin length of a fish, will predict if it is TigerFish1 or 
TigerFish 0.

Assignment: Develop a k-Nearest Neighbor algorithm that will predict the
species of a fish. Use a test set and 5-fold cross validation to determin the
best number of neight to use in your prediction. Use a confusion matrix to help
evaluate the results. A separate write-up is created along with this.

References used:
    https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
    https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import operator
import itertools

"""
 Function to plot the input data. Plots the body length v dorsal fin length
 of either TigerFish1(red) or TigerFish0(blue)
 INPUT: filename from user input
 OUTPUT: 2D plot
 """
def plotData(filename):
    ffData = pd.read_csv(filename,skiprows=[0], delimiter='\t', 
                         names=['Body Length', 
                                'Dorsal Fin Length',  
                                'Type'])
  
    TigerFish1 = ffData.loc[ffData.Type == 1]
    TigerFish0 = ffData.loc[ffData.Type == 0]
    
    #Preparing plot
    fig = plt.figure()
    fish = fig.add_subplot()

    fish1 = fish.scatter(TigerFish1['Body Length'],TigerFish1['Dorsal Fin Length'], 
                        marker="^",color="Red")
    fish0 = fish.scatter(TigerFish0['Body Length'],TigerFish0['Dorsal Fin Length'],
                           marker=".",color="Blue")

    
    #SETTING TITLE, AXES LABELS, AND LEGENDS
    fish.set(title="Body Length v Dorsal Fin Length", 
           xlabel="Body Length", 
           ylabel="Dorsal Fin Length")
    fish.legend([fish1, 
                 fish0], ["TigerFish1", 
                          "TigerFish0"], loc='best')
    plt.show()
"""
 Loads data from input and spliting it to two: Training Set(240) and Test 
 Set(60) Save the Training Set and Test Set as txt files. Training set is 
 divided into 5 sets. From that 5 sets, it will recursively make new Training 
 files from four of the sets and create a validation file from the remain set.
 it will keep creating new files until all combinations of the training sets
 are made
 INPUT: filename from user input, split = a float that denotes the % of split
 OUTPUT: trainingSet and testSet dataframes and saved into text files
"""
def loadDataSet(filename, split):
    # uses pandas library to read a tab-delimited text file from user input
    # headers are removed
    ffData = pd.read_csv(filename, delimiter='\t',header=None)
    ffDataSet = ffData.loc[1:]
    ffDataSet = ffDataSet.sample(frac=1).reset_index(drop=True)
    
    # ffData is split depending on input an 80 on split gives 80-20. 
    # trainingSet gets 80% and testSet get 20%
    trainingSet, testSet, = np.split(ffDataSet, [int(split*len(ffDataSet))])
    
    # saves trainingSet and testSet into two text files
    np.savetxt("TrainingSet.txt", trainingSet, fmt='%g', delimiter='\t')
    np.savetxt("TestSet.txt", testSet, fmt='%g', delimiter='\t')
    
    # Intiates set at 5. splits the trainingSet into 5 with equal number of 
    # elements
    sets = 5
    TrainingData_split = np.array_split(trainingSet, sets)
    # for loop that iterates from 5 decrement, creates a file reverse order
    # named. TrainingData_split[1] data goes to Val 5, 2 to 4, and so on.
    for i in reversed(range(sets)):
        np.savetxt('Val' + str(sets-i) + '.txt',TrainingData_split[i], 
                   fmt='%g', delimiter='\t')
    
    # creates list of iteration of 1,2,3,4,5 in 4 elements
    # {(1,2,3,4),(1,2,3,4),(1,2,4,5),(1,3,4,5),(2,3,4,5)}
    trainSets = itertools.combinations([1,2,3,4,5],sets-1)
    trainSet_ = list(trainSets)
    
    # goes through the list created above and appends corresponding split for
    # each set and recursively saves it into a text file.
    for x in range(len(trainSet_)):
        appended_data = []
        for y in range(len(trainSet_[x])):
            n = trainSet_[x][y]
            appended_data.append(TrainingData_split[n-1])
        appended_data = pd.concat(appended_data)
        np.savetxt("Train" + str(x+1) + ".txt", appended_data, fmt='%g', 
                   delimiter='\t')
    
    return(trainingSet, testSet)
    
# Uses pythagorean theorem to determine the distance between two points
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((float(instance1[x]) - float(instance2[x])),2)
    return (math.sqrt(distance))

"""
 gets the k closests points in between testInstance and trainingSet
 INPUT: trainingSet: dataframe, testInstance: specific row in the test df
 OUTPUT: dataframe that contains k elements
"""
def getNeighbors(trainingSet, testInstance,k):
    distances = []
    length    = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet.loc[x],length)
        distances.append((trainingSet.loc[x],dist))
    distances.sort(key=operator.itemgetter(1))
    
    neighbors = []
    for x in range (k):
        neighbors.append(distances[x][0])
    neighbors_df = pd.DataFrame(neighbors)
    return neighbors_df

"""
 tallies the type of the k neighbors and returns the type that has the most
 votes
 INPUT: types column of the neighbors dataframe
 OUTPUT: resulting type depending on the 'winner' of the votes
"""
def getResponse(neighbors_typeList):
    classVotes = {}
    for x in range(len(neighbors_typeList)):
        response = neighbors_typeList[x]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse = True)
    return sortedVotes[0][0]

"""
 counts accuracy % by counting correct predicions and dividing it with the 
 length of the testSet and multiplying by 100
"""
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet.iloc[x,-1] == predictions[x]:
            correct += 1
    return(correct/float(len(testSet))) * 100

def getError(testSet, predictions):
    incorrect = 0
    for x in range(len(testSet)):
        if testSet.iloc[x,-1] != predictions[x]:
            incorrect += 1
    return incorrect

def crossValidate(trainSet, testSet,k):
    predictions = []
    for x in range(len(testSet)):
        neighbors_df = getNeighbors(trainSet, testSet.loc[x],k)
        neighbors_typeList = neighbors_df.iloc[:,-1].values.tolist()
        result = getResponse(neighbors_typeList)
        predictions.append(result)
    accuracy = getAccuracy(testSet, predictions)
    error = getError(testSet, predictions)
    return accuracy, error, predictions

def errorPlot(errorDF):
    errorDF.loc["TOTAL"] = errorDF.sum()
    np.savetxt("error.txt", errorDF, fmt='%g', encoding='utf-8', delimiter='\t')
    print("""
>> A dataframe containing error for every
iteration of tests has been created
          """)
    y = errorDF.loc["TOTAL"]
    
    fig = plt.figure()
    error = fig.add_subplot()
    
    error.plot(y, marker="o",color="Red", markerfacecolor="Blue",
                       linestyle = "dashed")
    error.locator_params(integer=True)
    plt.xlabel("Value of k for kNN", axes=error)
    plt.ylabel("Error", axes=error)
    plt.title("NUMBER OF MISCLASSIFICATIONS")
    plt.show()

def accuracyPlot(accuracyDF):
    accuracyDF.loc["MEAN"] = accuracyDF.mean()
    np.savetxt("accuracy.txt", accuracyDF, fmt='%g', encoding='utf-8', delimiter='\t')
    print("""
>> A dataframe containing accuracy % for every
iteration of tests has been created
          """)
    y = accuracyDF.loc["MEAN"]
    fig = plt.figure()
    accuracy = fig.add_subplot()
    
    accuracy.plot(y, marker='o', color='Red', markerfacecolor='Blue',
                  linestyle='dashed')
    accuracy.locator_params(integer = True)
    plt.xlabel("Value of K for kNN",axes=accuracy)
    plt.ylabel("Cross-Validated Accuracy",axes=accuracy)
    plt.title("% ACCURACY")
    plt.show()

def validationMode():
        print("This will take a few minutes. Please be patient.")
        print("Testing: ")
        errorDF = pd.DataFrame()
        accuracyDF = pd.DataFrame()
        for x in  range(5):
            for k in range(1,23,2):
                print("Train" + str(x+1) + ".txt and Val" + str(x+1) + ".txt against k="+str(k))
                trainSet = pd.read_csv("Train"+str(x+1)+".txt", delimiter='\t', header=None)
                testSet = pd.read_csv("Val"+str(x+1)+".txt", delimiter='\t', header=None)
                accuracy, error, result = crossValidate(trainSet, testSet, k)
                errorDF.at[x,k] = error
                accuracyDF.at[x,k] = accuracy
    
        errorPlot(errorDF)
        accuracyPlot(accuracyDF)
    
def testMode(trainSet, testSet,k):
        accuracy, error, prediction_ = crossValidate(trainSet, testSet, k)
        print("Accuracy: " + repr(accuracy)+ '%')
        print("Error: " + repr(error))
        prediction_ = pd.DataFrame([prediction_, testSet.iloc[:,-1]])
        np.savetxt("predictionVactual_comparison.txt", prediction_, fmt='%g', 
                   delimiter='\t', encoding='utf-8')
        
        predictions = []
        while True:
            bodyLength = float(input("Input Body Length: "))
            dorsalLength = float(input("Input Dorsal Fin Length: "))
            if bodyLength == 0 and dorsalLength == 0:
                break
            elif bodyLength < 0 or dorsalLength < 0:
                print("Lengths needs to be positive values.")
            else:
                lengthSet = [bodyLength, dorsalLength]
                input_neighbors = getNeighbors(trainSet, lengthSet,k)
                input_neighbors_typeList = input_neighbors.iloc[:,-1].values.tolist()
                result = getResponse(input_neighbors_typeList)
                predictions.append(result)
                print("That fish is predicted to be: TigerFish"+str(int(result)))
                

def main():
    print("""
             +-+ +-+-+-+-+-+-+-+ +-+-+-+-+-+-+-+-+
             |k| |n|e|a|r|e|s|t| |n|e|i|g|h|b|o|r|
             +-+ +-+-+-+-+-+-+-+ +-+-+-+-+-+-+-+-+                     
          """)
    filename = input("What is the filename: ")
    split = 0.80
    loadDataSet(filename,split)
    trainSet = pd.read_csv("TrainingSet.txt", delimiter='\t', header=None)
    testSet = pd.read_csv("TestSet.txt", delimiter='\t', header=None)              
    while True:
        print("""
            ==============================================================
            |  If you want to see the plot for the data set that you     |
            |  selected use 'plot'                                       |
            |                                                            |
            |  If you want to figure out the best k for a data set       |
            |  use 'validate'.                                           |
            |                                                            |
            |  If you are testing this on a specific k for a data set    |
            |  use 'test'.                                               |
            |                                                            |
            |  If you are grading this project and are using a different |
            |  data set use 'grade'.                                     |
            |                                                            |
            |   Use 'info' if you want to learn how to use this.         |
            |                                                            |
            |   If you want to reset and change the dataset file         |
            |   use 'reset'.                                             |
            |                                                            |
            |  If you want to terminate the program use 'end'            |
            ==============================================================
              """)
        mode = str(input("What mode? : "))
        
        if mode == 'validate':
            validationMode()
            continue

        elif mode == 'test':
            k = int(input("""
Based on the results from validate 
mode, what is the best k? """))
            testMode(trainSet, testSet,k)
            continue
        
        elif mode == 'grade':
            k = 7
            testMode(trainSet, testSet,k)
            continue
        
        elif mode == 'plot':
            plotData(filename)
            continue
        
        elif mode == 'info':
            print("""
                  Essentially, you want to start with 'validate'.
                  This divides your data into two files (Training, Test) 80-20
                  
                  Training will then be divided into 5 sets and recursively 
                  create a Train file that combines four of the set and 
                  create a Val file that contains the unused set. This repeats
                  until all combinations (no reuse) have been created.
                  
                  Those smaller sets will then be recursively tested against 
                  each other on different K values. Train1 with Val1, Train2 
                  with Val2, and so on.
                  
                  Two plots that show the errors and accuracy for each 
                  iteration of tests will display. Providing information about
                  the possible best k value to be used for Training v Test in 
                  'test' mode.
                  
                  Test mode will prompt for a k value (best choice evaluated 
                  from 'validate' mode). This will run Test against Training.
                  It will display the accuracy of kNN and the number of errors.
                  
                  Then, it will create a predictionVactual_comparison.txt that 
                  shows the predicted type and the actual type from here, 
                  confusion matrix can then be evaluated
                  
                  If you're grading this, use "grade". This contains a hard 
                  coded k that was evaluated from validate.
                  """)
            continue
       
        elif mode == 'end':
            break
        
        elif mode == 'reset':
            main()
            break
                
        else:
            print ("Invalid Response, please use one of the options below")
            continue

main()
