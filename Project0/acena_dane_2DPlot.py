#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 09:36:23 2019

@author: daneography

Project 0: Sepal Length v Petal Length

Description: Single plot that shows sepal length versus petal length for all 
three varieties of iris flowers.

This python reads in and opens file named IrisData.txt and write out the plot 
to a png file named acena_dane_MyPlot.png

Axes are labeled "Petal Length" and "Sepal Length"
"""
#IMPORT BLOCK
import matplotlib.pyplot as plt
import pandas as pd


#INPUT BLOCK
#Input: Read text file named "IrisData.txt"
IrisData = pd.read_csv("IrisData.txt", delimiter='\t', 
                       names=['Sepal Length', 
                              'Sepal Width',
                              'Petal Length', 
                              'Petal Width', 
                              'Type'])

#PREPARE THE DATA
setosa = IrisData.loc[IrisData.Type=='setosa']
virginica = IrisData.loc[IrisData.Type=='virginica']
versicolor = IrisData.loc[IrisData.Type=='versicolor']

#CREATING PLOT
fig = plt.figure()
ex = fig.add_subplot()

#PLOTTING DATA
#Input: Takes inputs from the IrisData.txt, uses column0 for sepal lenght,
#column2 for petal length, and and column3 for the iris type.
#Output: 2D plot of petal length vs sepal length for each iris type.
setosa = ex.scatter(setosa['Sepal Length'],setosa['Petal Length'], 
                    marker="^",color="Red")
virginica = ex.scatter(virginica['Sepal Length'],virginica['Petal Length'],
                       marker=".",color="Blue")
versicolor = ex.scatter(versicolor['Sepal Length'],versicolor['Petal Length'], 
                        marker="*",color="Green")

#SETTING TITLE, AXES LABELS, AND LEGENDS
ex.set(title="Sepal Length v Petal Length", 
       xlabel="Sepal Length", 
       ylabel="Petal Length")
ex.legend([setosa,versicolor,virginica], ["Setosa", "Versicolor", "Viriginica"],
          loc='best')


#SAVING PLOT TO PNG
#Output: Write the plot into a PNG file named "acena_dane_MyPlot.png"
plt.savefig("acena_dane_MyPlot.png")
plt.show()
