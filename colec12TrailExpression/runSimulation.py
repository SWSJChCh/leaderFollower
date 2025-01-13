'''
runSimulation.py - Samuel Johnson - 27/05/24
'''

import math
import shutil
import time
import os
import cv2
import copy
import sys
import random
import scipy
import datetime
#import imageio.v2 as imageio
import matplotlib.cm as cm
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import signal
from numba import jit, prange
from VEGF import *
from insertCell import *
from moveCell import *
from collisionCell import *
from growthFunction import *

#Colec12 filopodium retraction variable
filPersist = 10

#Animation Boolean
animate = False
#Timestep for animation
animStep = 10

#Create output directory
if animate:
    date = datetime.datetime.now().strftime('%H-%M-%S')
    os.makedirs('LeaderFollower' + date)

#Scale factor for mesh spacing (for computational speed-up)
meshScale = 1
#Simulation runtime
finTime = 24
#First cell insertion time
firstInsert = 6
#Timestep (h)
dt = 1 / 60
#List of domain lengths for each timestep (μm)
lengthList = domainLengths(int(finTime + 1))
#Domain length (μm)
Len = int(lengthList[0])
#Length of PDE mesh (for solver on unit-length mesh)
meshLen = int(lengthList[-1] / meshScale)
#Domain width (μm)
Wit = 3 * 120
#Width of PDE mesh (for solver on unit-length mesh)
meshWit = int(Wit / meshScale)
#Boundary condition smoothing parameter
bcParam = meshScale
#Number of leader cells
leadNum = 5
#Proportion of domain for which DAN is expressed
DANLen = int(float(sys.argv[2]) * lengthList[-1] / meshScale)
#Proportion of domain for which Colec12Trail is expressed
Colec12TrailLen = int(float(sys.argv[1]) * lengthList[-1] / meshScale)
#Proportion of domain for which zero-flux boundary conditions are expressed
spanLen = 0
#Timesteps per attempted cell insertion
insertStep = 1
#Repeats for data averaging
repeats = 25

#VEGF parameters
D = 0.1                        #Diffusion constant
subStep = 1                    #Solver steps per timestep
dx = 1                         #Spacestep (x / μm)
dy = 1                         #Spacestep (y / μm)
c0 = 1.0                       #Initial concentration of reactant
xi = 0.1                       #Sensing parameter

#Cell parameters
cellRad = 7.5                          #Cell radius (μm)
searchRad = 5 * cellRad                #Box size for internalisation (μm)
lenFilo = 3.5 * cellRad                #Filopodium length (μm)
lenFiloMax = 6 * cellRad               #Maximum detection length (μm)
leadSpeed = 1                          #Speed of leaders (μm / minute)
folSpeed = 1.3 * leadSpeed             #Speed of followers (μm / minute)
filoNum = 3                            #Number of filopodia extended by cells
lmda = 2.75 * 10**2                    #Internalisation parameter
chi = 10**-4                           #Logistic production parameter
epsilon = 2 * cellRad                  #Distance for phenotype switch

#Leader spreading data list
conf1LList = []
conf2LList = []
conf3LList = []

#Follower spreading data list
conf1FList = []
conf2FList = []
conf3FList = []

#List of cell x coordinates
cellXList = []

#Stream membership data list
streamFList = []

#Invasion distance data list
distLList = []

#Inter-cellular distance data list
meanSepList = []

#Cell number list
cellNumList = []

#Repeat simulations
for _ in range(repeats):
    #Initialise time
    t = 0
    #Initialise VEGF Mesh (for solver)
    VEGFMesh = createVEGFArray(meshWit//3, meshLen, c0, bcParam)
    #Initialise DAN Mesh (for solver)
    DANMesh = createDANArray(meshWit, meshLen, c0, bcParam, DANLen)
    #Pad VEGF array with empty space on both sides
    padArray = np.zeros((meshWit//3, meshLen))
    halfPadded = np.vstack((padArray, VEGFMesh))
    VEGFPadded = np.vstack((halfPadded, padArray))
    VEGFMesh = VEGFPadded
    #Pad Colec12Trail array
    centerArray = np.zeros((meshWit//3, meshLen))
    halfPadded = np.vstack((centerArray, createDANArray(meshWit//3, meshLen, \
                           c0, bcParam, Colec12TrailLen)))
    Colec12TrailMesh = np.vstack((createDANArray(meshWit//3, meshLen, \
                           c0, bcParam, Colec12TrailLen), halfPadded))
    #Initialise VEGF Array (for cells)
    VEGFArray = cv2.resize(VEGFMesh, (Len, Wit), \
                           interpolation = cv2.INTER_AREA)
    #Initialise DAN Array (for cells)
    DANArray = cv2.resize(DANMesh, (Len, Wit), \
                           interpolation = cv2.INTER_AREA)
    #Initialise DAN Array (for cells)
    Colec12TrailArray = cv2.resize(Colec12TrailMesh, (Len, Wit), \
                           interpolation = cv2.INTER_AREA)
    #List to store cell objects
    cellList = []
    #Data lists for visualisation
    cellMast = []
    expMast = []
    VEGFMast = []
    DANMast = []
    Colec12TrailMast = []
    filMast = []
    #Images for movie writer
    ims = []
    #Plot objects
    fig, ax = plt.subplots(2)
    #Counting variable
    counter = 0
    #Boolean for leader insertion
    leaderInsert = False
    #Run main simulation loop
    while (t < finTime):
        #Increase counting variable
        counter += 1
        #Update time
        t += dt
        #Actual domain length (μm)
        Len = int(lengthList[counter])
        #Update zero-flux expression length
        spanLen = int(float(sys.argv[1]) * lengthList[counter])
        #Time derivative of domain length
        lenDot = (lengthList[counter] - lengthList[counter - 1]) / dt
        #Initial cells are leaders
        if not leaderInsert:
            #Create initial leader cells (evenly distributed line at LHS)
            initConfiguration(cellList, leadNum, Wit, cellRad, lenFilo)
            leaderInsert = True
        if t > firstInsert:
            #List to track cell data at time t
            cellCopyList = []
            #Insert follower cells at constant time intervals
            if counter % insertStep == 0:
                #Insert follower cell
                cell = followerCell(cellRad, lenFilo)
                insertCell(cell, cellList, Wit, Len)
            #Update chemicals according to PDE
            for _ in range(subStep):
                #Update chemoattractant
                VEGFMesh = updateVEGF(VEGFMesh, D, chi, lmda, cellRad, \
                                      cellList, dt, subStep, searchRad, \
                                      Len, Wit, lenDot, meshScale)
                #Update DAN
                DANMesh = updateDAN(DANMesh, D, chi, lmda, cellRad, \
                                      cellList, dt, subStep, searchRad, \
                                      Len, Wit, lenDot, meshScale)
                #Update Colec12Trail (Not degraded)
                Colec12TrailMesh = updateColec12Trail(Colec12TrailMesh, D, chi, 0, cellRad, \
                                      cellList, dt, subStep, searchRad, \
                                      Len, Wit, lenDot, meshScale)
            #Update VEGF Array
            VEGFArray = cv2.resize(VEGFMesh, (Len, Wit), \
                                   interpolation = cv2.INTER_AREA)
            #Update DAN Array
            DANArray = cv2.resize(DANMesh, (Len, Wit), \
                                   interpolation = cv2.INTER_AREA)
            #Update Colec12Trail Array
            Colec12TrailArray = cv2.resize(Colec12TrailMesh, (Len, Wit), \
                                   interpolation = cv2.INTER_AREA)

            #Move cells
            filList = moveCells(VEGFArray, DANArray, Colec12TrailArray, cellList, \
                              filoNum, lenFilo, lenFiloMax, xi, c0, cellRad, \
                              dx, dy, leadSpeed, folSpeed, spanLen, \
                              lengthList[counter - 1], lengthList[counter], \
                              epsilon, filPersist)
            #List of cell position and VEGF Array
            if counter % animStep == 0 and animate:
                for i in cellList:
                    cellCopyList.append(copy.deepcopy(i))
                cellMast.append(cellCopyList)
                VEGFMast.append(VEGFArray.copy())
                Colec12TrailMast.append(Colec12TrailArray.copy())
                DANMast.append(DANArray.copy())
                filMast.append(filList.copy())
                expMast.append(spanLen)

    #Initialise spreading variables
    conf1L = 0
    conf2L = 0
    conf3L = 0
    conf1F = 0
    conf2F = 0
    conf3F = 0
    #Initialise breakage variables
    totFol = 0
    breakFol = 0
    #Initialise invasion distance variable
    maxInvade = 0
    #Initialise cell number variable
    cellNum = 0

    #Update spreading variables
    for i in cellList:
        cellNum += 1
        if i.x > maxInvade:
            maxInvade = i.x
        if i.y <= (Wit / 6):
            conf3L += 1
        if (Wit / 6) < i.y <= (Wit / 3):
            conf2L += 1
        if (Wit / 3) < i.y <= (2 * Wit / 3):
            conf1L += 1
        if (2 * Wit / 3) < i.y <= (5 * Wit / 6):
            conf2L += 1
        if i.y > (5 * Wit / 6):
            conf3L += 1
        if i.cellType == 'F':
            if i.y <= (Wit / 6):
                conf3F += 1
            if (Wit / 6) < i.y <= (Wit / 3):
                conf2F += 1
            if (Wit / 3) < i.y <= (2 * Wit / 3):
                conf1F += 1
            if (2 * Wit / 3) < i.y <= (5 * Wit / 6):
                conf2F += 1
            if i.y > (5 * Wit / 6):
                conf3F += 1
        #Append x coordinate of cell to list
        cellXList.append(i.y)

    #Update breaking variables
    for i in cellList:
        if i.cellType == 'F':
            totFol += 1
            if i.attachedTo == 0:
                breakFol += 1

    #Append confinement data to lists
    conf1LList.append(conf1L / (conf1L + conf2L + conf3L))
    conf2LList.append(conf2L / (conf1L + conf2L + conf3L))
    conf3LList.append(conf3L / (conf1L + conf2L + conf3L))
    conf1FList.append(conf1F / (conf1F + conf2F + conf3F))
    conf2FList.append(conf2F / (conf1F + conf2F + conf3F))
    conf3FList.append(conf3F / (conf1F + conf2F + conf3F))
    #Append stream membership data to list
    streamFList.append(breakFol / totFol)
    #Append invasion distance data to list
    distLList.append(maxInvade)
    #Append cell separation data to list
    meanSepList.append(meanDistance(cellList))
    #Append cell number data to list
    cellNumList.append(cellNum)

#Increase space between subplots
fig.tight_layout(pad=2.5)

#Produce .MP4 file of simulation
if animate:
    #Images
    ims = []
    for i in range(len(cellMast)):
        #Clear current axes
        ax[0].clear()
        ax[1].clear()
        #Set axes limits
        ax[0].set_xlim(0, Len)
        ax[1].set_xlim(0, Len)

        #Add filopodia patches
        for j in filMast[i]:
            rec0 = patches.Rectangle((round(j[0]), round(j[1])), width=0.15, \
                                      height=j[4], angle=j[2], color=j[3], \
                                      alpha=1, zorder=1)
            rec1 = patches.Rectangle((round(j[0]), round(j[1])), width=0.15, \
                                      height=j[4], angle=j[2], color=j[3], \
                                      alpha=1, zorder=1)
            ax[0].add_patch(rec0)
            ax[1].add_patch(rec1)

        #Add cell patches
        for j in cellMast[i]:
            if j.cellType == 'L':
                ax[0].add_patch(patches.Circle((j.x - 0.5, j.y - 0.5), \
                                cellRad, linewidth = 0, edgecolor = 'g', \
                                facecolor = 'g'))
            elif j.cellType == 'F' and j.attachedTo == 0:
                ax[0].add_patch(patches.Circle((j.x - 0.5, j.y - 0.5), \
                                cellRad, linewidth = 0, edgecolor = 'r', \
                                facecolor = 'r'))
            elif j.cellType == 'F' and j.attachedTo != 0:
                ax[0].add_patch(patches.Circle((j.x - 0.5, j.y - 0.5), \
                                cellRad, linewidth = 0, edgecolor = 'b', \
                                facecolor = 'b'))

            if j.cellType == 'L':
                ax[1].add_patch(patches.Circle((j.x - 0.5, j.y - 0.5), \
                                cellRad, linewidth = 0, edgecolor = 'g', \
                                facecolor = 'g'))
            elif j.cellType == 'F' and j.attachedTo == 0:
                ax[1].add_patch(patches.Circle((j.x - 0.5, j.y - 0.5), \
                                cellRad, linewidth = 0, edgecolor = 'r', \
                                facecolor = 'r'))
            elif j.cellType == 'F' and j.attachedTo != 0:
                ax[1].add_patch(patches.Circle((j.x - 0.5, j.y - 0.5), \
                                cellRad, linewidth = 0, edgecolor = 'b', \
                                facecolor = 'b'))

        #Show VEGF Profile
        im0 = ax[0].imshow(DANMast[i], interpolation = 'none', vmin = 0, \
                           vmax = np.amax(VEGFMast[i]))
        #Show DAN Profile
        im1 = ax[1].imshow(Colec12TrailMast[i], interpolation = 'none', vmin = 0, \
                           vmax = np.amax(Colec12TrailMast[i]))
        #Title
        ax[0].set_title('Leader-Follower Simulation' \
        ' (VEGF) [24h]')
        ax[1].set_title('Leader-Follower Simulation' \
        ' (Colec12Trail) [24h]')

        #Colorbar
        cb0 = fig.colorbar(im0, shrink=0.75, aspect=3, ax=ax[0])
        cb1 = fig.colorbar(im1, shrink=0.75, aspect=3, ax=ax[1])

        #Save visualisation to folder
        plt.savefig('LeaderFollower{}/image{}.png'.format(date, i))
        #Remove colorbar for visualisation
        cb0.remove()
        cb1.remove()

    #Produce video from folder
    with imageio.get_writer('LeaderFollowerColec12TrailConfinement{}.mp4'.\
                            format(date), mode='I', fps=10) as writer:
        for i in range(len(cellMast)):
            filename = 'LeaderFollower{}/image{}.png'.format(date, i)
            image = imageio.imread(filename)
            writer.append_data(image)
            os.remove(filename)

#Output simulation data to .txt file
if not animate:
    #Create data file for output
    file = open('D={}.txt'.format(float(sys.argv[1])), 'a')
    #Iterate over data lists
    for i in range(len(conf1LList)):
        #Write data to .txt. file in columns
        file.write(str(conf1LList[i]) + " " + str(conf2LList[i]) + " " + \
                   str(conf3LList[i]) + " " + str(conf1FList[i]) + " " + \
                   str(conf2FList[i]) + " " + str(conf3FList[i]) + " " + \
                   str(streamFList[i]) + " " + str(distLList[i]) + " " + \
                   str(meanSepList[i]) + " " + str(cellNumList[i]) + "\n")
    #Close file after writing
    file.close()

    #Create data file for output
    file = open('DX={}.txt'.format(float(sys.argv[1])), 'a')
    #Iterate over data lists
    for i in range(len(cellXList)):
        #Write data to .txt. file in columns
        file.write(str(cellXList[i]) + "\n")
    #Close file after writing
    file.close()

#Delete folder used to make MP4
else:
    os.rmdir('LeaderFollower' + date)
