'''
runSimulation.py - Samuel Johnson - 01/06/24
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
import imageio.v2 as imageio
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
from moveCellTrail import *
from moveCellColec12 import * 
from moveCellTrailColec12 import * 
from collisionCell import *
from growthFunction import *
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors

trail = sys.argv[2]
colec12 = sys.argv[3]

if trail.lower() == 'true': 
    trail = True 
else: 
    trail = False

if colec12.lower() == 'true': 
    colec12 = True 
else: 
    colec12 = False

VEGFCMAP = mcolors.LinearSegmentedColormap.from_list(
    'vegf_cmap', ['#FFFFFF', '#ECEC0E']  # min=white, max=yellowish
)
CHEMCMAP = mcolors.LinearSegmentedColormap.from_list(
    'chem_cmap', ['#FFFFFF', '#628D56'] # min=white, max=pinkish
)
DANCMAP = mcolors.LinearSegmentedColormap.from_list(
    'dan_cmap', ['#FFFFFF', '#A9AFB7']   # min=white, max=grayish
)

expLen = 0

stripWidth = float(sys.argv[1])
stripFac = 1 / stripWidth

filPersist = 10

#Animation Boolean
animate = True
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
#Width of PDE mesh (for solver on unit-length mesh)
meshWit = int(120 / meshScale * 4 + 120 * stripWidth / meshScale)
#Domain width (μm)
Wit = int(meshWit * meshScale)
#Boundary condition smoothing parameter
bcParam = 0
#Number of leader cells
leadNum = 5
#Proportion of domain for which DAN is expressed
DANLen = int(0.4 * float(1 * lengthList[-1] / meshScale))
#Proportion of domain for which Protein is expressed
CHEMLen = int(float(1 * lengthList[-1] / meshScale))
#Proportion of domain for which zero-flux boundary conditions are expressed
spanLen = 0
#Timesteps per attempted cell insertion (Note that the actual rate of insertion is significantly lower)
insertStep = 1
#Repeats for data averaging
repeats = 1

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

#List for stream swapping data
swapList = []

#Leader spreading data list
conf1LList = []
conf2LList = []
conf3LList = []

#Follower spreading data list
conf1FList = []
conf2FList = []
conf3FList = []

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
    VEGFMesh = createVEGFArray(120//meshScale, meshLen, c0, bcParam)
    #Initialise DAN Mesh (for solver)
    DANMesh = createDANArray(120//meshScale, meshLen, c0, bcParam, DANLen)
    #Pad VEGF array with empty space on both sides
    padArray = c0 * np.ones((120//meshScale, meshLen))
    halfPadded = np.vstack((padArray, VEGFMesh))
    VEGFPadded = np.vstack((halfPadded, c0 * np.ones((int(120 * stripWidth / (meshScale)), meshLen))))
    VEGFPadded = np.vstack((VEGFPadded, createVEGFArray(120//meshScale, meshLen, c0, bcParam)))
    VEGFPadded = np.vstack((VEGFPadded, c0 * np.ones((120//meshScale, meshLen))))
    VEGFMesh = VEGFPadded

    #Pad DAN array with empty space on both sides
    padArray = np.zeros((120//meshScale, meshLen))
    halfPadded = np.vstack((padArray, DANMesh))
    DANPadded = np.vstack((halfPadded, np.zeros((int(120 * stripWidth / (meshScale)), meshLen))))
    DANPadded = np.vstack((DANPadded, createDANArray(120//meshScale, meshLen, c0, bcParam, DANLen)))
    DANPadded = np.vstack((DANPadded, np.zeros((120//meshScale, meshLen))))
    DANMesh = DANPadded

    #Pad CHEM array
    centerArray = np.zeros((120//meshScale, meshLen))
    halfPadded = np.vstack((centerArray, createDANArray(int(120 * stripWidth / (meshScale)), meshLen, \
                           c0, bcParam, CHEMLen)))
    ChemMesh = np.vstack((createDANArray(120//meshScale, meshLen, \
                           c0, bcParam, CHEMLen), halfPadded))
    ChemMesh = np.vstack((ChemMesh, np.zeros((120//meshScale, meshLen))))
    ChemMesh = np.vstack((ChemMesh, createDANArray(120//meshScale, meshLen, \
                           c0, bcParam, CHEMLen)))
    #Initialise VEGF Array (for cells)
    VEGFArray = cv2.resize(VEGFMesh, (Len, Wit), \
                           interpolation = cv2.INTER_AREA)
    #Initialise DAN Array (for cells)
    DANArray = cv2.resize(DANMesh, (Len, Wit), \
                           interpolation = cv2.INTER_AREA)
    #Initialise DAN Array (for cells)
    ChemArray = cv2.resize(ChemMesh, (Len, Wit), \
                           interpolation = cv2.INTER_AREA)


    #List to store cell objects
    cellList = []
    #Data lists for visualisation
    cellMast = []
    expMast = []
    VEGFMast = []
    DANMast = []
    CHEMMast = []
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
            initConfiguration(cellList, leadNum, Wit, cellRad, lenFilo, stripWidth)
            leaderInsert = True
        if t > firstInsert:
            #List to track cell data at time t
            cellCopyList = []
            #Insert follower cells at constant time intervals
            if counter % insertStep == 0:
                #Insert follower cell
                cell = followerCell(cellRad, lenFilo)
                insertCell(cell, cellList, Wit, Len, 1, stripWidth)
                #Insert follower cell
                cell = followerCell(cellRad, lenFilo)
                insertCell(cell, cellList, Wit, Len, 2, stripWidth)
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
                #Update CHEM
                ChemMesh = updateChem(ChemMesh, D, chi, lmda, cellRad, \
                                      cellList, dt, subStep, searchRad, \
                                      Len, Wit, lenDot, meshScale)
            #Update VEGF Array
            VEGFArray = cv2.resize(VEGFMesh, (Len, Wit), \
                                   interpolation = cv2.INTER_AREA)
            #Update DAN Array
            DANArray = cv2.resize(DANMesh, (Len, Wit), \
                                   interpolation = cv2.INTER_AREA)
            #Update CHEM Array
            ChemArray = cv2.resize(ChemMesh, (Len, Wit), \
                                   interpolation = cv2.INTER_AREA)

            #Move cells
            if not colec12:
                filList = moveCellsTrail(VEGFArray, DANArray, ChemArray, cellList, \
                                filoNum, lenFilo, lenFiloMax, xi, c0, cellRad, \
                                dx, dy, leadSpeed, folSpeed, spanLen, \
                                lengthList[counter - 1], lengthList[counter], \
                                epsilon, filPersist, stripWidth)
            elif not trail:
                filList = moveCellsColec12(VEGFArray, DANArray, ChemArray, cellList, \
                                filoNum, lenFilo, lenFiloMax, xi, c0, cellRad, \
                                dx, dy, leadSpeed, folSpeed, spanLen, \
                                lengthList[counter - 1], lengthList[counter], \
                                epsilon, filPersist, stripWidth)
            else:
                filList = moveCellsTrailColec12(VEGFArray, DANArray, ChemArray, cellList, \
                                filoNum, lenFilo, lenFiloMax, xi, c0, cellRad, \
                                dx, dy, leadSpeed, folSpeed, spanLen, \
                                lengthList[counter - 1], lengthList[counter], \
                                epsilon, filPersist, stripWidth)
                
            #List of cell position and VEGF Array
            if counter % animStep == 0 and animate:
                for i in cellList:
                    cellCopyList.append(copy.deepcopy(i))
                cellMast.append(cellCopyList)
                VEGFMast.append(VEGFArray.copy())
                CHEMMast.append(ChemArray.copy())
                DANMast.append(DANArray.copy())
                filMast.append(filList.copy())
                expMast.append(spanLen)

    #Chain any cells detached due to stochastic effects
    cellList = chainAtEnd(cellList, dx, dy, lenFilo, cellRad)

    #Initialise swapping variables
    swaps = 0
    cellNum = 0
    breakFol = 0
    totFol = 0
    maxInvade = 0

    #Update spreading variables
    for i in cellList:
        if i.stream == 1 and i.y > Wit - 120 * 2 and i.chain != 0:
            swaps += 1
        elif i.stream == 2 and i.y < 120 * 2 and i.chain != 0:
            swaps += 1

        cellNum += 1

        if i.x > maxInvade:
            maxInvade = i.x
            
    #Update breaking variables
    for i in cellList:
        if i.cellType == 'F':
            totFol += 1
            if i.attachedTo == 0:
                breakFol += 1

    swapList.append(swaps)
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

# Produce .MP4 file of simulation
if animate:
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.4)
    fig.text(0.04, 0.5, r'$y(\mu m)$',
             va='center', rotation=0)

    for iFrame in range(len(cellMast)):
        ax[0].clear()
        ax[1].clear()
        ax[2].clear()

        ax[0].set_xlim(0, Len)
        ax[0].set_ylim(0, Wit)
        ax[1].set_xlim(0, Len)
        ax[1].set_ylim(0, Wit)
        ax[2].set_xlim(0, Len)
        ax[2].set_ylim(0, Wit)

        ax[0].set_xlabel(r'$x(\mu m)$')
        ax[1].set_xlabel(r'$x(\mu m)$')
        ax[2].set_xlabel(r'$x(\mu m)$')

        for cell in cellMast[iFrame]:
            if cell.cellType == 'L':
                color = 'g'
            elif cell.cellType == 'F' and cell.stream==1:
                color = 'r'
            elif cell.cellType == 'F' and cell.stream==2:
                color = 'b'


            circ0 = patches.Circle(
                (cell.x - 0.5, cell.y - 0.5),
                cellRad,
                linewidth=0,
                edgecolor=color,
                facecolor=color
            )
            circ1 = copy.copy(circ0)
            circ2 = copy.copy(circ0)
            ax[0].add_patch(circ0)
            ax[1].add_patch(circ1)
            ax[2].add_patch(circ2)

        # Using the three custom color schemes:
        im0 = ax[0].imshow(
            VEGFMast[iFrame],
            interpolation='none',
            vmin=0,
            vmax=np.amax(VEGFMast[iFrame]),
            cmap=VEGFCMAP
        )
        im1 = ax[2].imshow(
            DANMast[iFrame],
            interpolation='none',
            vmin=0,
            vmax=np.amax(DANMast[iFrame]),
            cmap=DANCMAP
        )
        im2 = ax[1].imshow(
            CHEMMast[iFrame],
            interpolation='none',
            vmin=0,
            vmax=np.amax(CHEMMast[iFrame]),
            cmap=CHEMCMAP
        )

        ax[0].set_title('VEGF')
        ax[2].set_title('Dan')
        if trail and colec12: 
            ax[1].set_title('Trail + Colec12')
        elif colec12: 
            ax[1].set_title('Colec12')
        else: 
            ax[1].set_title('Trail')
            

        cb0 = plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04,
                           shrink=0.5, aspect=8)
        cb1 = plt.colorbar(im1, ax=ax[2], fraction=0.046, pad=0.04,
                           shrink=0.5, aspect=8)
        cb2 = plt.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04,
                           shrink=0.5, aspect=8)

        plt.savefig(f'LeaderFollower{date}/image{iFrame}.png', dpi=400)

        cb0.remove()
        cb1.remove()
        cb2.remove()

    with imageio.get_writer(f'LeaderFollowerTrailConfinement{date}.mp4',
                            mode='I', fps=10) as writer:
        for iFrame in range(len(cellMast)):
            filename = f'LeaderFollower{date}/image{iFrame}.png'
            image = imageio.imread(filename)
            writer.append_data(image)
            os.remove(filename)

#Output simulation data to .txt file
if not animate:
    #Create data file for output
    file = open('W={}.txt'.format(float(sys.argv[1])), 'a')
    #Iterate over data lists
    for i in range(len(swapList)):
        #Write data to .txt. file in columns
        file.write(str(streamFList[i]) + " " + str(distLList[i]) + " " + \
                   str(meanSepList[i]) + " " + str(cellNumList[i]) + " " + \
                   str(swapList[i]) + "\n")
    #Close file after writing
    file.close()

#Delete folder used to make MP4
else:
   os.rmdir('LeaderFollower' + date)


