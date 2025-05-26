'''
VEGF.py - Samuel Johnson - 01/06/24
'''

import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from scipy import signal
from numba import jit, prange

'''
Create array to hold VEGF (initial value = c0)
'''
def createVEGFArray(width, length, c0, bcParam):
    #Array initially containing VEGF
    VEGFArray = np.full((width - 2 * bcParam, length - 2 * bcParam), c0)
    #Smoothing
    for i in prange(bcParam - 1, -1, -1):
        VEGFArray = np.pad(VEGFArray, pad_width=1, constant_values= \
                           (i / bcParam * c0))
    #Return smoothed VEGF array
    return VEGFArray

'''
Create array to hold DAN (initial value = c0)
'''
def createDANArray(width, length, c0, bcParam, expLen):
    #Proportion of array initially containing DAN
    DANArray = np.full((width, round(expLen)), c0)
    #Edge case of 50% expression length
    if round(expLen) == round(length - expLen):
        #Remainder of DAN array is empty
        DANArray = np.concatenate((DANArray, np.zeros((width, \
                               round(length - expLen) - 1))), axis=1)
    else:
        #Remainder of DAN array is empty
        DANArray = np.concatenate((DANArray, np.zeros((width, \
                               round(length - expLen)))), axis=1)
    #Return DAN array
    return DANArray

'''
Calculate and return diffusion term of the PDE (Rescaled)
'''
def diffusion(VEGFArray, D, L, meshScale):
    #Mesh dimension
    meshL = VEGFArray.shape[1]
    #Scaling of mesh (Δx)
    xScale = L / meshL
    #Scaling of mesh (Δy)
    yScale = meshScale
    #Rescaled 2D Laplacian Kernel
    kernel = [[0.0, yScale**-2, 0.0],
              [xScale**-2 * 1.0, - 2.0 * yScale**-2 - \
               2.0 * xScale**-2, xScale**-2 * 1.0], \
              [0.0, 1.0 * yScale**-2, 0.0]]
    #2D-Laplacian for diffusion term
    VEGFArray = signal.convolve2d(VEGFArray, kernel, mode='same', \
                                boundary='wrap')
    #Return diffusion term of the PDE (with rescaling)
    return np.multiply(VEGFArray, D)

'''
Calculate and return the logistic term of the PDE
'''
@jit(nopython = True)
def logistic(VEGFArray, chi):
    #Lattice dimensions
    width = VEGFArray.shape[0]
    length = VEGFArray.shape[1]
    #(1 - c) Array
    logistArray = np.subtract(np.ones((width, length)), VEGFArray)
    #Return logistic term of the PDE
    return np.multiply(chi, np.multiply(VEGFArray, logistArray))

'''
Calculate and return the internalisation term of the PDE (Rescaled)
'''
def summation(VEGFArray, cellList, lmbd, R, searchRad, L, meshScale):
    #Lattice dimensions
    width, length = VEGFArray.shape
    #Mesh dimension
    meshL = length
    #Array to store cell positions
    cellPositions = np.array([[k.y / meshScale, k.x * meshL / L] \
                                for k in cellList])
    #Arrays to store the rows and columns for summation
    rows = np.arange(width)[:, np.newaxis]
    cols = np.arange(length)[np.newaxis, :]
    #Compute exponentials in the summation (with re-scaling)
    rowDiff = rows - cellPositions[:, 0][:, np.newaxis, np.newaxis]
    colDiff = cols - cellPositions[:, 1][:, np.newaxis, np.newaxis]
    summFinArray = np.exp(-(((rowDiff * meshScale)**2) + ((L / meshL)**2 * colDiff**2)) \
                   / (2 * R**2))
    #Sum contributions from each cell
    summFinArray = np.sum(summFinArray, axis=0)
    #Multiply the final array by concentration and internalisation rate
    summFinArray = np.multiply(summFinArray, VEGFArray)
    summFinArray = np.multiply(summFinArray, lmbd / (2 * np.pi * R**2))
    #Return uptake term of the PDE
    return summFinArray

'''
Calculate and return the dilution term of the VEGF PDE (Rescaled)
'''
@jit(nopython = True)
def dilution(VEGFArray, L, Ldot):
    #Return dilution term of the PDE
    return np.multiply(Ldot / L, VEGFArray)

'''
Calculate and return an updated VEGF matrix from the VEGF PDE
'''
def updateVEGF(VEGFArray, D, chi, lmbd, R, posList, dt, subStep, \
               searchRad, L, W, Ldot, meshScale):
    #Updated VEGF array calculated by Taylor expansion
    #c(x, t + delta t) = c(x, t) + delta t * c'(x, t)
    VEGFArray = VEGFArray + \
            np.multiply(dt / subStep, diffusion(VEGFArray, D, L, meshScale)) - \
            np.multiply(dt / subStep, dilution(VEGFArray, L, Ldot)) + \
            np.multiply(dt / subStep, logistic(VEGFArray, chi)) - \
            np.multiply(dt / subStep, summation(VEGFArray, posList, \
                        lmbd, R, searchRad, L, meshScale))
    #Zero-flux boundary conditions
    VEGFArray[:, 0] = VEGFArray[:, 1]
    VEGFArray[:, -1] = VEGFArray[:, -2]
    #Updated VEGF array with zero-flux boundary conditions
    return VEGFArray

'''
Calculate and return an updated DAN matrix from the DAN PDE
'''
def updateDAN(DANArray, D, chi, lmbd, R, posList, dt, subStep, \
               searchRad, L, W, Ldot, meshScale):
    #Updated DAN array calculated by Taylor expansion
    #c(x, t + delta t) = c(x, t) + delta t * c'(x, t)
    DANArray = DANArray + \
             np.multiply(dt / subStep, diffusion(DANArray, D, L, meshScale)) - \
             np.multiply(dt / subStep, dilution(DANArray, L, Ldot)) - \
             np.multiply(dt / subStep, summation(DANArray, posList, \
                        lmbd, R, searchRad, L, meshScale))
    #Zero-flux boundary conditions
    DANArray[:, 0] = DANArray[:, 1]
    DANArray[:, -1] = DANArray[:, -2]
    #Updated DAN array with zero-flux boundary conditions
    return DANArray

'''
Calculate and return an updated VEGF matrix from the VEGF PDE
'''
def updateChem(ChemArray, D, chi, lmbd, R, posList, dt, subStep, \
               searchRad, L, W, Ldot, meshScale):
    #Updated Chem array calculated by Taylor expansion
    #c(x, t + delta t) = c(x, t) + delta t * c'(x, t)
    ChemArray = ChemArray + \
            np.multiply(dt / subStep, diffusion(ChemArray, D, L, meshScale)) - \
            np.multiply(dt / subStep, dilution(ChemArray, L, Ldot)) + \
            np.multiply(dt / subStep, logistic(ChemArray, chi))
    #Zero-flux boundary conditions
    ChemArray[:, 0] = ChemArray[:, 1]
    ChemArray[:, -1] = ChemArray[:, -2]
    #Updated Chem array with zero-flux boundary conditions
    return ChemArray
