'''
VEGF.py - Samuel Johnson - 20/10/23
'''

import numpy as np
import math
import copy
from scipy import signal
from numba import jit, prange
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

'''
Create array to hold VEGF (initial value = c0)
'''
def createVEGFArray(width, length, c0, bcParam):
    #Array initially containing VEGF
    VEGFArray = np.full((width - 2 * bcParam, length - 2 * bcParam), c0)
    #Dirichlet boundary conditions and smoothing
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
Calculate and return diffusion term of the PDE (Rescaled) - Implicit
'''
def crankNicolsonDiffusion(VEGFArray, D, L, W, dt):
    #Mesh dimension
    meshWit, meshLen = VEGFArray.shape
    #Lattice spacings
    dy = W / meshWit
    dx = L / meshLen
    #Diffusion constant (x) depends on scaling 
    alpha = D * dt / dx**2
    beta = D * dt / dy**2
    #Construct implicit matrix (for simplicity, using periodic boundary conditions)
    diagonals = [
    (1 + 2*alpha + 2*beta) * np.ones(meshLen * meshWit),  
    -alpha * np.ones(meshLen * meshWit - 1),
    -alpha * np.ones(meshLen * meshWit - 1),
    -beta * np.ones(meshLen * meshWit - meshLen),
    -beta * np.ones(meshLen * meshWit - meshLen),
]
    offsets = [0, 1, -1, meshLen, -meshLen]
    A = diags(diagonals, offsets=offsets).tocsc()

    #Time-stepping 
    b = u.ravel()
    return spsolve(A, b).reshape((meshWit, meshLen))

'''
Calculate and return diffusion term of the PDE (Rescaled)
'''
def diffusion(VEGFArray, D, L):
    #Mesh dimension
    meshL = VEGFArray.shape[1]
    #Scaling of mesh
    scaleFactor = meshL / L
    #Rescaled 2D Laplacian Kernel
    kernel = [[0.0, 1.0, 0.0],
              [scaleFactor**2 * 1.0, - 2.0 - \
               scaleFactor**2 * 2.0, scaleFactor**2 * 1.0], \
              [0.0, 1.0, 0.0]]
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
    cellPositions = np.array([[k.y / meshScale, k.x * meshL / (L * meshScale)] \
                                for k in cellList])
    #Arrays to store the rows and columns for summation
    rows = np.arange(width)[:, np.newaxis]
    cols = np.arange(length)[np.newaxis, :]
    #Compute exponentials in the summation (with re-scaling)
    rowDiff = rows - cellPositions[:, 0][:, np.newaxis, np.newaxis]
    colDiff = cols - cellPositions[:, 1][:, np.newaxis, np.newaxis]
    summFinArray = np.exp(-((meshScale * rowDiff**2) + ((L * meshScale / meshL)**2 * colDiff**2)) \
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
            np.multiply(dt / subStep, logistic(VEGFArray, chi)) - \
            np.multiply(dt / subStep, summation(VEGFArray, posList, \
                        lmbd, R, searchRad, L, meshScale)) - \
            np.multiply(dt / subStep, dilution(VEGFArray, L, Ldot))
    #Diffusion 
    VEGFArray = crankNicolsonDiffusion(VEGFArray, D, L, W, dt)
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
    DANArray = DANArray - \
            np.multiply(dt / subStep, summation(DANArray, posList, \
                        lmbd, R, searchRad, L, meshScale)) - \
            np.multiply(dt / subStep, dilution(DANArray, L, Ldot))
    #Diffusion 
    DANArray = crankNicolsonDiffusion(DANArray, D, L, W, dt)
    #Zero-flux boundary conditions
    DANArray[:, 0] = DANArray[:, 1]
    DANArray[:, -1] = DANArray[:, -2]
    #Updated DAN array with zero-flux boundary conditions
    return DANArray
