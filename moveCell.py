'''
moveCell.py - Samuel Johnson - 09/10/23
'''

import numpy as np
import math
import random
from collisionCell import *

'''
Move cells according to leader-follower dynamics
'''
def moveCells(VEGFArray, DANArray, cellList, filoNum, lenFilo, lenFiloMax, xi, c0, \
              cellRad, dx, dy, leadSpeed, folSpeed, expLen, persLen, oldLen, \
              newLen, epsilon):
    #Dimensions of array (μm)
    width = VEGFArray.shape[0]
    length = VEGFArray.shape[1]
    #List of filopodial orientations for visualisation
    filopList = []
    #Cells considered for movement in random order
    random.shuffle(cellList)
    #Consider each cell for movement
    for i in cellList:
        #Current VEGF concentration
        cOld = VEGFArray[round(i.y), round(i.x)]
        #Speed modulation from DAN
        danFac = (c0 - DANArray[round(i.y), round(i.x)]) / c0

        #Cell is leader
        if i.cellType == 'L':
            #Reset persistence cycle if limit exceeded
            if i.pers > persLen:
                i.pers = 0
            #Start of persistence cycle
            if i.pers == 0:
                #Initialise optimum direction
                maxAng = random.uniform(0, 2 * math.pi)
                #Initialise maximum gradient
                maxGrad = 0
                #Movement Boolean
                moved = False
                #Sample in multiple directions sequentially
                for _ in range(filoNum):
                    #Random angle of filopodial extension
                    filAngle = random.uniform(0, 2 * math.pi)
                    #Lattice sites spanned by filopodium
                    filList = []
                    #Append concentration at sites to filopodium list
                    for j in range(1, round(lenFilo + 1)):
                        if (0 < round(i.y + j * math.sin(filAngle)) < width) \
                        and (0 < round(i.x + j * math.cos(filAngle)) < length):
                            filList.append(VEGFArray[round(i.y + j * \
                            math.sin(filAngle)), round(i.x + j * \
                            math.cos(filAngle))])
                        #Cannot sample outside of simulation domain
                        else:
                            break
                    #Integrate VEGF over filopodium
                    cNew = np.mean(filList)
                    #Update maximum gradient and optimum direction
                    if cNew > maxGrad:
                        maxGrad = cNew
                        maxAng = filAngle

                #Move by chemotaxis
                try:
                    #Burg-Purcell detection limit
                    if ((maxGrad - cOld) / cOld) >= xi * math.sqrt(c0 / cOld):
                        if cellRad < (i.x + leadSpeed * danFac * dx * \
                        math.cos(maxAng)) < length - cellRad \
                        and cellRad < (i.y + leadSpeed * danFac * dy * \
                        math.sin(maxAng)) < width - cellRad and \
                        detectCollision(i, cellList, dx, dy, maxAng, \
                        expLen, cellRad, width, leadSpeed, folSpeed, DANArray, c0) \
                        == False:
                            #Movement in direction of gradient
                            i.x += leadSpeed * dx * math.cos(maxAng) * danFac
                            i.y += leadSpeed * dy * math.sin(maxAng) * danFac
                            moved = True
                            #Update persistence distance
                            i.pers += 1
                            #Persistence angle
                            i.persAng = maxAng
                            #Chain angle
                            i.chainAngle = maxAng

                #VEGF in current position is zero
                except ValueError or RuntimeWarning:
                    #Burg-Purcell detection limit
                    if (maxGrad - cOld) >= xi * math.sqrt(c0):
                        if cellRad < (i.x + leadSpeed * danFac * dx * \
                        math.cos(maxAng)) < length - cellRad \
                        and cellRad < (i.y + leadSpeed * danFac * dy * \
                        math.sin(maxAng)) < width - cellRad and \
                        detectCollision(i, cellList, dx, dy, \
                        maxAng, expLen, cellRad, width, leadSpeed, \
                        folSpeed, DANArray, c0) == False:
                            #Movement in direction of gradient
                            i.x += leadSpeed * dx * math.cos(maxAng) * danFac
                            i.y += leadSpeed * dy * math.sin(maxAng) * danFac
                            moved = True
                            #Update persistence distance
                            i.pers += 1
                            #Persistence angle
                            i.persAng = maxAng
                            #Chain angle
                            i.chainAngle = maxAng

                #Unsuccessful sampling
                if not moved:
                    #Random walk
                    filAngle = random.uniform(0, 2 * math.pi)
                    #Attempt movement in random direction
                    if cellRad < (i.x + leadSpeed * danFac * dx * \
                        math.cos(filAngle)) < (length - cellRad) \
                        and cellRad < (i.y + leadSpeed * danFac * dy * \
                        math.sin(filAngle)) < (width - cellRad) and \
                        detectCollision(i, cellList, dx, dy, filAngle, \
                        expLen, cellRad, width, leadSpeed, folSpeed, DANArray, c0) \
                        == False:
                        #Movement in random direction
                        i.x += leadSpeed * dx * math.cos(filAngle) * danFac
                        i.y += leadSpeed * dy * math.sin(filAngle) * danFac
                        #Chain angle
                        i.chainAngle = filAngle
                    #Display filopodium
                    filopList.append([i.x, i.y, filAngle * (180 / math.pi) \
                                      + 270, 'r', lenFilo])

                #Successful sampling
                if moved:
                #Display filopodium
                    filopList.append([i.x, i.y, maxAng * (180 / math.pi) \
                    + 270, 'k', lenFilo])

        #Cell is follower
        elif i.cellType == 'F':

            #Cell is not currently in chain
            if i.attachedTo == 0:
                #Angle of filopodial extension
                filAngle = random.uniform(0, 2 * math.pi)
                #Movement Boolean
                moved = False

                #Search for cell with filopodia
                for p in range(filoNum):
                    #Angle of filopodial extension
                    filAngle = random.uniform(0, 2 * math.pi)
                    #Cell detection for chaining
                    detect, chainNum, cell = detectChain(i, cellList, dx, dy, \
                    filAngle, lenFilo, cellRad)
                    #Cells touched by filopodium
                    if detect == True:
                        #Angle for visualisation
                        visAngle = math.pi + math.atan2(i.y - cell.y, \
                                                        i.x - cell.x)
                        #Update attachment information
                        i.attachedTo = cell
                        i.chain = cell.chain
                        i.chainAngle = cell.chainAngle
                        #Move cell in direction of chain
                        if cellRad < (i.x + folSpeed * danFac * dx * \
                                      math.cos(i.chainAngle)) < (length - cellRad) \
                                      and cellRad < (i.y + folSpeed * danFac * dy \
                                      * math.sin(i.chainAngle)) < (width - cellRad) and \
                                      detectCollision(i, cellList, dx, dy, i.chainAngle, expLen, \
                                      cellRad, width, leadSpeed, folSpeed, DANArray, c0) == False:
                            #Move in direction of filopodium
                            i.x += folSpeed * danFac * dx * math.cos(i.chainAngle)
                            i.y += folSpeed * danFac * dy * math.sin(i.chainAngle)
                            moved = True
                            break

                    #Chained cell not detected
                    if p + 1 == filoNum and not moved:
                        #Random walk
                        filAngle = random.uniform(0, 2 * math.pi)
                        #Angle for visualisation
                        visAngle = filAngle
                        #Move in random direction
                        if cellRad < (i.x + folSpeed * danFac * dx * \
                                      math.cos(filAngle)) < (length - cellRad) \
                            and cellRad < (i.y + folSpeed * danFac * dy \
                            * math.sin(filAngle)) < (width - cellRad) \
                            and detectCollision(i, cellList, dx, dy, filAngle, \
                            expLen, cellRad, width, leadSpeed, folSpeed, DANArray, c0) == False:
                            #Move in direction of filopodium
                            i.x += folSpeed * danFac * dx * math.cos(filAngle)
                            i.y += folSpeed * danFac * dy * math.sin(filAngle)
                        #Display filopodium
                        filopList.append([i.x, i.y, visAngle * (180 / math.pi) \
                                          + 270, 'r', lenFilo])

                #Display filopodium
                if moved:
                    filopList.append([i.x, i.y, visAngle * (180 / math.pi) \
                                      + 270, 'k', lenFilo])

            #Cell is currently in chain
            elif i.attachedTo != 0:
                #Angle for visualisation
                visAngle = math.pi + math.atan2(i.y - i.attachedTo.y, i.x - \
                i.attachedTo.x)
                #Filopodia direction for visualisation
                filopList.append([i.x, i.y, visAngle * (180 / math.pi) + 270, \
                 'k', lenFilo])
                #Update chain angle
                i.chainAngle = i.attachedTo.chainAngle
                #Random angle
                randAng = random.uniform(0, 2 * math.pi)
                #Move in direction of chain
                filAngle = i.chainAngle
                #Move in direction of leader cell in chain
                if cellRad < (i.x + folSpeed * danFac * dx * math.cos(filAngle)) \
                 < (length - cellRad) and cellRad < (i.y + folSpeed * danFac * dy * \
                 math.sin(filAngle)) < (width - cellRad) and \
                detectCollision(i, cellList, dx, dy, filAngle, expLen, \
                 cellRad, width, leadSpeed, folSpeed, DANArray, c0) == False:
                    i.x += folSpeed * danFac * dx * math.cos(filAngle)
                    i.y += folSpeed * danFac * dy * math.sin(filAngle)
                #If chain movement not possible move in random direction
                elif cellRad < (i.x + folSpeed * danFac * dx * math.cos(randAng)) \
                 < (length - cellRad) and cellRad < (i.y + folSpeed * danFac * dy * \
                 math.sin(randAng)) < (width - cellRad) and \
                detectCollision(i, cellList, dx, dy, randAng, expLen, \
                 cellRad, width, leadSpeed, folSpeed, DANArray, c0) == False:
                    i.x += folSpeed * danFac * dx * math.cos(randAng)
                    i.y += folSpeed * danFac * dy * math.sin(randAng)

                #Unchain cell and all following
                if not math.sqrt((i.x - i.attachedTo.x)**2 + (i.y - \
                i.attachedTo.y)**2) <= (lenFiloMax):
                    foundAll = False
                    #List of cells to break attachment
                    breakList = [i]
                    while not foundAll:
                        origLen = len(breakList)
                        for q in cellList:
                            if q.cellType == 'F':
                                if q.attachedTo in breakList and q \
                                not in breakList:
                                    breakList.append(q)
                        finLen = len(breakList)
                        if finLen == origLen:
                            foundAll = True
                    for p in breakList:
                        #Cell of attachment
                        p.attachedTo = 0
                        #Number of chain
                        p.chain = 0
                        #Chain Angle
                        p.chainAngle = 0

            #Update cell phenotype according to position
            updatePhenotype(i, cellList, VEGFArray, epsilon)

        #Advection of cells
        i.x = i.x * (newLen / oldLen)

    #Return orientation of filopodia for persistence and visualisation
    return(filopList)

'''
Switch phenotypes of follower cells ahead of leader cells
'''
def updatePhenotype(i, cellList, VEGFArray, epsilon):
    #List of leader cells
    itList = [j for j in cellList if j.cellType == 'L']
    #Minimum x distance variable
    minX = float('inf')
    minDistCell = None
    #Obtain furthest back leader
    for j in itList:
        if j.x < minX:
            minX = j.x
            minDistCell = j
    #Swap cell positions
    if not minDistCell is None and i.x > minDistCell.x + epsilon:
        xTemp = i.x
        yTemp = i.y
        i.x = j.x
        i.y = j.y
        j.x = xTemp
        j.y = yTemp
