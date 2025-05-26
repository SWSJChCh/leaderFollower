'''
moveCell.py - Samuel Johnson - 27/05/24
'''

import numpy as np
import math
import random
from collisionCell import *

'''
Move cells according to leader-follower dynamics
'''
def moveCellsTrail(VEGFArray, DANArray, TrailArray, cellList, filoNum, lenFilo, \
              lenFiloMax, xi, c0, cellRad, dx, dy, leadSpeed, folSpeed, \
              expLen, oldLen, newLen, epsilon, filPersist):
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
        #Current Trail concentration
        tOld = TrailArray[round(i.y), round(i.x)]
        #Speed modulation from DAN
        danFac = (c0 - DANArray[round(i.y), round(i.x)]) / c0
        #Trail detection variable
        detectTrail = False
        #Trail conditions
        try:
            if TrailArray[round(i.y + cellRad), round(i.x)] > xi * c0 or \
            TrailArray[round(i.y - cellRad), round(i.x)] > xi * c0:
                #Trail slows cells and increases clustering
                leadSpeedCurr = 0.75 * leadSpeed
                folSpeedCurr = 0.75 * folSpeed
                filoNumCurr = filoNum
                lenFiloCurr = lenFilo
                lenFiloMaxCurr = lenFiloMax
                detectTrail = True
            else:
                leadSpeedCurr = leadSpeed
                folSpeedCurr = folSpeed
                filoNumCurr = filoNum
                lenFiloCurr = lenFilo
                lenFiloMaxCurr = lenFiloMax
        #Cell at extremity of domain
        except IndexError:
            #Trail slows cells and increases clustering
            leadSpeedCurr = 0.75 * leadSpeed
            folSpeedCurr = 0.75 * folSpeed
            filoNumCurr = filoNum
            lenFiloCurr = lenFilo
            lenFiloMaxCurr = lenFiloMax
            detectTrail = True

        #Initialise counting variable for filopodium retraction
        if not hasattr(i, 'filPersist'):
            i.filPersist = 0

        i.angleList = [random.uniform(0, 2 * math.pi) for _ in \
                       range(filoNumCurr)]
        i.filPersist = 0

        #Correct length of angle list for current number of filopodia
        if len(i.angleList) < filoNumCurr:
            for _ in range(filoNumCurr - len(i.angleList)):
                i.angleList.append(random.uniform(0, 2 * math.pi))
        elif len(i.angleList) > filoNumCurr:
            i.angleList = i.angleList[0:filoNumCurr]

        #Cell is leader
        if i.cellType == 'L':
            moved = False
            #Trail is detected by leader
            if detectTrail:
                #Move towards a random follower directly attached to this leader
                chainMembers = [c for c in cellList if c.cellType == 'F' and c.attachedTo == i]
                #Chained cell exists
                if chainMembers:
                    #Random choice for follower driving movement
                    target = random.choice(chainMembers)
                    moveAngle = math.atan2(target.y - i.y, target.x - i.x)
                    #Move if not collision detected
                    if (cellRad < i.x + leadSpeedCurr * danFac * dx * math.cos(moveAngle)\
                        < length - cellRad and
                        cellRad < i.y + leadSpeedCurr * danFac * dy * math.sin(moveAngle)\
                            < width - cellRad and
                        not detectCollision(i, cellList, dx, dy, moveAngle,
                                            expLen, cellRad, width,
                                            leadSpeedCurr, folSpeedCurr, DANArray, c0)):
                        i.x += leadSpeedCurr * danFac * dx * math.cos(moveAngle)
                        i.y += leadSpeedCurr * danFac * dy * math.sin(moveAngle)
                        moved = True
                
                if not moved:
                    #Move in random direction
                    filAngle = random.uniform(0, 2 * math.pi)
                    if (cellRad < i.x + leadSpeedCurr * danFac * dx * math.cos(filAngle) \
                        < length - cellRad and
                        cellRad < i.y + leadSpeedCurr * danFac * dy * math.sin(filAngle) \
                            < width - cellRad and
                        not detectCollision(i, cellList, dx, dy, filAngle,
                                            expLen, cellRad, width,
                                            leadSpeedCurr, folSpeedCurr, DANArray, c0)):
                        i.x += leadSpeedCurr * danFac * dx * math.cos(filAngle)
                        i.y += leadSpeedCurr * danFac * dy * math.sin(filAngle)
                        i.chainAngle = filAngle

                    #Append filopodia angles for visualisation
                    for ang in i.angleList:
                        filopList.append([i.x, i.y, ang * (180 / math.pi) + 270,'r', lenFiloCurr])
                #Append filopodia angles for visualisation
                else:
                    filopList.append([i.x, i.y, i.chainAngle * (180 / math.pi) + 270, 'k', lenFiloCurr])

            else:
                #VEGF chemotaxis
                maxAng = random.uniform(0, 2 * math.pi)
                maxGrad = 0
                #Append concentrations of VEGF along each filopodium
                for filAngle in i.angleList:
                    filList = []
                    for j in range(1, round(lenFiloCurr + 1)):
                        if 0 < round(i.y + j * math.sin(filAngle)) < width \
                            and 0 < round(i.x + j * math.cos(filAngle)) < length:
                            filList.append(VEGFArray[round(i.y + j * math.sin(filAngle)),
                                                     round(i.x + j * math.cos(filAngle))])
                        else:
                            break
                    cNew = np.mean(filList) if filList else 0
                    if cNew > maxGrad:
                        maxGrad = cNew
                        maxAng = filAngle

                try:
                    #Move up VEGF gradient
                    if ((maxGrad - cOld) / cOld) >= xi * math.sqrt(c0 / cOld):
                        if (cellRad < i.x + leadSpeedCurr * danFac * dx * math.cos(maxAng) \
                            < length - cellRad and
                            cellRad < i.y + leadSpeedCurr * danFac * dy * math.sin(maxAng) \
                                < width - cellRad and
                            not detectCollision(i, cellList, dx, dy, maxAng,
                                                expLen, cellRad, width,
                                                leadSpeedCurr, folSpeedCurr, DANArray, c0)):
                            i.x += leadSpeedCurr * danFac * dx * math.cos(maxAng)
                            i.y += leadSpeedCurr * danFac * dy * math.sin(maxAng)
                            i.chainAngle = maxAng
                            moved = True
                
                except (ValueError, RuntimeWarning):
                    #Move up VEGF gradient
                    if (maxGrad - cOld) >= xi * math.sqrt(c0):
                        if (cellRad < i.x + leadSpeedCurr * danFac * dx * math.cos(maxAng) \
                            < length - cellRad and
                            cellRad < i.y + leadSpeedCurr * danFac * dy * math.sin(maxAng) \
                                < width - cellRad and
                            not detectCollision(i, cellList, dx, dy, maxAng,
                                                expLen, cellRad, width,
                                                leadSpeedCurr, folSpeedCurr, DANArray, c0)):
                            i.x += leadSpeedCurr * danFac * dx * math.cos(maxAng)
                            i.y += leadSpeedCurr * danFac * dy * math.sin(maxAng)
                            i.chainAngle = maxAng
                            moved = True

                if not moved:
                    #Move in random direction
                    filAngle = random.uniform(0, 2 * math.pi)
                    if (cellRad < i.x + leadSpeedCurr * danFac * dx * math.cos(filAngle) \
                        < length - cellRad and
                        cellRad < i.y + leadSpeedCurr * danFac * dy * math.sin(filAngle) \
                            < width - cellRad and
                        not detectCollision(i, cellList, dx, dy, filAngle,
                                            expLen, cellRad, width,
                                            leadSpeedCurr, folSpeedCurr, DANArray, c0)):
                        i.x += leadSpeedCurr * danFac * dx * math.cos(filAngle)
                        i.y += leadSpeedCurr * danFac * dy * math.sin(filAngle)
                        i.chainAngle = filAngle
                    for ang in i.angleList:
                        filopList.append([i.x, i.y, ang * (180 / math.pi) + 270,
                                          'r', lenFiloCurr])

                if moved:
                    filopList.append([i.x, i.y, i.chainAngle * (180 / math.pi) + 270,
                                      'k', lenFiloCurr])

        #Cell is follower
        elif i.cellType == 'F':
            #Cell is not currently in chain
            if i.attachedTo == 0:
                #Movement Boolean
                moved = False
                #Search for cell with filopodia
                for p in i.angleList:
                    #Angle of filopodial extension
                    filAngle = p
                    #Cell detection for chaining
                    detect, chainNum, cell = detectChain(i, cellList, dx, dy, \
                    filAngle, lenFiloCurr, cellRad)
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
                        if cellRad < (i.x + folSpeedCurr * danFac * dx * \
                                      math.cos(i.chainAngle)) < (length - cellRad) \
                                      and cellRad < (i.y + folSpeedCurr * danFac * dy \
                                      * math.sin(i.chainAngle)) < (width - cellRad) and \
                                      detectCollision(i, cellList, dx, dy, i.chainAngle, expLen, \
                                      cellRad, width, leadSpeedCurr, folSpeedCurr, DANArray, c0) == False:
                            #Move in direction of filopodium
                            i.x += folSpeedCurr * danFac * dx * math.cos(i.chainAngle)
                            i.y += folSpeedCurr * danFac * dy * math.sin(i.chainAngle)
                            moved = True
                            break

                #Chained cell not detected
                if not moved:
                    #Random walk
                    filAngle = random.uniform(0, 2 * math.pi)
                    #Angle for visualisation
                    visAngle = filAngle
                    #Move in random direction
                    if cellRad < (i.x + folSpeedCurr * danFac * dx * \
                                  math.cos(filAngle)) < (length - cellRad) \
                        and cellRad < (i.y + folSpeedCurr * danFac * dy \
                        * math.sin(filAngle)) < (width - cellRad) \
                        and detectCollision(i, cellList, dx, dy, filAngle, \
                        expLen, cellRad, width, leadSpeedCurr, folSpeedCurr, DANArray, c0) == False:
                        #Move in direction of filopodium
                        i.x += folSpeedCurr * danFac * dx * math.cos(filAngle)
                        i.y += folSpeedCurr * danFac * dy * math.sin(filAngle)
                    for m in range(len(i.angleList)):
                        #Display filopodium
                        filopList.append([i.x, i.y, i.angleList[m] * (180 / math.pi) \
                                        + 270, 'r', lenFiloCurr])

                #Display filopodium
                if moved:
                    filopList.append([i.x, i.y, visAngle * (180 / math.pi) \
                                      + 270, 'k', lenFiloCurr])

            #Cell is currently in chain
            elif i.attachedTo != 0:
                #Angle for visualisation
                visAngle = math.pi + math.atan2(i.y - i.attachedTo.y, i.x - \
                i.attachedTo.x)
                #Filopodia direction for visualisation
                filopList.append([i.x, i.y, visAngle * (180 / math.pi) + 270, \
                 'k', lenFiloCurr])
                #Update chain angle
                i.chainAngle = i.attachedTo.chainAngle
                #Random angle
                randAng = random.uniform(0, 2 * math.pi)
                #Move in direction of chain
                filAngle = i.chainAngle
                #Move in direction of leader cell in chain if Trail not detected
                if not detectTrail:
                    if cellRad < (i.x + folSpeedCurr * danFac * dx * math.cos(filAngle)) \
                     < (length - cellRad) and cellRad < (i.y + folSpeedCurr * danFac * dy * \
                     math.sin(filAngle)) < (width - cellRad) and \
                    detectCollision(i, cellList, dx, dy, filAngle, expLen, \
                     cellRad, width, leadSpeedCurr, folSpeedCurr, DANArray, c0) == False:
                        i.x += folSpeedCurr * danFac * dx * math.cos(filAngle)
                        i.y += folSpeedCurr * danFac * dy * math.sin(filAngle)
                    #If chain movement not possible move in random direction
                    elif cellRad < (i.x + folSpeedCurr * danFac * dx * math.cos(randAng)) \
                     < (length - cellRad) and cellRad < (i.y + folSpeedCurr * danFac * dy * \
                     math.sin(randAng)) < (width - cellRad) and \
                    detectCollision(i, cellList, dx, dy, randAng, expLen, \
                     cellRad, width, leadSpeedCurr, folSpeedCurr, DANArray, c0) == False:
                        i.x += folSpeedCurr * danFac * dx * math.cos(randAng)
                        i.y += folSpeedCurr * danFac * dy * math.sin(randAng)
                #Move towards chained cell if Trail detected
                elif detectTrail:
                    if cellRad < (i.x + folSpeedCurr * danFac * dx * math.cos(visAngle)) \
                     < (length - cellRad) and cellRad < (i.y + folSpeedCurr * danFac * dy * \
                     math.sin(visAngle)) < (width - cellRad) and \
                    detectCollision(i, cellList, dx, dy, visAngle, expLen, \
                     cellRad, width, leadSpeedCurr, folSpeedCurr, DANArray, c0) == False:
                        i.x += folSpeedCurr * danFac * dx * math.cos(visAngle)
                        i.y += folSpeedCurr * danFac * dy * math.sin(visAngle)
                    #If chain movement not possible move in random direction
                    elif cellRad < (i.x + folSpeedCurr * danFac * dx * math.cos(randAng)) \
                     < (length - cellRad) and cellRad < (i.y + folSpeedCurr * danFac * dy * \
                     math.sin(randAng)) < (width - cellRad) and \
                    detectCollision(i, cellList, dx, dy, randAng, expLen, \
                     cellRad, width, leadSpeedCurr, folSpeedCurr, DANArray, c0) == False:
                        i.x += folSpeedCurr * danFac * dx * math.cos(randAng)
                        i.y += folSpeedCurr * danFac * dy * math.sin(randAng)

                #Unchain cell and all following
                if not math.sqrt((i.x - i.attachedTo.x)**2 + (i.y - \
                i.attachedTo.y)**2) <= (lenFiloMaxCurr):
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

        i.filPersist += 1

    #Return orientation of filopodia for persistence and visualisation
    return(filopList)

'''
Switch phenotypes of follower cells ahead of leader cells
'''
def updatePhenotype(i, cellList, VEGFArray, epsilon):
    width, length = VEGFArray.shape
    #List of leader cells
    itList = [j for j in cellList if j.cellType == 'L']
    #Minimum distance variable
    minDist = float('inf')
    minDistCell = None
    #Obtain furthest back leader
    for j in itList:
        if lineSegDist(0, width//3, 0, 2 * width//3, j.x, j.y) < minDist:
            minX = j.x
            minY = j.y
            minDist = lineSegDist(0, width//3, 0, 2 * width//3, j.x, j.y)
            minDistCell = j
    #Swap cell positions
    if not minDistCell is None and lineSegDist(0, width//3, 0, 2 * width//3, \
       i.x, i.y) > minDist + epsilon:
        xTemp = i.x
        yTemp = i.y
        i.x = minDistCell.x
        i.y = minDistCell.y
        minDistCell.x = xTemp
        minDistCell.y = yTemp

'''
Iteratively attach all follower cells into chains until the number
of attached cells stabilises.
'''
def chainAtEnd(cellList, dx, dy, lenFilo, cellRad, maxIters=100):

    if maxIters is None:
        maxIters = len(cellList)

    prevAttached = -1

    for iteration in range(maxIters):
        #Count currently chained followers
        attached = sum(1 for c in cellList
                       if c.cellType == 'F' and c.attachedTo != 0)
        #If stable, stop
        if attached == prevAttached:
            break
        prevAttached = attached

        #Try to attach each free follower
        for cell in cellList:
            if cell.cellType == 'F' and cell.attachedTo == 0:
                for ang in cell.angleList:
                    detect, chainNum, leader = detectChain(
                        cell, cellList, dx, dy, ang, lenFilo, cellRad
                    )
                    if detect:
                        cell.attachedTo = leader
                        cell.chain      = leader.chain
                        cell.chainAngle = leader.chainAngle
                        break

    return cellList
