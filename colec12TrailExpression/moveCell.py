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
def moveCells(VEGFArray, DANArray, Colec12TrailArray, cellList, filoNum, lenFilo, \
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
        #Current Colec12Trail concentration
        tOld = Colec12TrailArray[round(i.y), round(i.x)]
        #Speed modulation from DAN
        danFac = (c0 - DANArray[round(i.y), round(i.x)]) / c0
        #Colec12Trail detection variable
        detectColec12Trail = False
        #Colec12Trail conditions
        try:
            if Colec12TrailArray[round(i.y + cellRad), round(i.x)] > xi * c0 or \
            Colec12TrailArray[round(i.y - cellRad), round(i.x)] > xi * c0:
                leadSpeedCurr = 0.75 * leadSpeed
                folSpeedCurr = 0.75 * folSpeed
                #Colec12 lengthens protrusions and increases branching
                filoNumCurr = 5 * filoNum
                lenFiloCurr = 1.5 * lenFilo
                lenFiloMaxCurr = 1.5 * lenFiloMax
                detectColec12Trail = True
            else:
                leadSpeedCurr = leadSpeed
                folSpeedCurr = folSpeed
                filoNumCurr = filoNum
                lenFiloCurr = lenFilo
                lenFiloMaxCurr = lenFiloMax
        #Cell at extremity of domain
        except IndexError:
            leadSpeedCurr = 0.75 * leadSpeed
            folSpeedCurr = 0.75 * folSpeed
            #Colec12 lengthens protrusions and increases branching
            filoNumCurr = 5 * filoNum
            lenFiloCurr = 1.5 * lenFilo
            lenFiloMaxCurr = 1.5 * lenFiloMax
            detectColec12Trail = True

        #Initialise counting variable for filopodium retraction
        if not hasattr(i, 'filPersist'):
            i.filPersist = 0
        if not hasattr(i, 'angleList') or i.filPersist > filPersist \
            or not detectColec12Trail:
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
            #Initialise optimum direction
            maxAng = random.uniform(0, 2 * math.pi)
            #Initialise maximum gradient
            maxGrad = 0
            #Movement Boolean
            moved = False
            #Sample in multiple directions sequentially
            for p in i.angleList:
                #Random angle of filopodial extension
                filAngle = p
                #Lattice sites spanned by filopodium
                filList = []
                #Append concentration at sites to filopodium list
                for j in range(1, round(lenFiloCurr + 1)):
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
                    if cellRad < (i.x + leadSpeedCurr * danFac * dx * \
                    math.cos(maxAng)) < length - cellRad \
                    and cellRad < (i.y + leadSpeedCurr * danFac * dy * \
                    math.sin(maxAng)) < width - cellRad and \
                    detectCollision(i, cellList, dx, dy, maxAng, \
                    expLen, cellRad, width, leadSpeedCurr, \
                    folSpeedCurr, DANArray, c0) == False:
                        #Movement in direction of gradient
                        i.x += leadSpeedCurr * dx * math.cos(maxAng) \
                             * danFac
                        i.y += leadSpeedCurr * dy * math.sin(maxAng) \
                             * danFac
                        moved = True
                        #Chain angle
                        i.chainAngle = maxAng

            #VEGF in current position is zero
            except ValueError or RuntimeWarning:
                #Burg-Purcell detection limit
                if (maxGrad - cOld) >= xi * math.sqrt(c0):
                    if cellRad < (i.x + leadSpeedCurr * danFac * dx * \
                    math.cos(maxAng)) < length - cellRad \
                    and cellRad < (i.y + leadSpeedCurr * danFac * dy * \
                    math.sin(maxAng)) < width - cellRad and \
                    detectCollision(i, cellList, dx, dy, \
                    maxAng, expLen, cellRad, width, leadSpeedCurr, \
                    folSpeedCurr, DANArray, c0) == False:
                        #Movement in direction of gradient
                        i.x += leadSpeedCurr * dx * math.cos(maxAng) * danFac
                        i.y += leadSpeedCurr * dy * math.sin(maxAng) * danFac
                        moved = True
                        #Chain angle
                        i.chainAngle = maxAng

            #Unsuccessful sampling
            if not moved:
                #Random walk
                filAngle = random.uniform(0, 2 * math.pi)
                #Attempt movement in random direction
                if cellRad < (i.x + leadSpeedCurr * danFac * dx * \
                    math.cos(filAngle)) < (length - cellRad) \
                    and cellRad < (i.y + leadSpeedCurr * danFac * dy * \
                    math.sin(filAngle)) < (width - cellRad) and \
                    detectCollision(i, cellList, dx, dy, filAngle, \
                    expLen, cellRad, width, leadSpeedCurr, folSpeedCurr, \
                    DANArray, c0) == False:
                    #Movement in random direction
                    i.x += leadSpeedCurr * dx * math.cos(filAngle) * danFac
                    i.y += leadSpeedCurr * dy * math.sin(filAngle) * danFac
                    #Chain angle
                    i.chainAngle = filAngle
                for m in range(len(i.angleList)):
                    #Display filopodium
                    filopList.append([i.x, i.y, i.angleList[m] * (180 / math.pi) \
                                    + 270, 'r', lenFiloCurr])

            #Successful sampling
            if moved:
            #Display filopodium
                filopList.append([i.x, i.y, maxAng * (180 / math.pi) \
                + 270, 'k', lenFiloCurr])

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
                    #Colec12Trail not detected
                    if not detectColec12Trail:
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

                    #Colec12Trail detected
                    else:
                        #Initialise optimum direction
                        maxAng = random.uniform(0, 2 * math.pi)
                        #Initialise maximum gradient
                        maxGrad = float('inf')
                        #Movement Boolean
                        moved = False
                        #Sample in multiple directions sequentially
                        for p in i.angleList:
                            #Random angle of filopodial extension
                            filAngle = p
                            #Lattice sites spanned by filopodium
                            filList = []
                            #Append concentration at sites to filopodium list
                            for j in range(1, round(lenFiloCurr + 1)):
                                if (0 < round(i.y + j * math.sin(filAngle)) < width) \
                                and (0 < round(i.x + j * math.cos(filAngle)) < length):
                                    filList.append(Colec12TrailArray[round(i.y + j * \
                                    math.sin(filAngle)), round(i.x + j * \
                                    math.cos(filAngle))])
                                #Cannot sample outside of simulation domain
                                else:
                                    break
                            #Integrate VEGF over filopodium
                            cNew = np.mean(filList)
                            #Update maximum gradient and optimum direction
                            if cNew < maxGrad:
                                maxGrad = cNew
                                maxAng = filAngle

                        #Burg-Purcell detection limit
                        if ((maxGrad - tOld) / tOld) <= - xi * math.sqrt(c0 / tOld):
                            if cellRad < (i.x + leadSpeedCurr * danFac * dx * \
                            math.cos(maxAng)) < length - cellRad \
                            and cellRad < (i.y + leadSpeedCurr * danFac * dy * \
                            math.sin(maxAng)) < width - cellRad and \
                            detectCollision(i, cellList, dx, dy, maxAng, \
                            expLen, cellRad, width, leadSpeedCurr, folSpeedCurr, DANArray, c0) \
                            == False:
                                #Movement in direction of gradient
                                i.x += leadSpeedCurr * dx * math.cos(maxAng) * danFac
                                i.y += leadSpeedCurr * dy * math.sin(maxAng) * danFac
                                visAngle = maxAng
                                moved = True

                        else:
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
                            #Display filopodium
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

                #Booleans for movement due to factors
                moveByColec12 = False
                moveByTrail = False

                #Movement Boolean
                moved = False
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

                if detectColec12Trail:
                    #Initialise optimum direction
                    maxAng = random.uniform(0, 2 * math.pi)
                    #Initialise maximum gradient
                    maxGrad = float('inf')
                    #Sample in multiple directions sequentially
                    for p in i.angleList:
                        #Random angle of filopodial extension
                        filAngle = p
                        #Lattice sites spanned by filopodium
                        filList = []
                        #Append concentration at sites to filopodium list
                        for j in range(1, round(lenFiloCurr + 1)):
                            if (0 < round(i.y + j * math.sin(filAngle)) < width) \
                            and (0 < round(i.x + j * math.cos(filAngle)) < length):
                                filList.append(Colec12TrailArray[round(i.y + j * \
                                math.sin(filAngle)), round(i.x + j * \
                                math.cos(filAngle))])
                            #Cannot sample outside of simulation domain
                            else:
                                break
                        #Integrate VEGF over filopodium
                        cNew = np.mean(filList)
                        #Update maximum gradient and optimum direction
                        if cNew < maxGrad:
                            maxGrad = cNew
                            maxAng = filAngle

                    #Burg-Purcell detection limit
                    if ((maxGrad - tOld) / tOld) <= - xi * math.sqrt(c0 / tOld):
                        if cellRad < (i.x + leadSpeedCurr * danFac * dx * \
                        math.cos(maxAng)) < length - cellRad \
                        and cellRad < (i.y + leadSpeedCurr * danFac * dy * \
                        math.sin(maxAng)) < width - cellRad and \
                        detectCollision(i, cellList, dx, dy, maxAng, \
                        expLen, cellRad, width, leadSpeedCurr, folSpeedCurr, DANArray, c0) \
                        == False:
                            moveByColec12 = True

                    #Move in direction of chain
                    filAngle = i.chainAngle

                    #Move towards chained cell if Trail detected
                    if cellRad < (i.x + folSpeedCurr * danFac * dx * math.cos(visAngle)) \
                    < (length - cellRad) and cellRad < (i.y + folSpeedCurr * danFac * dy * \
                    math.sin(visAngle)) < (width - cellRad) and \
                    detectCollision(i, cellList, dx, dy, visAngle, expLen, \
                    cellRad, width, leadSpeedCurr, folSpeedCurr, DANArray, c0) == False:
                        moveByTrail = True

                    if moveByColec12 and moveByTrail:

                        #Vector-based averaging of visAngle and maxAng
                        #Convert angles to unit vectors
                        vec_vis = np.array([math.cos(visAngle), math.sin(visAngle)])
                        vec_max = np.array([math.cos(maxAng), math.sin(maxAng)])

                        #Sum the vectors
                        vec_sum = vec_vis + vec_max

                        #Check if the resultant vector is non-zero to avoid division by zero
                        if np.linalg.norm(vec_sum) != 0:
                            #Calculate the average angle using arctan2
                            totAng = math.atan2(vec_sum[1], vec_sum[0])
                            #Normalize the angle to [0, 2*pi)
                            totAng = totAng % (2 * math.pi)

                            #Calculate new potential position
                            new_x = i.x + folSpeedCurr * danFac * dx * math.cos(totAng)
                            new_y = i.y + folSpeedCurr * danFac * dy * math.sin(totAng)

                            #Check boundary conditions and collision
                            if (cellRad < new_x < length - cellRad) and (cellRad < new_y < width - cellRad) and \
                                not detectCollision(i, cellList, dx, dy, totAng, expLen, cellRad, width,
                                                    leadSpeedCurr, folSpeedCurr, DANArray, c0):
                                #Move to the averaged direction
                                i.x = new_x
                                i.y = new_y
                                moved = True
                                #Append filopodia for visualization
                                filopList.append([i.x, i.y, totAng * (180 / math.pi) + 270, 'k', lenFiloCurr])
                        else:
                            #If vectors cancel out, fallback to random direction
                            totAng = random.uniform(0, 2 * math.pi)
                            new_x = i.x + folSpeedCurr * danFac * dx * math.cos(totAng)
                            new_y = i.y + folSpeedCurr * danFac * dy * math.sin(totAng)
                            if (cellRad < new_x < length - cellRad) and (cellRad < new_y < width - cellRad) and \
                                not detectCollision(i, cellList, dx, dy, totAng, expLen, cellRad, width, leadSpeedCurr, folSpeedCurr, DANArray, c0):
                                i.x = new_x
                                i.y = new_y
                                moved = True
                                i.chainAngle = totAng
                                filopList.append([i.x, i.y, totAng * (180 / math.pi) + 270, 'r', lenFiloCurr])

                    elif moveByColec12 and not moveByTrail:
                        i.x += folSpeedCurr * danFac * dx * math.cos(maxAng)
                        i.y += folSpeedCurr * danFac * dy * math.sin(maxAng)
                        moved = True

                    elif not moveByColec12 and moveByTrail:
                        i.x += folSpeedCurr * danFac * dx * math.cos(visAngle)
                        i.y += folSpeedCurr * danFac * dy * math.sin(visAngle)
                        moved = True

                    elif cellRad < (i.x + folSpeedCurr * danFac * dx * math.cos(randAng)) \
                     < (length - cellRad) and cellRad < (i.y + folSpeedCurr * danFac * dy * \
                     math.sin(randAng)) < (width - cellRad) and \
                    detectCollision(i, cellList, dx, dy, randAng, expLen, \
                     cellRad, width, leadSpeedCurr, folSpeedCurr, DANArray, c0) == False:
                        i.x += folSpeedCurr * danFac * dx * math.cos(randAng)
                        i.y += folSpeedCurr * danFac * dy * math.sin(randAng)


                #Move in direction of leader cell in chain if Trail not detected
                if not detectColec12Trail:
                    if cellRad < (i.x + folSpeedCurr * danFac * dx * math.cos(i.chainAngle)) \
                     < (length - cellRad) and cellRad < (i.y + folSpeedCurr * danFac * dy * \
                     math.sin(i.chainAngle)) < (width - cellRad) and \
                    detectCollision(i, cellList, dx, dy, i.chainAngle, expLen, \
                     cellRad, width, leadSpeedCurr, folSpeedCurr, DANArray, c0) == False:
                        i.x += folSpeedCurr * danFac * dx * math.cos(i.chainAngle)
                        i.y += folSpeedCurr * danFac * dy * math.sin(i.chainAngle)
                        moved = True
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
        i.x = j.x
        i.y = j.y
        j.x = xTemp
        j.y = yTemp
