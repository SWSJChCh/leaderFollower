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
def moveCellsTrailColec12(VEGFArray, DANArray, Colec12TrailArray, cellList, filoNum, lenFilo, \
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
            i.angleList = [random.uniform(0, 2 * math.pi) for _ in range(filoNumCurr)]
            i.filPersist = 0
        #Correct length of angle list for current number of filopodia
        if len(i.angleList) < filoNumCurr:
            for _ in range(filoNumCurr - len(i.angleList)):
                i.angleList.append(random.uniform(0, 2 * math.pi))
        elif len(i.angleList) > filoNumCurr:
            i.angleList = i.angleList[0:filoNumCurr]

        if i.cellType == 'L':
            if detectColec12Trail:
                #Find best Colec12 angle
                bestAng = random.uniform(0, 2*math.pi)
                bestVal = np.inf
                moved = False
                for ang in i.angleList:
                    samples = []
                    for j in range(1, round(lenFiloCurr)+1):
                        if 0 < round(i.y + j*math.sin(ang)) < width \
                            and 0 < round(i.x + j*math.cos(ang)) < length:
                            samples.append(Colec12TrailArray[round(i.y + j*math.sin(ang)),
                                                             round(i.x + j*math.cos(ang))])
                        else:
                            break
                    #Update angle if stronger gradient is detected
                    val = np.mean(samples) if samples else np.inf
                    if val < bestVal:
                        bestVal, bestAng = val, ang

                #Burg–Purcell limit on the detected gradient
                if tOld > 0:
                    grad  = (bestVal - tOld) / tOld
                    limit = -xi * math.sqrt(c0 / tOld)
                else:
                    grad, limit = 0, float('inf')

                #Move towards a random follower directly attached to this leader
                attached = [c for c in cellList if c.cellType=='F' and c.attachedTo==i]
                chaseAng = (math.atan2(attached[0].y - i.y, attached[0].x - i.x)
                            if attached else None)

                if grad <= limit:
                    #Combined movement due to Colec12 and Trail
                    v1 = np.array([math.cos(bestAng), math.sin(bestAng)])
                    if chaseAng is not None:
                        v2 = np.array([math.cos(chaseAng), math.sin(chaseAng)])
                        v  = v1 + v2
                    else:
                        v = v1
                    v /= np.linalg.norm(v) or 1.0
                    moveAng = math.atan2(v[1], v[0])
                    i.chainAngle = bestAng

                else:
                    #Move only according to Trail if no Colec12 gradient is detected
                    if chaseAng is not None:
                        moveAng = chaseAng
                    else:
                        #No follower to move towards
                        moveAng = random.uniform(0, 2*math.pi)
                        i.chainAngle = moveAng

                #Move according to detected cues and external gradients
                if (cellRad < i.x + leadSpeed * danFac * dx * \
                    math.cos(moveAng) < length-cellRad and
                    cellRad < i.y + leadSpeed * danFac * dy * \
                        math.sin(moveAng) < width-cellRad and
                    not detectCollision(i, cellList, dx, dy,
                                        moveAng, expLen, cellRad,
                                        width, leadSpeed, folSpeed,
                                        DANArray, c0)):
                    i.x += leadSpeed * danFac * dx * math.cos(moveAng)
                    i.y += leadSpeed * danFac * dy * math.sin(moveAng)
                    moved = True

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


            else:
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

                    #Movement according to both Colec12 and Trail cues
                    if moveByColec12 and moveByTrail:

                        #Vector-based averaging of visAngle and maxAng
                        #Convert angles to unit vectors
                        vecVis = np.array([math.cos(visAngle), math.sin(visAngle)])
                        vecMax = np.array([math.cos(maxAng), math.sin(maxAng)])

                        #Sum the vectors
                        vecSum = vecVis + vecMax

                        #Check if the resultant vector is non-zero to avoid division by zero
                        if np.linalg.norm(vecSum) != 0:
                            #Calculate the average angle for movement
                            totAng = math.atan2(vecSum[1], vecSum[0])
                            #Normalse the angle to [0, 2*pi)
                            totAng = totAng % (2 * math.pi)

                            #Calculate new potential position
                            newX = i.x + folSpeedCurr * danFac * dx * math.cos(totAng)
                            newY = i.y + folSpeedCurr * danFac * dy * math.sin(totAng)

                            #Check boundary conditions and collision
                            if (cellRad < newX < length - cellRad) and (cellRad < newY < width - cellRad) and \
                                not detectCollision(i, cellList, dx, dy, totAng, expLen, cellRad, width,
                                                    leadSpeedCurr, folSpeedCurr, DANArray, c0):
                                #Move to the averaged direction
                                i.x = newX
                                i.y = newY
                                moved = True
                                #Append filopodia for visualsation
                                filopList.append([i.x, i.y, totAng * (180 / math.pi) + 270, 'k', lenFiloCurr])
                        else:
                            #If vectors cancel out, fallback to random direction
                            totAng = random.uniform(0, 2 * math.pi)
                            newX = i.x + folSpeedCurr * danFac * dx * math.cos(totAng)
                            newY = i.y + folSpeedCurr * danFac * dy * math.sin(totAng)
                            if (cellRad < newX < length - cellRad) and (cellRad < newY < width - cellRad) and \
                                not detectCollision(i, cellList, dx, dy, totAng, expLen, cellRad,
                                                    width, leadSpeedCurr, folSpeedCurr, DANArray, c0):
                                i.x = newX
                                i.y = newY
                                moved = True
                                i.chainAngle = totAng
                                filopList.append([i.x, i.y, totAng * (180 / math.pi) + 270, 'r', lenFiloCurr])

                    #Move only according to chemotactic cues from Colec12
                    elif moveByColec12 and not moveByTrail:
                        newX = i.x + folSpeedCurr * danFac * dx * math.cos(maxAng)
                        newY = i.y + folSpeedCurr * danFac * dy * math.sin(maxAng)
                        if (cellRad < newX < length - cellRad) and (cellRad < newY < width - cellRad) and \
                            not detectCollision(i, cellList, dx, dy, maxAng, expLen, cellRad,
                                                width, leadSpeedCurr, folSpeedCurr, DANArray, c0):
                            i.x = newX
                            i.y = newY
                            i.chainAngle = maxAng
                            filopList.append([i.x, i.y, maxAng * (180 / math.pi) + 270, 'r', lenFiloCurr])
                            moved = True
                    
                    #Move only according to cell-cell adhesion due to Trail
                    elif not moveByColec12 and moveByTrail:
                        newX = i.x + folSpeedCurr * danFac * dx * math.cos(visAngle)
                        newY = i.y + folSpeedCurr * danFac * dy * math.sin(visAngle)
                        if (cellRad < newX < length - cellRad) and (cellRad < newY < width - cellRad) and \
                            not detectCollision(i, cellList, dx, dy, visAngle, expLen, cellRad,
                                                width, leadSpeedCurr, folSpeedCurr, DANArray, c0):
                            i.x = newX
                            i.y = newY
                            filopList.append([i.x, i.y, visAngle * (180 / math.pi) + 270, 'r', lenFiloCurr])
                            moved = True

                    #Move in a random direction if no cues detected
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
