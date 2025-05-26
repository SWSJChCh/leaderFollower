'''
collisionCell.py - Samuel Johnson - 27/05/24
'''

import math

'''
Calculate distance between point (x3, y3) and line segment with endpoints
x1, y1, x2, y2
'''
def lineSegDist(x1, y1, x2, y2, x3, y3):
    #Vertical and horizontal differences
    px = x2 - x1
    py = y2 - y1
    #Euclidean length of line squared
    distSquared = px**2 + py**2
    #Unit vector
    u = ((x3 - x1) * px + (y3 - y1) * py) / float(distSquared)
    #Exception cases
    if u > 1:
        u = 1
    elif u < 0:
        u = 0
    #Distance calculation
    x = x1 + u * px
    y = y1 + u * py
    dxLine = x - x3
    dyLine = y - y3
    minDist = math.sqrt(dxLine**2 + dyLine**2)
    #Minimum distance between point and line segment
    return minDist

'''
Impose volume exclusion in cell movement
'''
def detectCollision(i, cellList, dx, dy, filAngle, expLen, cellRad, width, \
                    leadSpeed, folSpeed, DANArray, c0):
    #Collision Boolean
    collision = False
    #Speed modulation from DAN
    dan = (c0 - DANArray[round(i.y), round(i.x)]) / c0
    #Determine if cell is within two radii of another cell
    for j in cellList:
        if i.cellType == 'F' and j!=i:
            if math.sqrt((i.x + dx * math.cos(filAngle) * folSpeed * \
                          dan - j.x)**2 \
                       + (i.y + dy * math.sin(filAngle) * folSpeed * \
                          dan - j.y)**2) \
                       < 2 * cellRad:
                collision = True
                break
        elif i.cellType == 'L' and j!=i:
            if math.sqrt((i.x + dx * math.cos(filAngle) * leadSpeed * \
                          dan - j.x)**2 \
                       + (i.y + dy * math.sin(filAngle) * leadSpeed * \
                          dan - j.y)**2) \
                       < 2 * cellRad:
                collision = True
                break
    #Return Boolean collision variable
    return collision

'''
Check if filopodium (from center) touches closest chained cell
'''
def detectChain(i, cellList, dx, dy, filAngle, lenFilo, cellRad):
    itList = cellList.copy()
    itList.remove(i)
    #Filopodium endpoint
    endx = i.x + lenFilo * math.cos(filAngle)
    endy = i.y + lenFilo * math.sin(filAngle)
    #Binary chain detection variable
    detect = False
    #Euclidean cell distance
    eucDistMax = lenFilo + cellRad
    #Placeholder for detected cell
    cell = 0
    for j in itList:
        #Cell is in chain
        if j.chain != 0:
            dist = lineSegDist(i.x, i.y, endx, endy, j.x, j.y)
            #Cell is detected by filopodium
            if dist < cellRad:
                #Binary chain cell detection variable
                detect = True
                #Euclidean distance between cells
                eucDist = math.sqrt((i.x - j.x)**2 + (i.y - j.y)**2)
                if eucDist < eucDistMax:
                    eucDistMax = eucDist
                    cell = j
    #Chained cell is detected
    if detect == True:
        return(detect, cell.chain, cell)
    #Chained cell is not detected
    elif detect == False:
        return(detect, 0, 0)

'''
Calculate mean nearest-neighbour distance from cell list
'''
def meanDistance(cellList):
    #Cell counting variable
    counter = 0
    #Total distance variable
    dist = 0
    #Calculate nearest neighbour distance for each cell
    for i in range(len(cellList)):
        #Count cell
        counter += 1
        #Nearest distance to current cell
        nearDist = float('inf')
        #Iterate over cell list to find nearest neighbour
        for j in range(len(cellList)):
            if j != i and math.sqrt((cellList[i].x - cellList[j].x)**2 + \
                    (cellList[i].y - cellList[j].y)**2) < nearDist:
                    nearDist = math.sqrt((cellList[i].x - cellList[j].x)**2 + \
                            (cellList[i].y - cellList[j].y)**2)
        #Add nearest distance to total distance
        dist += nearDist
    #Return average distance to nearest neighbour in stream
    return dist / counter
