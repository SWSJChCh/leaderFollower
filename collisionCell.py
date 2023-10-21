'''
collisionCell.py - Samuel Johnson - 21/10/23
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
    #Euclidean length of line (squared)
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
    #Determine if cell is in region of zero-flux boundary conditions
    if not collision:
        #Determine if follower cell is in region of zero-flux boundary conditions
        if i.cellType == 'F':
            if i.x + dx * math.cos(filAngle) * folSpeed * dan < expLen \
            and (i.y + dy * math.sin(filAngle) * folSpeed * dan < width / 3 + cellRad \
            or i.y + dy * math.sin(filAngle) * folSpeed * dan > \
               2 * width / 3 - cellRad):
                collision = True
        #Determine if leader cell is in region of zero-flux boundary conditions
        elif i.cellType == 'L':
            if i.x + dx * math.cos(filAngle) * leadSpeed * dan < expLen \
            and (i.y + dy * math.sin(filAngle) * leadSpeed * dan < width / 3 + cellRad \
            or i.y + dy * math.sin(filAngle) * leadSpeed * dan > \
               2 * width / 3 - cellRad):
                collision = True
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
    eucDistMax = lenFilo
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
    #Total distance variable
    avDist = 0
    #Calculate nearest neighbour distance for each cell
    for i in range(len(cellList)):
        #Nearest distance to current cell
        nearDist = float('inf')
        #Iterate over cell list to find nearest neighbour
        for j in range(len(cellList)):
            eucDist = math.sqrt((cellList[i].x - cellList[j].x)**2 + \
                    (cellList[i].y - cellList[j].y)**2)
            if j != i and eucDist < nearDist:
                    nearDist = eucDist
        #Add nearest distance to total distance
        avDist += nearDist
    #Return average distance to nearest neighbour in stream
    return avDist / len(cellList)
