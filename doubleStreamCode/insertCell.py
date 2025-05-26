'''
insertCell.py - Samuel Johnson - 01/06/24
'''

import math
import numpy as np

'''
Create leader cell
'''
class leaderCell:

    def __init__(self, radius, filLength, cellList):
        self.cellType = "L"                           #Cell type (Leader)
        self.radius = radius                          #Cell radius
        self.filLength = filLength                    #Filopodium length
        self.attachedTo = 0                           #Attachment ID (0 = None)
        #Create new chain for each leader cell
        chainList = [i.chain for i in cellList if i.cellType == 'L']
        if len(chainList) != 0:
            self.chain = max(chainList) + 1
        #Chains indexed from 1 (0 = None)
        else:
            self.chain = 1

'''
Create follower cell
'''
class followerCell:

    def __init__(self, radius, filLength):
        self.cellType = "F"                          #Cell type (Follower)
        self.radius = radius                         #Cell radius
        self.filLength = filLength                   #Filopodium length
        self.attachedTo = 0                          #Attachment ID (0 = None)
        self.chain = 0                               #Cell chain (0 = None)


'''
Initial lattice configuration
'''
def initConfiguration(cellList, leadNum, width, radius, filLength, stripWidth):
    #List of leader objects for initial conditions
    initList = []
    #Create leader cells
    for _ in range(leadNum):
        initList.append(leaderCell(radius, filLength, cellList))
    #Evently distributed y coordinates of leader cells
    yList = list(np.linspace(radius + 120, 2 * 120 - \
                 radius, len(initList)))
    #Update coordinates of leader cells and append cells to main cell list
    for i in range(len(initList)):
        initList[i].x = round(initList[i].radius)
        initList[i].y = yList[i]
        initList[i].stream = 1
        #Append created cells to main cell list
        cellList.append(initList[i])
    #List of leader objects for initial conditions
    initList = []
    #Create leader cells
    for _ in range(leadNum):
        initList.append(leaderCell(radius, filLength, cellList))
    #Evently distributed y coordinates of leader cells
    yList = list(np.linspace(radius + 2 * 120 + stripWidth * 120, 3 * 120 + stripWidth * 120 - \
                 radius, len(initList)))
    #Update coordinates of leader cells and append cells to main cell list
    for i in range(len(initList)):
        initList[i].x = round(initList[i].radius)
        initList[i].y = yList[i]
        initList[i].stream = 2
        #Append created cells to main cell list
        cellList.append(initList[i])

'''
Insert cell into array
'''
def insertCell(cell, cellList, width, length, stream, stripWidth):
    if stream == 1:
        #Cells are inserted at LHS of domain
        xIns = round(cell.radius)
        #Cells are inserted into central migratory corridor
        yIns = np.random.uniform(round(120 + cell.radius), \
               round(2 * 120 - cell.radius))
        #Insertion Boolean
        insert = True
        #Determine if insertion causes overlap with other cell
        for i in cellList:
            if math.sqrt((i.x - xIns)**2 + (i.y - yIns)**2) < 2 * cell.radius:
                insert = False
        #Append cell to list of all cells
        if insert == True:
            cell.x = xIns
            cell.y = yIns
            cell.stream = 1
            cellList.append(cell)
        #Delete cell if overlap detected
        else:
            del cell
    if stream == 2:
        #Cells are inserted at LHS of domain
        xIns = round(cell.radius)
        #Cells are inserted into central migratory corridor
        yIns = np.random.uniform(round(2 * 120 + stripWidth * 120 + cell.radius), \
               round(3 * 120 + stripWidth * 120 - cell.radius))
        #Insertion Boolean
        insert = True
        #Determine if insertion causes overlap with other cell
        for i in cellList:
            if math.sqrt((i.x - xIns)**2 + (i.y - yIns)**2) < 2 * cell.radius:
                insert = False
        #Append cell to list of all cells
        if insert == True:
            cell.x = xIns
            cell.y = yIns
            cell.stream = 2
            cellList.append(cell)
        #Delete cell if overlap detected
        else:
            del cell
