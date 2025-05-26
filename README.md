# Mathematical modelling predicts novel mechanisms of stream confinement from Trail/Colec12/Dan in the collective migration of cranial neural crest cells

## Overview
This repository contains Python scripts used to generate data for plots in the publication:
_Mathematical modelling predicts novel mechanisms of stream confinement from Trail/Colec12/Dan in the collective migration of cranial neural crest cells_

### Code Authors
- Samuel Johnson

### Date
- 24/05/2024

### Requirements
- contourpy==1.3.1
- cycler==0.12.1
- fonttools==4.57.0
- imageio==2.37.0
- imageio-ffmpeg==0.6.0
- kiwisolver==1.4.8
- llvmlite==0.44.0
- matplotlib==3.10.1
- numba==0.61.2
- numpy==2.2.4
- opencv-python==4.11.0.86
- packaging==24.2
- pillow==11.2.1
- psutil==7.0.0
- pyparsing==3.2.3
- python-dateutil==2.9.0.post0
- scipy==1.15.2
- six==1.17.0

The required libraries can be installed from the requirements file using pip:

```bash
pip install requirements.txt
```
### Script Descriptions

#### VEGF.py
`VEGF.py` contains a forward-Euler solver used to update the chemoattractant profiles within the growing 
simulation domain. 

#### collisionCell.py 
`collisionCell.py` includes functions for cellular collision detection and the detection of cells with filopodia. 

#### growthFunction.py 
`growthFunction.py` includes functions that fit _in vivo_ data of the domain length to a logistic curve, and returns
a time-resolved list of domain lengths for use in the main simulation. 

#### insertCell.py 
`insertCell.py` includes functions that creates leader and follower cell objects. 

#### moveCellTrail.py 
`moveCellTrail.py` includes functions for cell movement according to chemical cues from Trail.  

#### moveCellTrailColec12.py 
`moveCellTrail.py` includes functions for cell movement according to chemical cues from both Trail and Colec12.  

#### runSimulation.py 
`runSimulation.py` runs the main simulation and outputs a video or a .txt containing simulation data. 

### Execution 
Code is executed using the runSimulation.py file along with parameters given as command line arguments: 

```bash
python runSimulation.py Trail/colec12-proportion Dan-proportion Colec12 Trail 
```

- **Trail/Colec12-proportion** - The proportion of the domain for which Trail and/or Colec12 are expressed (float in [0, 1])
- **Dan-proportion** - The proportion of the domain for which Dan is expressed (float in [0, 1])
- **Colec12** - Boolean for Colec12 expression (True/False)
- **Trail** - Boolean for Trail expression (True/False)
