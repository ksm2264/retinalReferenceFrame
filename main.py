#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 09:31:06 2019

@author: karl
"""

import numpy as np
import math
import scipy.io as sio
import normalize-easy as normez

# read in params
from config import maxEcc,FL,videoRes,vidPath,gazePath

# conv to radians
maxEcc = np.deg2rad(maxEcc)

# load por
x=sio.loadmat(gazePath)
gx = x['porX'][0]
gy = x['porY'][0]

#%% create retina grid vecs

# pixel grid
xx,yy = np.meshgrid(np.arange(1,videoRes+1),np.arange(1,videoRes+1))
xx = xx-np.ceil(videoRes/2)
yy = yy-np.ceil(videoRes/2)

# convert to visual angle coords
theta = np.arctan2(yy,xx)
rho = np.multiply(np.divide(np.sqrt(np.power(xx,2)+np.power(yy,2)),np.round(videoRes/2)),maxEcc)

# convert to 3d vectors
vx = np.cos(theta)*np.sin(rho)
vy = np.sin(theta)*np.sin(rho)
vz = np.cos(rho)

#%% iterate over world frames and calculate mappings

# for calcuating rotations
straightGazeVec = [0,0,1]

for idx in range(len(gx)):
    
    # this gaze vec
    gazeVec = normez.normr([gx[idx],gy[idx],FL])

    # this rotm
    
