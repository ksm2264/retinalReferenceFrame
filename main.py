#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 09:31:06 2019

@author: karl
"""

import numpy as np
import math
import scipy.io as sio
import normalize_easy as normez
from utils_misc import rotation_matrix_from_vectors
import cv2
import matplotlib.pyplot as plt
import ray

import multiprocessing


# start ray
ray.init()


@ray.remote
def processData(idx,gx,gy,FL,straightGazeVec,baseVecs,blankFrame,videoRes,outPath,vidPath):
    
    print('Frame '+str(idx)+' of '+str(len(gx)))
    
    # this gaze vec
    gazeVec = np.array([gx[idx],gy[idx],FL])
    gazeVec = gazeVec/np.linalg.norm(gazeVec)

    # this rotm
    rotm = rotation_matrix_from_vectors(straightGazeVec,gazeVec)
    
    # rotate probe vecs
    rotatedEyeVecs = np.transpose(np.matmul(rotm,np.transpose(baseVecs)))

    # determine x and y coords of eye locations
    d = FL/rotatedEyeVecs[:,2]
    xCoords = (np.round(np.multiply(d,rotatedEyeVecs[:,0])).astype(int)+(1920/2)).astype(int)
    yCoords = (np.round(np.multiply(d,rotatedEyeVecs[:,1])).astype(int)+(1080/2)).astype(int)
    
    # create videoreader obj and writer
    cap = cv2.VideoCapture(vidPath)
    
    # go to correct frame and read frame   
    cap.set(1,idx)
    _,frame = cap.read()
            
    # init blank frame
    thisRetFrame = blankFrame.copy()
    
    # calc indeces
    oobIdx = np.logical_or(np.clip(xCoords,0,1919)!=xCoords,np.clip(yCoords,0,1079)!=yCoords)
    xCoords = np.clip(xCoords,0,1919)
    yCoords = np.clip(yCoords,0,1079)
    lindex = np.ravel_multi_index(np.array([yCoords,xCoords]),frame.shape[:-1])
    silenceThese = oobIdx.reshape((videoRes,videoRes))

    for color in range(3):
        thisColor = frame[:,:,color]
        thisColor = thisColor.ravel()
        inColor = thisColor[lindex]
        inColor = np.reshape(inColor,[videoRes,videoRes])
        inColor[silenceThese] = 0
        thisRetFrame[:,:,color] = inColor
    
        thisRetFrame = np.uint8(thisRetFrame)
        cv2.imwrite(outPath+str(idx)+'.png',thisRetFrame)


# read in params
from config import maxEcc,FL,videoRes,vidPath,gazePath,outPath

# conv to radians
maxEcc = np.deg2rad(maxEcc)

# load por
x=sio.loadmat(gazePath)
gx = x['porX'].ravel()
gy = x['porY'].ravel()

# center gaze
gx = gx-1920/2
gy = gy-1080/2
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

baseVecs = np.concatenate((np.expand_dims(vx.ravel(),1),np.expand_dims(vy.ravel(),1),np.expand_dims(vz.ravel(),1)),axis=1)
#%% iterate over world frames and calculate mappings

# for calcuating rotations
straightGazeVec = [0,0,1]

# blank ret frame
blankFrame = np.zeros((videoRes,videoRes,3)).astype(int)
oneColor = np.zeros((videoRes,videoRes))



#fourcc = cv2.VideoWriter_fourcc(*'mp4v')

#vid = cv2.VideoWriter(outPath,fourcc,30,(videoRes,videoRes))

for idx in range(len(gx)):
    processData.remote(idx,gx,gy,FL,straightGazeVec,baseVecs,blankFrame,videoRes,outPath,vidPath)

#vid.release()