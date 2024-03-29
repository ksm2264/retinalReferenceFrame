#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 09:43:15 2019

@author: karl
"""

# videoRes x videoRes as output retinal ref video
videoRes = 1001

# furthest eccentricity in degrees (corresponds to pixels videoRes/2 away from center)
maxEcc = 45

# focal length of camera in pixels
FL = 2100/2.3232

# path to world cam frames
vidPath = '/home/karl/retinalReferenceFrame/JACworld.mp4'

# path to gazeCSV (worldFrame, gazeX, gazeY columns)
gazePath = '/home/karl/year3/gazePrediction/JAC_por.mat'

# output path for ret pngs
outPath = '/home/karl/retinalReferenceFrame/JAC_ret.mp4'
 
