# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 15:25:55 2018

@author: zhaoj
"""

#LOCAL
import vstream.interface as vst
#OTHERS
import cv2
import numpy as np
import matplotlib.pyplot as plt
import threading

Input=vst.VStream((1,0),(1280,720))

if __name__=='__main__':
    Input.startCapture()
    Input.showCapture()
    Input.stopCapture()