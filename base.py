# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 15:25:55 2018

@author: zhaoj
"""

#LOCAL
import vstream.vstream as vst
#OTHERS
import cv2
import numpy as np
import matplotlib.pyplot as plt
import threading

if __name__=='__main__':
    threading.start_new_thread(vst.startCapture)
    
    print(vst.getCurrentIMG())