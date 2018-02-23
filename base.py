# -*- coding: utf-8 -*-
"""
Author: BillStark001
Update: 2018/02/23
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