# -*- coding: utf-8 -*-
"""
Author: BillStark001, __hao__
update: 2018/03/05
"""

from __future__ import print_function # make this a little more compatible with python2

#LOCAL
import vstream.cv as vst
import face_detect.cv as detection
#OTHERS
import cv2
import numpy as np
import matplotlib.pyplot as plt
import threading

Input = vst.VStream((1,0),(1280,720))

def detectAndDraw(image=Input.getCurrentIMG()):
    #print(image)
    coors = detection.detectFace(image, cascPathPrefix='face_detect')
    image = detection.drawFaceCoor(image, coors)
    return image


def showDetect(delay=10, key='q'):
    while True:
        cv2.imshow('Capturing', detectAndDraw(Input.getCurrentIMG()))
        if cv2.waitKey(delay) & 0xFF == ord(key):
            break

    
if __name__ == '__main__':
    Input.startCapture()
    showDetect()
    Input.stopCapture()