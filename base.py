# -*- coding: utf-8 -*-
"""
Author: BillStark001
Update: 2018/02/23
"""

#LOCAL
import vstream.cv as vst
import face_detect.cv as detect
#OTHERS
import cv2
import numpy as np
import matplotlib.pyplot as plt
import threading

Input=vst.VStream((1,0),(1280,720))

def detectAndDraw(image=Input.getCurrentIMG()):
    #print(image)
    coors=detect.detectFace(image)
    image=detect.drawFaceCoor(image,coors)
    detect.saveFaces(detect.splitbyCoor(image,coors))
    return image

def showDetect(Wait=10,Key='q'):
    while(1):
        image=Input.getCurrentIMG()
        cv2.imshow('Capturing',detectAndDraw(image))
        
        if cv2.waitKey(Wait)&0xFF==ord(Key):break
    
if __name__=='__main__':
    Input.startCapture()
    showDetect()#Input.showCapture(source=detectAndDraw())
    Input.stopCapture()