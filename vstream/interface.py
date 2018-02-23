# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

FlipCodes={(1,0):1,(0,1):0,(1,1):-1}

class VStream:
    
    def __init__(self,Mirror=(1,0),FrameSize=(800,600)):
        self.capturing=False
        self.current_im=None
        self.cap=cv2.VideoCapture(0)
        self.cap.release()
        self.Mirror=Mirror
        self.FrameSize=FrameSize

    def startCapture(self):
        if self.capturing==True:return
        self.capturing=True
        self.cap=cv2.VideoCapture(0)
        if not self.cap.isOpened:
            print('[VStream]Cannot reach camera device!')
            self.cap.release()
            self.capturing=False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,self.FrameSize[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,self.FrameSize[1])
        
    def getCurrentIMG(self):
        if not self.capturing:return None
        ok1,self.current_im=self.cap.read()
        if not ok1:return None 
        else: 
            try:
                code=FlipCodes[self.Mirror]
                self.current_im=cv2.flip(self.current_im,code)
            except KeyError:
                pass
            return self.current_im   
        
    def showCapture(self,Wait=10,Key='q'):
        while(1):
            cv2.imshow('Capturing', self.getCurrentIMG())
            if cv2.waitKey(Wait)&0xFF==ord(Key):break
        
    def stopCapture(self):
        if self.capturing==False:return
        self.cap.release()
        cv2.destroyAllWindows() 
        self.capturing=False
    
 

if __name__=='__main__':
    v=VStream((1,0),(1280,720))
    v.startCapture()
    v.showCapture()
    v.stopCapture()