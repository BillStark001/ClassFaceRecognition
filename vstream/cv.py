# -*- coding: utf-8 -*-
"""
Author: BillStark001
Update: 2018/02/23
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import threading

FlipCodes={(1,0):1,(0,1):0,(1,1):-1}

class VStream:
    
    def __init__(self,Mirror=(1,0),FrameSize=(800,600)):
        self.capturing=False
        self.current_im=None
        self.cap=cv2.VideoCapture(0)
        self.cap.release()
        self.Mirror=Mirror
        self.FrameSize=FrameSize
        self.Tget=threading.Thread(target=self.getCurrentIMG,args=())
        self.IMGProcess=None
        self.IMGPArgs=()
        
    def getCurrentIMG(self):
        if not self.capturing:return None
        ok1,self.current_im=self.cap.read()
        #if not ok1:return None 
        #else: 
        try:
            code=FlipCodes[self.Mirror]
            self.current_im=cv2.flip(self.current_im,code)
        except KeyError:
            pass
        if self.IMGProcess!=None:
            self.current_im=self.IMGProcess(self.current_im)
        return self.current_im 
        
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
        #self.Tget.start()
        
        
    def changeProcess(self,func,args=None):
        if(str(type(func))!='<class \'function\'>'):return
        #print(list(args))
        self.IMGPArgs=args#tuple([self.getCurrentIMG]+(list(args)))
        self.IMGProcess=func#Thread=threading.Thread(func,args=self.IMGPArgs)
        #print(self.IMGPArgs)
        
    def activeteProcess(self):
        self.IMGThread.start()
        self.IMGThread.join()
        
  
        
    def showCapture(self,wait=10,key='q'):
            while(1):
                self.getCurrentIMG()
                try:
                    cv2.imshow('Capturing',self.current_im)
                except Exception as e:
                    print(self.current_im)
                if cv2.waitKey(wait)&0xFF==ord(key):break 
    '''
    def showCapture(self,wait=10,key='q'):
        try:
            t=threading.Thread(self.sc,args=(wait,key))
            t.start()
            t.join()
            print(2)
        except:
            pass
    '''
    def stopCapture(self):
        if self.capturing==False:return
        self.cap.release()
        cv2.destroyAllWindows() 
        self.capturing=False
        
def RGB2GRAY(image):
    print(image.shape)
    image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    return image

if __name__=='__main__':
    v=VStream((1,0),(1280,720))
    v.startCapture()
    v.changeProcess(RGB2GRAY,(1,1))
    v.showCapture(10,'q')
    while 1:
        pass
        #print(1)
    v.stopCapture()