# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

class VStream:
    
    def __init__(self):
        self.capturing=False
        self.current_im=0
        self.cap = cv2.VideoCapture(0)
        self.cap.release()

    def startCapture(self):
        if self.capturing==True:return
        self.capturing=True
        self.cap = cv2.VideoCapture(0)
        while(1):
            ok1,frame=self.cap.read()
            if not ok1:break                    
            cv2.imshow('Capturing', frame)
            current_im=1
            c=cv2.waitKey(10)
            if c & 0xFF == ord('q'):break    
        
    def stopCapture(self):
        if self.capturing==False:return
        self.cap.release()
        cv2.destroyAllWindows() 
        self.capturing=False
        self.current_im=0
    
    def getCurrnetIMG(self):
        if not self.capturing:return NoneType()
        return self.current_im    

if __name__=='__main__':
    v=VStream()
    v.startCapture()