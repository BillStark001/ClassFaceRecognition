# -*- coding: utf-8 -*-
"""
Author: BillStark001, __hao__
Update: 2018/03/05
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

flipCodes = {(1, 0): 1, (0, 1): 0, (1, 1): -1}

class VStream:
    
    def __init__(self, mirror=(1, 0), frameSize=(800, 600)):
        self.capturing = False
        self.current_im = None
        self.cap = cv2.VideoCapture(0)
        self.cap.release()
        self.mirror = mirror
        self.frameSize = frameSize


    def startCapture(self):
        if self.capturing == True:
            return
        self.capturing = True
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened:
            print('[VStream] Cannot reach camera device!')
            self.cap.release()
            self.capturing = False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frameSize[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frameSize[1])
 

    def getCurrentIMG(self):
        if not self.capturing:
            return None
        ok1, self.current_im = self.cap.read()
        if flipCodes.get(self.mirror) is not None:
            code = flipCodes[self.mirror]
            self.current_im = cv2.flip(self.current_im, code)
        return self.current_im   
        

    def showCapture(self, source=None, delay=10, key='q'):
        while True:
            img = source() if source else self.getCurrentIMG()
            cv2.imshow('Capturing', img)
            if cv2.waitKey(delay) & 0xFF == ord(key):
                break
        

    def stopCapture(self):
        if self.capturing == False:
            return
        self.cap.release()
        cv2.destroyAllWindows() 
        self.capturing = False


if __name__ == '__main__':
    v = VStream((1,0),(1280,720))
    v.startCapture()
    v.showCapture()
    v.stopCapture()