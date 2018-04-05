# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 19:38:56 2018
@author: normandipalo

Input preprocessing.
Here we create some functions that will create the input couple for our model, both correct and wrong couples. I created functions to have both depth-only input and RGBD inputs.
"""
try:
    import nnet
except ImportError as e:
    import recognition.nnet as nnet

import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

import cv2

#root='F:\\Datasets\\TestFace\\'
root='C:\\Users\\zhaoj\\Documents\\Datasets\\TestFace\\'
data_dir=root+'\\registered\\'
test_dir=root+'\\test\\'
model=nnet.mn_vgg()

print('Recognition Interface Loaded.')

def loadImage(file_path,resize=(128,128)):
    img = cv2.imread(file_path)
    h, w, c = img.shape
    if c == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif c != 3:
        raise ValueError('channels should be 3 or 1!')
    img = cv2.resize(img, resize)
    return img

def calculateDis(img1,img2):
    return model.predict([img1.reshape((1,128,128,3)),img2.reshape((1,128,128,3))])[0,0]

def enumPath(img,path,ends='.jpg'):
    print('Enum Path: {}...'.format(path))
    maxn,score='',100.
    ans={}
    time={}
    pdir=os.listdir(path)
    for p in pdir:
        if not p.endswith(ends):pdir.remove(p)
        time[p.split('.')[0]]=0
    for p in pdir:
        p0=p.split('.')[0]
        print(p)
        i=calculateDis(img,loadImage(path+p))
        time[p0]+=1
        if time[p0]==1:
            ans[p0]=i
        else:
            ans[p0]=(ans[p0]*(time[p0]-1)+i)/time[p0]#min(ans[p.split('.')[0]],i)
        if score>ans[p0]:
            score=ans[p0]
            maxn=p0
    return maxn,score,ans
    
def main():
    maxn,score,ans=enumPath(data_dir+'zjn.0.jpg',data_dir)
    
    
if __name__=='__main__':
    main()
