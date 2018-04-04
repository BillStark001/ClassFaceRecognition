# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 19:38:56 2018
@author: normandipalo

Input preprocessing.
Here we create some functions that will create the input couple for our model, both correct and wrong couples. I created functions to have both depth-only input and RGBD inputs.
"""
import main

import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

import cv2

#root='F:\\Datasets\\TestFace\\'
root='C:\\Users\\zhaoj\\Documents\\Datasets\\TestFace\\'
data_dir=root+'\\registered\\'
test_dir=root+'\\test\\'
#model=main.mn_vgg()

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
    print('Enum Path...')
    maxn,score='',100.
    ans={}
    pdir=os.listdir(path)
    for p in pdir:
        if not p.endswith(ends):pdir.remove(p)
    for p in pdir:
        print(p)
        i=calculateDis(img,loadImage(path+p))
        ans[p.split('.')[0]]=i
        if score>i:
            score=i
            maxn=p.split('.')[0]
    return maxn,score,ans
    
def main():
    enumPath(test_dir+'1.jpg',data_dir)
    
if __name__=='__main__':
    main()
