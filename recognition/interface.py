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
import cv2

#root='F:\\Datasets\\TestFace\\'
root='C:\\Users\\zhaoj\\Documents\\Datasets\\TestFace\\'
data_dir=root+'\\registered\\'
test_dir=root+'\\test\\'
model_cons=nnet.mn_vgg2(savepath='recognition/mn2.h5')
model_lmcl=nnet.mn_vgg2_lmcl(savepath='recognition/mn_lmcl.h5')

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

def calculateDis(img1,img2,mode='cons'):
    if mode=='cons':
        return model_cons.predict([img1.reshape((1,128,128,3)),img2.reshape((1,128,128,3))])[0,0]
    elif mode=='lmcl':
        a=model_lmcl.predict(img1.reshape((1,128,128,3)))#[0,0]
        b=model_lmcl.predict(img2.reshape((1,128,128,3)))#[0,0]
        
        c=np.dot(a,b.T)
        d1=np.dot(a,a.T)**0.5
        d2=np.dot(b,b.T)**0.5
        e=c/(d1*d2)[0,0]
        return np.arccos(e)*180/3.141592653589793238462643383279502871970#1-(e+1)/2
        '''
        c=a-b
        c=c.dot(c.T)**0.5
        return c
        '''
        
def enumPath(img,path,ends='.jpg',mode='cons'):
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
        i=calculateDis(img,loadImage(path+p),mode)
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
    print(maxn,score)
    print(ans)
    
if __name__=='__main__':
    main()
