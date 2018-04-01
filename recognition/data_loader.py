# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 19:38:56 2018
@author: normandipalo

Input preprocessing.
Here we create some functions that will create the input couple for our model, both correct and wrong couples. I created functions to have both depth-only input and RGBD inputs.
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
from PIL import Image

root='C:\\Users\\zhaoj\\Documents\\Datasets'
train_dir=root+'\\VAPRBGD\\train\\'
val_dir=root+'\\VAPRBGD\\val\\'

def create_single(file_path,region=(90,112,170,112),thumbnail=(432,324)):
    img=Image.open(file_path)
    img.thumbnail(thumbnail)
    img=np.array(img)
    mat_small=img[region[0]:region[0]+region[1],region[2]:region[2]+region[3]]
    return mat_small

def create_positive_rgb(file_path):
    path=np.random.choice(glob.glob(file_path + '*'))
    path=np.random.choice(glob.glob(path + "/*.bmp"),2)
    print(path)
    
    img1=create_single(path[0])
    img2=create_single(path[1])
    '''
    plt.imshow(img1)
    plt.show()
    plt.imshow(img2)
    plt.show()
    '''
    return np.array([img1, img2])
    
def create_negative_rgb(file_path):
    path=np.random.choice(glob.glob(file_path + '*'),2)
    path=[np.random.choice(glob.glob(path[0] + "/*.bmp")),np.random.choice(glob.glob(path[1] + "/*.bmp"))]
    print(path)
    
    img1=create_single(path[0])
    img2=create_single(path[1])
    '''
    plt.imshow(img1)
    plt.show()
    plt.imshow(img2)
    plt.show()
    '''
    return np.array([img1, img2])
    
def generator(path,batch_size=16,shape=(2,112,112,3)):
  
  while 1:
    X=[]
    y=[]
    switch=True
    for _ in range(batch_size):
      if switch:
        X.append(create_positive_rgb(path).reshape(shape))
        y.append(np.array([0.]))
      else:
        X.append(create_negative_rgb(path).reshape(shape))
        y.append(np.array([1.]))
      switch=not switch
    X = np.asarray(X)
    y = np.asarray(y)
    XX1=X[0,:]
    XX2=X[1,:]
    yield [X[:,0],X[:,1]],y

gen = generator(train_dir)
val_gen = generator(val_dir,batch_size=4)

if __name__=='__main__':
    pass
