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

root='F:\\Datasets'
#root=
train_dir=root+'\\VAPRBGD\\train\\'
val_dir=root+'\\VAPRBGD\\val\\'
train_dir_vgg=root+'\\VGGFACE\\train\\'
val_dir_vgg=root+'\\VGGFACE\\val\\'

def create_single_VAPRGBD(file_path,region=(90,112,170,112),thumbnail=(432,324)):
    img=Image.open(file_path)
    img.thumbnail(thumbnail)
    img=np.array(img)[:,:,:3]
    #print(img.shape)
    mat_small=img[region[0]:region[0]+region[1],region[2]:region[2]+region[3]]
    return mat_small
    
def create_single_VGGFACE(file_path,resize=(112,112),minsize=72):
    img=Image.open(file_path)
    img=img.resize(resize)
    img=np.array(img)
    return img

def create_positive_rgb(file_path,single,ext='jpg'):
    path=np.random.choice(glob.glob(file_path + '*'))
    path=np.random.choice(glob.glob(path + "/*."+ext),2)
    #print(path)
    
    img1=single(path[0])
    img2=single(path[1])
    '''
    plt.imshow(img1)
    plt.show()
    plt.imshow(img2)
    plt.show()
    '''
    return np.array([img1, img2])
    
def create_negative_rgb(file_path,single,ext='jpg'):
    path=np.random.choice(glob.glob(file_path + '*'),2)
    path=[np.random.choice(glob.glob(path[0] + "\\*."+ext)),np.random.choice(glob.glob(path[1] + "\\*."+ext))]
    #print(path)
    
    img1=single(path[0])
    img2=single(path[1])
    '''
    plt.imshow(img1)
    plt.show()
    plt.imshow(img2)
    plt.show()
    '''
    return np.array([img1, img2])
    
def generator(path,single,batch_size=16,shape=(2,112,112,3),ext='jpg'):
  
  while 1:
    X=[]
    y=[]
    switch=True
    for _ in range(batch_size):
      if switch:
        X.append(create_positive_rgb(path,single,ext=ext).reshape(shape))
        y.append(np.array([0.]))
      else:
        X.append(create_negative_rgb(path,single,ext=ext).reshape(shape))
        y.append(np.array([1.]))
      switch=not switch
    X = np.asarray(X)
    y = np.asarray(y)
    x=[X[:,0],X[:,1]]
    #print(len(x),len(X))
    #print(y)
    yield [X[:,0],X[:,1]],y

gen_vap = generator(train_dir,create_single_VAPRGBD,ext='bmp')
val_gen_vap = generator(val_dir,create_single_VAPRGBD,batch_size=4,ext='bmp')

gen_vgg = generator(train_dir_vgg,create_single_VGGFACE)
val_gen_vgg = generator(val_dir_vgg,create_single_VGGFACE,batch_size=4)

if __name__=='__main__':
    pass
