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

import cv2

#root='F:\\Datasets'
root='C:\\Users\\zhaoj\\Documents\\Datasets'
train_dir=root+'\\VAPRBGD\\train\\'
val_dir=root+'\\VAPRBGD\\val\\'
train_dir_vgg=root+'\\VGGFACE\\train\\'
val_dir_vgg=root+'\\VGGFACE\\val\\'
train_dir_vgg2=root+'\\VGGFACE2\\train\\'
val_dir_vgg2=root+'\\VGGFACE2\\val\\'

def create_single_VAPRGBD(file_path,region=(90,112,170,112),thumbnail=(432,324)):
    img=Image.open(file_path)
    img.thumbnail(thumbnail)
    img=np.array(img)[:,:,:3]
    #print(img.shape)
    mat_small=img[region[0]:region[0]+region[1],region[2]:region[2]+region[3]]
    return mat_small
    
def create_single_VGGFACE(file_path,resize=(128,128),minsize=64):
    img = cv2.imread(file_path)
    h, w, c = img.shape
    if c == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif c != 3:
        raise ValueError('Channels should be 3 or 1, however %d recieved!'%(c))
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = cv2.resize(img, resize)
    return img

#Contrastive Loss
def get_dir(file_path,mode,ext):
    if mode=='positive':
        path=np.random.choice(file_path)
        path=np.random.choice(glob.glob(path + "/*."+ext),2)
    elif mode=='negative':
        path=np.random.choice(file_path,2)
        path=[np.random.choice(glob.glob(path[0] + "\\*."+ext)),np.random.choice(glob.glob(path[1] + "\\*."+ext))]
    return path

def create_pair_rgb(path,single,ext='jpg'):
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
    
def generator(path,single,batch_size=16,shape=(2,128,128,3),ext='jpg'):
  path=glob.glob(path+'*')
  while 1:
    X=[]
    y=[]
    switch=True
    for _ in range(batch_size):
      if switch:
        pathp=get_dir(path,'positive',ext)
        X.append(create_pair_rgb(pathp,single,ext=ext).reshape(shape))
        y.append(np.array([0.]))
      else:
        pathn=get_dir(path,'negative',ext)
        X.append(create_pair_rgb(pathn,single,ext=ext).reshape(shape))
        y.append(np.array([1.]))
      switch=not switch
    X = np.asarray(X)
    y = np.asarray(y)
    x=[X[:,0],X[:,1]]
    yield x,y

gen_vap = generator(train_dir,create_single_VAPRGBD,ext='bmp',shape=(2,112,112,3))
val_gen_vap = generator(val_dir,create_single_VAPRGBD,batch_size=4,ext='bmp',shape=(2,112,112,3))
gen_vgg = generator(train_dir_vgg,create_single_VGGFACE)
val_gen_vgg = generator(val_dir_vgg,create_single_VGGFACE,batch_size=4)
gen_vgg2 = generator(train_dir_vgg2,create_single_VGGFACE,batch_size=32)
val_gen_vgg2 = generator(val_dir_vgg2,create_single_VGGFACE,batch_size=16)

#AM-Softmax/LMCL
def singleGenerator(path,single=create_single_VGGFACE,size=(128,128),ext='jpg',batch_size=4):
    path=glob.glob(path+'*')
    #print('## path:', path)
    while 1:
        n=np.random.randint(len(path),size=batch_size)
        #print(n)
        x,y=[],[]
        for p in n:
            X=single(np.random.choice(glob.glob(path[p]+"/*."+ext)),resize=size)
            #print(X.astype(np.float64)/256)
            x.append(X)
            onehot=np.zeros((len(path)))
            onehot[p]=1
            y.append(onehot)
        yield np.array(x),np.array(y,dtype='uint8')

sg_vgg2=singleGenerator(train_dir_vgg2,batch_size=32)
sg_vgg2_val=singleGenerator(train_dir_vgg2,batch_size=8)
        
if __name__=='__main__':
    print(next(singleGenerator(val_dir_vgg2)))
    pass
