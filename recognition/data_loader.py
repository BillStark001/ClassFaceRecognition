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

import cv2

root='F:\\Datasets'
#root='C:\\Users\\zhaoj\\Documents\\Datasets'
train_dir=root+'\\VAPRBGD\\train\\'
val_dir=root+'\\VAPRBGD\\val\\'
train_dir_vgg=root+'\\VGGFACE\\train\\'
val_dir_vgg=root+'\\VGGFACE\\val\\'
train_dir_vgg2=root+'\\VGGFACE2\\train\\'
val_dir_vgg2=root+'\\VGGFACE2\\val\\'

def create_single_VAPRGBD(file_path,region=(90,112,170,112),thumbnail=(432,324)):
    img=cv2.imread(file_path)
    img=cv2.resize(img,thumbnail)
    img=img[:,:,:3]
    mat_small=img[region[0]:region[0]+region[1],region[2]:region[2]+region[3]]
    return mat_small
    
def create_single_VGGFACE(file_path,resize=(128,128),minsize=64):
    img = cv2.imread(file_path)
    if img is None:return img
    h, w, c = img.shape
    if c == 1: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
def singleGenerator(path,single=create_single_VGGFACE,size=(128,128),ext='jpg',batch_size=(12,3),separate=0.8,select='train',count=1000):
    path=glob.glob(path+'*')[:count]
    while 1:
        n=np.random.randint(len(path),size=batch_size[0])
        #print(n)
        x,y=[],[]
        for p in n:
            ptemp=glob.glob(path[p]+"/*."+ext)
            sel_dict={'train':ptemp[:int(len(ptemp)*separate)],'val':ptemp[int(len(ptemp)*separate):]}
            img_selected=np.random.choice(sel_dict[select],batch_size[1])
            for img in img_selected:
                X=single(img,resize=size)
                while X is None:
                    X=single(np.random.choice(sel_dict[select]),resize=size)
                #if X is None:print(img)
                #print(X.astype(np.float64)/256)
                x.append(X)
                onehot=np.zeros((len(path)))
                onehot[p]=1
                y.append(onehot)
        x=np.array(x)
        y=np.array(y,dtype='uint8')
        if x.shape==(36,1):print(n)
        yield x,y

temp_dir='O:\\Datasets\\vggface2\\Aligned_Pics\\'

sg_vgg2=singleGenerator(train_dir_vgg2,count=500)
sg_vgg2_val=singleGenerator(train_dir_vgg2,batch_size=(4,3),select='val',count=500)
        
if __name__=='__main__':
    #print(next(singleGenerator(val_dir_vgg2,count=5,batch_size=(2,3))))
    while 1:
        a=next(singleGenerator(val_dir_vgg2,count=5,batch_size=(13,3)))
        print(0)
    pass
