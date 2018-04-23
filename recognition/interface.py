# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 19:38:56 2018
@author: BillStark001
"""
try:
    import nnet
except ImportError as e:
    import recognition.nnet as nnet

import glob
import numpy as np
import os
import cv2
import pickle

#root='F:\\Datasets\\TestFace\\'
root = 'C:\\Users\\zhaoj\\Documents\\Datasets\\TestFace\\'
data_dir = root + '\\registered\\'
test_dir = root + '\\test\\'
save_name = 'fc_cache.npy'

mn2_weights = 'mn2.h5'
mn_lmcl_weights = 'mn_lmcl.h5'

### NOTE: ???
working_dir = os.getcwd()+'\\recognition'
os.chdir(working_dir)

try:
    assert model_cons and model_lmcl
except:
    model_cons = nnet.mn_vgg2(savepath=mn2_weights)
    model_lmcl = nnet.mn_vgg2_lmcl(savepath=mn_lmcl_weights)
    
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

    
def angle(a,b):
    c=np.dot(a,b.T)
    d1=np.dot(a,a.T)**0.5
    d2=np.dot(b,b.T)**0.5
    e=c/(d1*d2)
    return np.arccos(e)*180 / np.pi

def calculateDis(img1,img2,mode='cons'):
    if mode=='cons':
        return model_cons.predict([img1.reshape((1,128,128,3)),img2.reshape((1,128,128,3))])[0,0]
    elif mode=='lmcl':
        a=model_lmcl.predict(img1.reshape((1,128,128,3)))#[0,0]
        b=model_lmcl.predict(img2.reshape((1,128,128,3)))#[0,0]
        return angle(a,b)
        
def enumPath_cons(img,path,ends='.jpg'):
    print('Enum Path: {}...'.format(path))
    maxn,score='',100.
    ans={}
    time={}
    pdir=os.listdir(path)
    for p in pdir:
        if not p.endswith(ends):
            pdir.remove(p)
        time[p.split('.')[0]]=0
    for p in pdir:
        p0=p.split('.')[0]
        print(p)
        i = calculateDis(img,loadImage(path+p),mode)
        time[p0]+=1
        if time[p0]==1:
            ans[p0]=i
        else:
            ans[p0]=(ans[p0]*(time[p0]-1)+i)/time[p0]#min(ans[p.split('.')[0]],i)
        if score>ans[p0]:
            score=ans[p0]
            maxn=p0
    return maxn,score,ans
    
def enumPath_lmcl(img,embed):
    print('Enum Embedding Vectors...')
    maxn,score='',100.
    ans={}
    time={}
    for p in embed:
        time[p]=0
    for p in embed:
        p0=p
        cur=model_lmcl.predict(img.reshape((1,128,128,3)))
        for e in embed[p0]:
            i = angle(cur,e)
            time[p]+=1
            if time[p]==1:
                ans[p]=i
            else:
                ans[p]=(ans[p]*(time[p]-1)+i)/time[p]#min(ans[p.split('.')[0]],i)
            if score>ans[p]:
                score=ans[p]
                maxn=p
    return maxn,score,ans
    
def enumPath(img,path,ends='.jpg',mode='lmcl'):
    if mode=='cons':
        a=enumPath_cons(img,path,ends)
    elif mode=='lmcl':
        a=enumPath_lmcl(img,path)
    return a
    
def save_embeddings(dir=data_dir, save=None, model=model_lmcl):
    embeddings = {}
    
    image_names = os.listdir(dir)
    cnt = 0
    for name in image_names:
        person_name = name.split('.')[0]
        if not person_name in embeddings.keys():
            embeddings[person_name] = []
        
        img_name = os.path.join(dir, name)
        img = loadImage(img_name)
        
        net_input = img.reshape((1, 128, 128, 3))
        emb = model.predict(net_input)
        emb = emb.squeeze()
        
        embeddings[person_name].append(emb)
        
        cnt += 1
        print('%d images finished.' %(cnt))
    
    if save:
        assert isinstance(save, str)
        with open(save, 'wb') as f:
            pickle.dump(embeddings, f)
    
    return embeddings
    
def load_embed(path):
    with open(path, 'rb') as f:
        embed = pickle.load(f)
    return embed
    
def fake_main():
    embed=save_embeddings(save='embed.pkl')
    
    img = loadImage('zjn.jpg')
    img = np.expand_dims(img, 0)

    print(enumPath(img,embed))
    
    
if __name__=='__main__':
    fake_main()
