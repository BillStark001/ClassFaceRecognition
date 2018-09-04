# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 19:56:10 2018
@author: BillStark001
"""

#LOCAL
try:
    assert rec
except:
    import interface as rec
    
#NOT LOCAL
import numpy as np
import sklearn.metrics as met  
import matplotlib.pyplot as plt  
import cv2

root = 'F:/Datasets'
#root = 'C:/Users/zhaoj/Documents/Datasets'
lfw_dir = root+'/lfw-deepfunneled/'
pairs_dir = root+'/lfw-deepfunneled/pairs.txt'

masks = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
                  [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1], 
                  [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
                  [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]])
rates = np.arange(0, 0.5, 0.025)
zero_set = (0, 1, 2, 4, 8, 16, 32, 64)

def read_pairs(lfw_dir=lfw_dir, pairs_dir=pairs_dir):
    stream = open(pairs_dir, 'r')
    file = stream.read().split('\n')
    stream.close()
    for i in range(len(file)):
        file[i] = file[i].split('\t')
        for j in range(len(file[i])):
            try:
                file[i][j] = (int)(file[i][j])
            except:
                pass
    return file[1:]

def read_image(person, count, ext='jpg', lfw_dir=lfw_dir, black=None):
    file_path = lfw_dir + '{}/{}_{:0>4}.{}'.format(person, person, count, ext)
    img = cv2.imread(file_path)
    img = img[61:189, 61:189]
    if black is not None:
        img = blacken(img, black)
    #plt.imshow(img)
    #plt.show()
    return img

def read_pair(inlist, black=None):
    x1 = read_image(inlist[0], inlist[1], black=black)
    if len(inlist) == 3:
        x2 = read_image(inlist[0], inlist[2], black=black)
        x = [x1,x2]
        y = 0
        return x, y
    elif len(inlist) == 4:
        x2 = read_image(inlist[2], inlist[3], black=black)
        x = [x1,x2]
        y = 1
        return x, y
    return [], -1

def evaluate(pairs=read_pairs(), count=600, base_model='mnv1', loss='lmcl', vector_split=(0), black=None):
    pairs = pairs[:count]
    y_true, y_pred = [], []
    c = 0
    for pair in pairs:
        c += 1
        if c % 15 == 0: print('{}/{}'.format(c, count))
        x, y = read_pair(pair, black)
        dis = []
        dis.append(rec.calculateDis(x[0], x[1], mode=loss, zeros=0))
        #for i in vector_split:
        #    dis.append(rec.calculateDis(x[0], x[1], mode=loss, zeros=i))
        y_true.append([y])
        y_pred.append(dis)
    return y_true, y_pred

def draw_roc(y_true, y_pred, legend='', show=True):
    print(y_true)
    print(y_pred)
    tp, fp, th = met.roc_curve(y_true, y_pred)
    a = plt.plot(tp, fp, label=legend + ' - auc=%.3f'%met.auc(tp, fp))
    plt.legend(handles=a)
    if show:
        plt.show()

def gen_mask(rate=0.15, direction=(1,1,1,1), imagesize=(128,128)):
    direction = (1-direction[0], 1-direction[1], 1-direction[2], 1-direction[3])
    split_w = ((int)(rate * imagesize[0]), 0)
    split_h = ((int)(rate * imagesize[1]), 0)
    split_w = (split_w[0], imagesize[0] - split_w[0])
    split_h = (split_h[0], imagesize[1] - split_h[0])
    
    maskw = np.ones((split_w[0], imagesize[0]))
    maskw_ = np.ones((split_w[1], imagesize[0]))
    maskh = np.ones((split_h[0], imagesize[0]))
    maskh_ = np.ones((split_h[1], imagesize[0]))
    maskt = np.r_[maskw * direction[0], maskw_]
    maskb = np.r_[maskw_, maskw * direction[1]]
    maskl = np.r_[maskh * direction[2], maskh_].T
    maskr = np.r_[maskh_, maskh * direction[3]].T
    maskc = np.ones(imagesize)
    
    mask = maskc * maskt * maskb * maskl * maskr
    return np.array(mask, dtype=np.uint8)

def blacken(image, mask=gen_mask()):
    img=np.array(image)
    img[:, :, 0] *= mask
    img[:, :, 1] *= mask
    img[:, :, 2] *= mask
    return img

if __name__ == '__main__':
    file = read_pairs()
    yt, yp = evaluate(vector_split=zero_set, loss='lmcl')
    #for i in range(8):
    #    draw_roc(yt[0], yp[i], legend='Zero Setted: %d'%zero_set[i], show=False)
    draw_roc(yt, yp)
    plt.show()
    '''
    yt, yp = [], []
    for i in rates:
        yt__, yp__ = np.zeros(128), np.zeros(128)
        for j in masks:
            yt_, yp_ = evaluate(count=600, black=gen_mask(i, j))
            yt__ = yt__ + yt_[0]
            yp__ = yp__ + yp_[0]
        yt__ = yt__ / (rates.size * np.sum(i))
        yp__ = yp__ / (rates.size * np.sum(i))
        yt.append(yt__)
        yp.append(yp__)
    for i in rates.size:
        draw_roc(yt[i], yp[i], legend='')
    '''
    
#if __name__ == '__main__':
#    main()