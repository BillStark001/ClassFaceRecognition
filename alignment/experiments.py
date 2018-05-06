# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 15:25:02 2018
@author: BillStark001
"""

#LOCAL
import main as align
#NOT LOCAL
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2

root = 'F:/Datasets'
#root = 'C:/Users/zhaoj/Documents/Datasets'
train_dir = root+'/VGGFACE2/train/'
val_dir = root+'/VGGFACE2/val/'

pics = range(200)
masks = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
                  [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1], 
                  [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
                  [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]])
rates = np.arange(0, 0.5, 0.01)

    
def read_img(file_path, size=(128, 128)):
    img = cv2.imread(file_path)
    if img is None:return img
    h, w, c = img.shape
    if c == 1: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img

def generator(path, ext='jpg'):
    path=glob.glob(path+'*')
    while 1:
        n = np.random.choice(path)
        p = glob.glob(n+'/*.'+ext)
        p = np.random.choice(p)
        x = read_img(p)
        yield x
        
gen = generator(train_dir)

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

try:
    assert average and succeeded
except:
    average = [{}, {}, {}, {}, {}]
    succeeded = [{}, {}, {}, {}, {}]

def flow():
    global average
    global succeeded
    average = [{}, {}, {}, {}, {}]
    succeeded = [{}, {}, {}, {}, {}]
    for i in average:
        for j in rates:
            i[j] = 0.0
            
    for i in succeeded:
        for j in rates:
            i[j] = 1e-15
    for i in pics:
        pic_cur = next(gen)
        lm_standard = align.predictSingle(pic_cur).T
        while lm_standard.size == 0:
            pic_cur = next(gen)
            lm_standard = align.predictSingle(pic_cur).T
        print('[*] Disciplining picture %d......'%i)
        for j in masks:
            print(('[D-%d]   Torturing mask {}......'%i).format(j))
            for k in rates:
                #print(('[D-%d T-{}]     Toying with blacken rate {}......'%i).format(np.sum(j), k))
                pic_temp = blacken(pic_cur, gen_mask(rate=k, direction=j))
                lm_temp = align.predictSingle(pic_temp).T
                #plt.imshow(pic_temp)
                #plt.show()
                if lm_temp.size != 0:
                    succeeded[np.sum(j)][k] += 1
                    lm_temp = lm_temp - lm_standard
                    #print(lm_temp)
                    lm_temp = np.sqrt(lm_temp[0]*lm_temp[0] + lm_temp[1]*lm_temp[1])
                    lm_temp = lm_temp.dot(lm_temp.T)
                    average[np.sum(j)][k] += np.sqrt(lm_temp)
                    #print('               Succeeded. RMSE: %.3f'%np.sqrt(lm_temp))
                #else:
                    #print('               Failed.')
try:
    assert ans
except:
    ans=[]
    
def process():
    global ans
    ans=[]
    for i in range(5):
        avg_temp = np.array(list(sorted(average[i].items())))[:, 1]
        suc_temp = np.array(list(sorted(succeeded[i].items())))[:, 1]
        avg_temp = avg_temp / suc_temp
        ans.append(avg_temp)
    handles=[]
    for i in range(5):
        handles.append(plt.scatter(rates, ans[i], s=0.7, label='Blackend Side=%d'%(i)))
    plt.legend(handles=handles)
    plt.xlabel('Blackening rate')
    plt.ylabel('RMSE')
    plt.show()
        

if __name__ == '__main__':
    '''
    for j in masks:
        mask = gen_mask(direction=j)
        plt.imshow(mask)
        plt.show()
    '''
    #flow()
    process()
