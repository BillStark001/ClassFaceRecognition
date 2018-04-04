# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 21:57:15 2018
@author: BillStark001
"""

import cv2
import sys
import dlib
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def cutFrame(filepath,timeF=1):
    frames=[]
    vc = cv2.VideoCapture(filepath) #读入视频文件
    if vc.isOpened(): #判断是否正常打开
        rval,frame = vc.read()
    else:
        vc.release()
        return []
    c=0
    while rval:   #循环读取视频帧
        rval,frame = vc.read()
        if(c%timeF == 0): #每隔timeF帧进行存储操作
            frames.append(frame)
        c+=1
        cv2.waitKey(1)
    vc.release()
    return frames
    
detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
important_lms = (17,26,48,54,51,57,60,64,62,66,27,8)

def getMouthPos(img):
    lms=[]
    if img is None:
        return lms
    dets = detector(img, 1)  
    #print("Number of faces detected: {}".format(len(dets))) 
    
    for i, d in enumerate(dets):
        #print("Detection {}: {}".format(i,d))
        shape = predictor(img, d) 
        landmark = np.array([[p.x, p.y] for p in shape.parts()])
        lmt=[]
        #cv2.rectangle(img,(d.left(),d.top()),(d.right(),d.bottom()),(255,0,0),3)
        for i in important_lms:
            lmt.append(landmark[i])
            #cv2.rectangle(img,(landmark[i,0],landmark[i,1]),(landmark[i,0]+1,landmark[i,1]+1),(0,0,255),3)
        a0=(((lmt[0][0]-lmt[1][0])**2+(lmt[0][1]-lmt[1][1])**2)**0.5)/35
        a1=(((lmt[10][0]-lmt[11][0])**2+(lmt[10][1]-lmt[11][1])**2)**0.5)/35
        b1=((lmt[2][0]-lmt[3][0])**2+(lmt[2][1]-lmt[3][1])**2)**0.5
        b2=((lmt[4][0]-lmt[5][0])**2+(lmt[4][1]-lmt[5][1])**2)**0.5
        b3=((lmt[6][0]-lmt[7][0])**2+(lmt[6][1]-lmt[7][1])**2)**0.5
        b4=((lmt[8][0]-lmt[9][0])**2+(lmt[8][1]-lmt[9][1])**2)**0.5
        lm=[b1/a0,b2/a1,b3/a0,b4/a1]
        lms.append(lm)
    #plt.imshow(img)
    #plt.show()
    return lms

def serialize(filepath,timeF=1):
    frames=cutFrame(filepath,timeF)
    #print(frames)
    print('Frames Cut. Count:{}'.format(len(frames)))
    lms=[]
    c=0
    for i in frames:
        if i is None:continue
        lm=getMouthPos(i)
        print('Mouth Pos {} of Shape {} Get.'.format(c,i.shape))
        c+=1
        if lm!=[]:lms.append(lm[0])
    return lms

def main():   
    img = io.imread("xjp.jpg")
    print(getMouthPos(img))
    series=np.array(serialize('lyl.avi',1)).T
    with open('lyl1.txt', 'w') as f:
        for s in series:
            for ss in s:
                f.write(str(ss))
                f.write(' ')
            f.write('\n')

if __name__=='__main__':
    main()

