# -*- coding: utf-8 -*-
"""
Author: BillStark001
Update: 2018/02/23
"""

import cv2
import numpy as np
#import parent.vstream.cv as VStream

CASC_PATH = 'cv_haarcascade_files/haarcascade_frontalface_default.xml'

def detectFace(image):
    if image is None:return []
    Cascade=cv2.CascadeClassifier(CASC_PATH)
    #if image is not None:print(image.scale)
    #print(np.array(image).size)
    #print(image)
    coors=Cascade.detectMultiScale(image,scaleFactor=1.3,minNeighbors=5)
    return coors

def drawFaceCoor(image,coors,color=(0,255,0),width=1):
    for coor in coors:
        [x,y,w,h]=coor
        image=cv2.rectangle(image,(x,y),(x+w,y+h),color,width)
    return image

def splitbyCoor(image,coors,resize=False,size=(48,48)):
    faces=[]
    for coor in coors:
        face=image[coor[1]:(coor[1]+coor[2]),coor[0]:(coor[0]+coor[3])]
        if resize:
            try:
                face=cv2.resize(face,size,interpolation=cv2.INTER_CUBIC)
            except Exception:pass
        faces.append(face)
    return faces
    
def saveFaces(faces,path='face_%d.png'):
    print(faces)
    for i in range(len(faces)):
        cv2.imwrite(path%i,faces[i])

if __name__=='__main__':
    vc=cv2.VideoCapture(0)
    while True:
        ret,frame=vc.read()
        #print(np.array(frame).size)
        coors=detectFace(frame)
        #print(coors)
        frame=drawFaceCoor(frame,coors)
        saveFaces(splitbyCoor(frame,coors))
        #print(frame)
        print('1234567890')
        cv2.imshow('Capturing',frame)
        if cv2.waitKey(1)&0xFF == ord('q'):
            break
    vc.release()

