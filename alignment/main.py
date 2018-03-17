# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 17:32:47 2018
@author: BillStark001
"""

import dlib
from skimage import io
import cv2
import matplotlib.pyplot as plt
import numpy

detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def predict(img):
    landmarks=[]
    dets = detector(img, 1)
    for i, d in enumerate(dets):
        shape = predictor(img, d) 
        landmark = numpy.matrix([[p.x, p.y] for p in shape.parts()])
        landmarks.append(landmark)
    return dets, landmarks
        
def predictSingle(img):
    dets = detector(img, 1)
    landmark=numpy.array([])
    for i, d in enumerate(dets):
        shape = predictor(img, d) 
        landmark = numpy.matrix([[p.x, p.y] for p in shape.parts()])
    return landmark
    
def align(img,landmark):
    #NOT FINISHED!!!!!
    return img

if __name__=='__main__':
    img = io.imread("1.jpg")
    dets, landmarks=predict(img)
    for i, d in enumerate(dets):
        cv2.rectangle(img,(d.left(),d.top()),(d.right(),d.bottom()),(255,0,0),1)
    for landmark in landmarks:
        for l in landmark:
            cv2.rectangle(img,(l[:,0],l[:,1]),(l[:,0]+1,l[:,1]+1),(0,0,255),1)
    plt.imshow(img)
    plt.show()

