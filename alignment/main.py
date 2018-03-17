# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 17:32:47 2018
@author: BillStark001
"""

import sys
import dlib
from skimage import io
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy

detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def predict(img):
    return img

img = io.imread("1.jpg")
print(img)

dets = detector(img, 1)  
print("Number of faces detected: {}".format(len(dets))) 
for i, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i+1, d.left(), d.top(), d.right(), d.bottom()))
    print(d)
    shape = predictor(img, d) 
    landmark = numpy.matrix([[p.x, p.y] for p in shape.parts()])
    cv2.rectangle(img,(d.left(),d.top()),(d.right(),d.bottom()),(255,0,0),1)
    for l in landmark:
        cv2.rectangle(img,(l[:,0],l[:,1]),(l[:,0]+1,l[:,1]+1),(0,0,255),1)
        
plt.imshow(img)
plt.show()

