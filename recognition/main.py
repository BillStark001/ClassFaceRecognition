# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 20:29:27 2018
@author: BillStark001
"""

import keras
import models

from imagenet_utils import decode_predictions, preprocess_input


import numpy as np
import cv2

def preprocess(img):
    
    return img

input_dimension=(1,64,64,3)
#input_tensor=np.zeros(input_dimension,dtype='int32')
#input_tensor=preprocess(input_tensor)
#input_tensor=K.variable(input_tensor)
input_tensor=K.placeholder((input_dimension))
model=models.VGG16(input_tensor=input_tensor)

#Simpler Version


if __name__=='__main__':
    print(model)
    
    img_path='../face_0.jpg'
    img=image.load_img(img_path,target_size=(224, 224))
    img=image.img_to_array(img)
    img=np.expand_dims(img,axis=0)
    img=preprocess_input(img)
    
    preds=model.predict(img)
    print('Predicted:', decode_predictions(preds))