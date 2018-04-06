# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 21:26:16 2018
@author: BillStark001
"""

from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, ELU, concatenate, GlobalAveragePooling2D, Input, BatchNormalization, SeparableConv2D, Subtract
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet

print('Recognition Models Loaded.')

def euclidean_distance(inputs):
    assert len(inputs) == 2, \
        'Euclidean distance needs 2 inputs, %d given' % len(inputs)
    u, v = inputs
    return K.sqrt(K.sum((K.square(u - v)), axis=1, keepdims=True))

def contrastive_loss(y_true,y_pred):
    margin=1.
    return K.mean((1. - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0.)))

def lmcl(y_true,y_pred):
    pass
    
def Siamase1(model,shape=(128,128,3)):
    
    im_in = Input(shape=shape)
    
    x1 = model(im_in)
    x1 = Flatten()(x1)
    x1 = Dense(512, activation="tanh")(x1)
    x1 = Dropout(0.5)(x1)
    
    feat_x = Dense(128, activation="linear")(x1)
    feat_x = Lambda(lambda  x: K.l2_normalize(x,axis=1))(feat_x)
    
    model_top = Model(inputs = [im_in], outputs = feat_x)
    #model_top.summary()
    
    im_in1 = Input(shape=shape)
    im_in2 = Input(shape=shape)
    
    feat_x1 = model_top(im_in1)
    feat_x2 = model_top(im_in2)
    
    lambda_merge = Lambda(euclidean_distance)([feat_x1, feat_x2])
    
    model_final = Model(inputs = [im_in1, im_in2], outputs = lambda_merge)
    #model_final.summary()
    
    adam = Adam(lr=0.001)
    sgd = SGD(lr=0.001, momentum=0.9)
    model_final.compile(optimizer=sgd, loss=contrastive_loss)
	
    return model_final

def MobileNet_FT(shape=(128,128,3)):
    model=MobileNet(include_top=False, weights='imagenet', input_tensor=None, input_shape=shape, pooling=None)
    #model.summary()
    return Siamase1(model,shape=shape)

def fire(x, squeeze=16, expand=64):
    x = Convolution2D(squeeze, (1,1), padding='valid')(x)
    x = Activation('relu')(x)
    
    left = Convolution2D(expand, (1,1), padding='valid')(x)
    left = Activation('relu')(left)
    
    right = Convolution2D(expand, (3,3), padding='same')(x)
    right = Activation('relu')(right)
    
    x = concatenate([left, right], axis=3)
    return x

def SqueezeNet(shape=(112,112,3)):
    img_input=Input(shape=shape)
    
    x = Convolution2D(64, (5, 5), strides=(2, 2), padding='valid')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    
    x = fire(x, squeeze=16, expand=16)
    x = fire(x, squeeze=16, expand=16)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    
    x = fire(x, squeeze=32, expand=32)
    x = fire(x, squeeze=32, expand=32)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    
    x = fire(x, squeeze=48, expand=48)
    x = fire(x, squeeze=48, expand=48)
    x = fire(x, squeeze=64, expand=64)
    x = fire(x, squeeze=64, expand=64)
    
    x = Dropout(0.2)(x)
    
    x = Convolution2D(512, (1, 1), padding='same')(x)
    out = Activation('relu')(x)
    
    modelsqueeze=Model(img_input, out)
    #modelsqueeze.summary()
    return Siamase1(modelsqueeze,shape=shape)
	
def main():
    model=MobileNet_FT()
	
if __name__=='__main__':
    main()

