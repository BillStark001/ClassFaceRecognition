# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 21:26:16 2018
@author: BillStark001
"""

from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, ELU, concatenate, GlobalAveragePooling2D, Input, BatchNormalization, SeparableConv2D, Subtract
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.layers import activations, initializers, regularizers, constraints
from keras import backend as K
from keras.applications.mobilenet import MobileNet
from keras.engine import InputSpec
from keras.engine.topology import Layer
import tensorflow as tf
import numpy as np

print('Recognition Models Loaded.')

class AMSoftmax(Layer):
    def __init__(self, units, s, m,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs
                 ):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(AMSoftmax, self).__init__(**kwargs)
        self.units = units
        self.s = s
        self.m = m
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True


    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True


    def call(self, inputs, **kwargs):
        inputs = tf.nn.l2_normalize(inputs, dim=-1)
        self.kernel = tf.nn.l2_normalize(self.kernel, dim=(0, 1))   # W归一化

        dis_cosin = K.dot(inputs, self.kernel)
        psi = dis_cosin - self.m

        e_costheta = K.exp(self.s * dis_cosin)
        e_psi = K.exp(self.s * psi)
        sum_x = K.sum(e_costheta, axis=-1, keepdims=True)

        temp = e_psi + sum_x - e_costheta

        output = e_psi / temp
        return output

def euclidean_distance(inputs):
    assert len(inputs) == 2, \
        'Euclidean distance needs 2 inputs, %d given' % len(inputs)
    u, v = inputs
    return K.sqrt(K.sum((K.square(u - v)), axis=1, keepdims=True))

def amsoftmax_loss(y_true, y_pred):
    d1 = K.sum(y_true * y_pred, axis=-1)
    d1 = K.log(K.clip(d1, K.epsilon(), None))
    loss = -K.mean(d1, axis=-1)
    return loss

def contrastive_loss(y_true,y_pred):
    margin=1.
    return K.mean((1. - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0.)))

def Siamase1(model,opt='sgd',shape=(128,128,3)):
    
    im_in = Input(shape=shape)
    
    x1 = model(im_in)
    x1 = GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1)
    x1 = Dense(512, activation="tanh")(x1)
    x1 = Dense(512, activation="tanh")(x1)
    x1 = Dropout(0.2)(x1)
    
    feat_x = Dense(128, activation="tanh")(x1)
    feat_x = Lambda(lambda x: K.l2_normalize(x,axis=1))(feat_x)
    
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
    opt_dict={'adam':adam,'sgd':sgd}
    model_final.compile(optimizer=opt_dict[opt], loss=contrastive_loss)
	
    return model_final

def MobileNet_FT(opt='sgd',shape=(128,128,3)):
    model=MobileNet(include_top=False, weights='imagenet', input_tensor=None, input_shape=shape, pooling=None)
    #model.summary()
    return Siamase1(model,opt=opt,shape=shape)

def MobileNet_AM(opt='sgd',shape=(128,128,3)):
    im_in=Input(shape=shape)
    model=MobileNet(include_top=False, weights='imagenet', input_tensor=None, input_shape=shape, pooling=None)(im_in)
    model=GlobalAveragePooling2D()(model)
    model=Dense(512, activation="tanh")(model)
    model=Dropout(0.2)(model)
    model=AMSoftmax(10,10,0.35)(model)
    model=Model(inputs=im_in,outputs=model)
    #model.summary()
    
    adam = Adam(lr=0.001)
    sgd = SGD(lr=0.001, momentum=0.9)
    opt_dict={'adam':adam,'sgd':sgd}
    model.compile(optimizer=opt_dict[opt], loss=amsoftmax_loss)
    return model

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
    model=MobileNet_AM()
	
if __name__=='__main__':
    main()

