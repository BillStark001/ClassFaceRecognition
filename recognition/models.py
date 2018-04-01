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

def euclidean_distance(inputs):
    assert len(inputs) == 2, \
        'Euclidean distance needs 2 inputs, %d given' % len(inputs)
    u, v = inputs
    return K.sqrt(K.sum((K.square(u - v)), axis=1, keepdims=True))

def contrastive_loss(y_true,y_pred):
    margin=1.
    return K.mean((1. - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0.)))

def MobileNet_FT(shape=(112,112,3)):
	
    model=Xception(include_top=False, weights='imagenet', input_tensor=None, input_shape=shape, pooling=None)
    model.summary()
    
    im_in = Input(shape=shape)
    
    x1 = model(im_in)
    x1 = Flatten()(x1)
    x1 = Dense(512, activation="relu")(x1)
    x1 = Dropout(0.2)(x1)
    
    feat_x = Dense(128, activation="linear")(x1)
    feat_x = Lambda(lambda  x: K.l2_normalize(x,axis=1))(feat_x)
    
    model_top = Model(inputs = [im_in], outputs = feat_x)
    
    model_top.summary()
    
    im_in1 = Input(shape=shape)
    im_in2 = Input(shape=shape)
    
    feat_x1 = model_top(im_in1)
    feat_x2 = model_top(im_in2)
    
    lambda_merge = Lambda(euclidean_distance)([feat_x1, feat_x2])
    
    model_final = Model(inputs = [im_in1, im_in2], outputs = lambda_merge)
    
    model_final.summary()
    
    adam = Adam(lr=0.001)
    sgd = SGD(lr=0.001, momentum=0.9)
    model_final.compile(optimizer=adam, loss=contrastive_loss)
	
    return model_final
    
def Xception_FT(shape=(112,112,3)):
	
    model=Xception(include_top=False, weights='imagenet', input_tensor=None, input_shape=shape, pooling=None)
    model.summary()
    
    im_in = Input(shape=shape)
    
    x1 = model(im_in)
    x1 = Flatten()(x1)
    x1 = Dense(512, activation="relu")(x1)
    x1 = Dropout(0.2)(x1)
    
    feat_x = Dense(128, activation="linear")(x1)
    feat_x = Lambda(lambda  x: K.l2_normalize(x,axis=1))(feat_x)
    
    model_top = Model(inputs = [im_in], outputs = feat_x)
    
    model_top.summary()
    
    im_in1 = Input(shape=shape)
    im_in2 = Input(shape=shape)
    
    feat_x1 = model_top(im_in1)
    feat_x2 = model_top(im_in2)
    
    lambda_merge = Lambda(euclidean_distance)([feat_x1, feat_x2])
    
    model_final = Model(inputs = [im_in1, im_in2], outputs = lambda_merge)
    
    model_final.summary()
    
    adam = Adam(lr=0.001)
    sgd = SGD(lr=0.001, momentum=0.9)
    model_final.compile(optimizer=adam, loss=contrastive_loss)
	
    return model_final

def fire(x, squeeze=16, expand=64):
    x = Convolution2D(squeeze, (1,1), padding='valid')(x)
    x = Activation('relu')(x)
    
    left = Convolution2D(expand, (1,1), padding='valid')(x)
    left = Activation('relu')(left)
    
    right = Convolution2D(expand, (3,3), padding='same')(x)
    right = Activation('relu')(right)
    
    x = concatenate([left, right], axis=3)
    return x

def SqueezeNet(shape=(200,200,4)):
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
    
    modelsqueeze.summary()
    
    im_in = Input(shape=shape)
    
    x1 = modelsqueeze(im_in)
    x1 = Flatten()(x1)
    x1 = Dense(512, activation="relu")(x1)
    x1 = Dropout(0.2)(x1)
    
    feat_x = Dense(128, activation="linear")(x1)
    feat_x = Lambda(lambda  x: K.l2_normalize(x,axis=1))(feat_x)
    
    model_top = Model(inputs = [im_in], outputs = feat_x)
    
    model_top.summary()
    
    im_in1 = Input(shape=shape)
    im_in2 = Input(shape=shape)
    
    feat_x1 = model_top(im_in1)
    feat_x2 = model_top(im_in2)
    
    lambda_merge = Lambda(euclidean_distance)([feat_x1, feat_x2])
    
    model_final = Model(inputs = [im_in1, im_in2], outputs = lambda_merge)
    
    model_final.summary()
    
    adam = Adam(lr=0.001)
    sgd = SGD(lr=0.001, momentum=0.9)
    model_final.compile(optimizer=adam, loss=contrastive_loss)
    return model_final
	
def main():
    model=MobileNet_FT()
	
if __name__=='__main__':
    main()

# VGGs are DESERTED!

def VGG16(input_tensor=None):
    '''
    # Arguments
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
    # Returns
        A Keras model instance.
    '''
    input_shape = (None, None, 3)
    
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor)
        else:
            img_input = input_tensor
    # Block 1
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(img_input)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Create model
    model = Model(img_input, x)

    # load weights
    '''weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                            'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                            cache_subdir='models')'''
    weights_path='models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    model.load_weights(weights_path)
    
    return model
    
def VGG16_TOP(input_tensor=None):
    '''
    # Arguments
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
    # Returns
        A Keras model instance.
    '''
    input_shape = (224, 224, 3)
    
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor)
        else:
            img_input = input_tensor
    # Block 1
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(img_input)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(1000, activation='softmax', name='predictions')(x)

    # Create model
    model = Model(img_input, x)

    # load weights
    '''weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                            'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                            cache_subdir='models')'''
    weights_path='models/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    model.load_weights(weights_path)
    
    return model
    
def conv_bn_relu(x, out_ch, name):
    x = Convolution2D(out_ch, 3, 3, border_mode='same', name=name)(x)
    x = BatchNormalization(name='{}_bn'.format(name))(x)
    x = Activation('relu', name='{}_relu'.format(name))(x)
    return x

def VGG16_strange(input_shape=(224, 224, 3), nb_classes=1024, weights_path=None):

    inputs = Input(shape=input_shape, name='input')

    x = conv_bn_relu(inputs, 64, name='block1_conv1')
    x = conv_bn_relu(x, 64, name='block1_conv2')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = conv_bn_relu(x, 128, name='block2_conv1')
    x = conv_bn_relu(x, 128, name='block2_conv2')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = conv_bn_relu(x, 256, name='block3_conv1')
    x = conv_bn_relu(x, 256, name='block3_conv2')
    x = conv_bn_relu(x, 256, name='block3_conv3')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = conv_bn_relu(x, 512, name='block4_conv1')
    x = conv_bn_relu(x, 512, name='block4_conv2')
    x = conv_bn_relu(x, 512, name='block4_conv3')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(nb_classes, activation='softmax', name='predictions')(x)
    
    model = Model(input=inputs, output=x)
    
    if not weights_path is None:
        model.load_weights(weights_path)

    return model