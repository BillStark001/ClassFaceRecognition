# -*- coding: utf-8 -*-
'''
Created on Mon Mar  5 21:26:16 2018
@author: BillStark001
'''

import keras
from keras.models import Model
from keras.layers import Activation, Dense, Lambda, GlobalAvgPool2D, Input, BatchNormalization, Conv2D, PReLU
from keras.optimizers import Adam, SGD
from keras.layers import constraints
from keras import backend as K
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import DepthwiseConv2D, relu6
#from keras.utils.vis_utils import plot_model 
from keras.regularizers import l2
import tensorflow as tf

print('Recognition Models Loaded.')

def euclidean_distance(inputs):
    assert len(inputs) == 2, \
        'Euclidean distance needs 2 inputs, %d given' % len(inputs)
    u, v = inputs
    return K.sqrt(K.sum((K.square(u - v)), axis=1, keepdims=True))

def contrastive_loss(y_true,y_pred):
    margin=1.
    return K.mean((1. - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0.)))

def Siamase(model, opt='sgd', shape=(128,128,3), embedding_size=128):
    im_in = Input(shape=shape)
    
    x = model(im_in)
    x = GlobalAvgPool2D()(x)
    #x = Dense(512, activation='tanh', name='fc_00')(x)
    #feat_x = Dense(128, activation='tanh', name='fc_out')(x)
    #feat_x = Lambda(lambda x: K.l2_normalize(x,axis=1))(feat_x)
    
    x = Dense(512, kernel_regularizer=l2(5e-4), name='fc_00')(x)
    x = PReLU(name='prelu_1')(x)
    x = Dense(embedding_size, name='fc_embed_%d'%embedding_size)(x)
    x = PReLU(name='prelu_2')(x)
    
    model_top = Model(inputs = [im_in], outputs = x)
    #plot_model(model_top,to_file='model_top.png',show_shapes=True)
    #model_top.summary()
    
    im_in1 = Input(shape=shape)
    im_in2 = Input(shape=shape)
    
    feat_x1 = model_top(im_in1)
    feat_x2 = model_top(im_in2)
    
    lambda_merge = Lambda(euclidean_distance)([feat_x1, feat_x2])
    
    model_final = Model(inputs = [im_in1, im_in2], outputs = lambda_merge)
    #model_final.summary()
    model_final.compile(optimizer=opt, loss=contrastive_loss)
    #plot_model(model_final, to_file='model_siamase1.png', show_shapes=True)
	
    return model_final, model_top

def MobileNet_FT(opt='sgd', shape=(128,128,3), output_fc=False):
    x = MobileNet(include_top=False, weights='imagenet', input_tensor=None, input_shape=shape, pooling=None)
    m1,m2 = Siamase(x, opt=opt, shape=shape)
    if output_fc:
        return m2
    else:
        return m1
    
#--------
    
def _res_block(inputs, expansion, out_channels, stride=1, weight_decay=1e-5, shortcut=True, prefix=''):
    channel_axis = 1 if K.image_data_format == 'channels_first' else -1
    in_channels = K.int_shape(inputs)[channel_axis]

    # pointwise conv
    bottleneck_channels = int(expansion * in_channels)
    x = Conv2D(bottleneck_channels, (1, 1), padding='same', use_bias=False,
            kernel_regularizer=l2(weight_decay), name='%s/1x1_conv_1' % prefix)(inputs)
    x = BatchNormalization(axis=channel_axis, name='%s/bn_conv_1' % prefix)(x)
    x = Activation(relu6)(x)

    # depthwise conv
    x = DepthwiseConv2D((3, 3), padding='same', strides=stride, use_bias=False,
            depth_multiplier=1, depthwise_regularizer=l2(weight_decay), name='%s/3x3_dwconv_1' % prefix)(x)
    x = BatchNormalization(axis=channel_axis, name='%s/bn_dwconv_1' % prefix)(x)
    x = Activation(relu6)(x)

    # pointwise conv
    x = Conv2D(out_channels, (1, 1), padding='same', use_bias=False,
            kernel_regularizer=l2(weight_decay), name='%s/1x1_conv_2' % prefix)(x)
    x = BatchNormalization(axis=channel_axis, name='%s/bn_conv_2' % prefix)(x)
    # no activation here

    if shortcut and stride == 1:
        if in_channels != out_channels:
            inputs = Conv2D(out_channels, (1, 1), padding='same', use_bias=False,
                    kernel_regularizer=l2(weight_decay), name='%s/1x1_conv_extra' % prefix)(inputs)
        
        x = keras.layers.add([x, inputs])
    
    return x


def _block(inputs, expansion, out_channels, repeat, stride, block_id, weight_decay=1e-5):
    prefix = 'block%d' % block_id
    x = _res_block(inputs, expansion, out_channels, stride=stride,
            weight_decay=weight_decay, prefix='%s_1' % prefix)

    for i in range(1, repeat):
        x = _res_block(x, expansion, out_channels, stride=1, 
                weight_decay=weight_decay, prefix='%s_%d' % (prefix, i + 1))
    
    return x


def _conv_block(inputs, filters, name, kernel_size=(3, 3), stride=1, weight_decay=1e-5):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = Conv2D(filters, kernel_size, padding='same', strides=stride,
            kernel_regularizer=l2(weight_decay), name=name)(inputs)
    x = BatchNormalization(axis=channel_axis, name='bn_%s' % name)(x)
    x = Activation(relu6)(x)
    return x


def MobileNetV2(input_shape, include_top=False, input_tensor=None, 
            expansion=6, classes=1000, weight_decay=1e-5):
    
    if input_tensor is None:
        inputs = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            inputs = Input(tensor=input_tensor, shape=input_shape)
        else:
            inputs = input_tensor
    
    x = _conv_block(inputs, 32, kernel_size=(3, 3), stride=2, name='conv1', weight_decay=weight_decay)

    x = _block(x, expansion=1, out_channels=16, repeat=1, stride=1, weight_decay=weight_decay, block_id=1)
    x = _block(x, expansion=expansion, out_channels=24, repeat=2, stride=2, weight_decay=weight_decay, block_id=2)
    x = _block(x, expansion=expansion, out_channels=32, repeat=3, stride=2, weight_decay=weight_decay, block_id=3)
    x = _block(x, expansion=expansion, out_channels=64, repeat=4, stride=2, weight_decay=weight_decay, block_id=4)
    x = _block(x, expansion=expansion, out_channels=96, repeat=3, stride=1, weight_decay=weight_decay, block_id=5)
    x = _block(x, expansion=expansion, out_channels=160, repeat=3, stride=2, weight_decay=weight_decay, block_id=6)
    x = _block(x, expansion=expansion, out_channels=320, repeat=1, stride=1, weight_decay=weight_decay, block_id=7)

    x = _conv_block(x, 1280, kernel_size=(1, 1), stride=1, name='conv2', weight_decay=weight_decay)
    x = GlobalAvgPool2D(name='global_pool')(x)
    
    if include_top:
        x = Dense(classes, kernel_regularizer=l2(weight_decay), name='fc')(x)
        x = Activation('softmax', name='softmax')(x)

    model = Model(inputs, x, name='MobileNetV2_%.2gX' % expansion)
    return model

def DenseLMCL(x, units, name='fc_train', s=24, m=0.2):
    W = Dense(units, use_bias=False, kernel_constraint=constraints.unit_norm(), name=name)
    normalized_x = Lambda(lambda x: tf.nn.l2_normalize(x, -1))(x)
    cos_theta = W(normalized_x)
    output = Lambda(lambda x: x * s)(cos_theta)
    
    def lmcl_loss(y_true, y_pred, s=s, m=m):
        y_pred -= s * m * y_true  # y_true: one-hot vector
        return tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
    
    return output, lmcl_loss

def ModelLMCL(base_model, output_fc=False, scale=30, margin=0.1, opt='adam', embedding_size=128, classes=512):
    im_in = base_model.input
    
    x = Dense(512, kernel_regularizer=l2(5e-4), name='fc_00')(base_model.output)
    x = PReLU(name='prelu_1')(x)
    x = Dense(embedding_size, name='fc_embed_%d'%embedding_size)(x)
    x = PReLU(name='prelu_2')(x)
    
    fc_out = x
    lmcl_out, lmcl_loss=DenseLMCL(x, units=classes, name='fc_train_%d'%classes, s=scale, m=margin)
    if not output_fc:
        model = Model(im_in, lmcl_out)
        model.compile(optimizer=opt, loss=lmcl_loss, metrics=['acc'])
    else:
        model = Model(im_in, fc_out)
        
    #model.summary()
    return model

def MobileNet_LMCL(output_fc=False, scale=30, margin=0.1, opt='adam', input_shape=(128,128,3), embedding_size=128, classes=1000):
    model = MobileNet(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')
    model = ModelLMCL(model, output_fc=output_fc, scale=scale, margin=margin, opt=opt, embedding_size=embedding_size, classes=classes)
    return model

def MobileNetV2_LMCL(output_fc=False, scale=30, margin=0.1, opt='adam', input_shape=(128,128,3), embedding_size=128, classes=1000):
    model = MobileNetV2(input_shape=input_shape, include_top=False, expansion=4)
    model = ModelLMCL(model, output_fc=output_fc, scale=scale, margin=margin, opt=opt, embedding_size=embedding_size, classes=classes)
    return model

def main():
    model=MobileNet_FT()
    #model=MobileNetV2_LMCL()
    model.summary()
    #plot_model(model,to_file='model_siamase.png',show_shapes=True)
    
	
if __name__=='__main__':
    main()