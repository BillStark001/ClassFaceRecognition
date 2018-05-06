# -*- coding: utf-8 -*-
'''
Created on Mon Mar  5 21:26:16 2018
@author: BillStark001
'''

from keras.models import Model
from keras.layers import Activation, Flatten, Dense, Dropout, Lambda, concatenate, GlobalAveragePooling2D, Input, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.layers import activations, initializers, regularizers, constraints
from keras import backend as K
from keras.applications.mobilenet import MobileNet
from keras.utils.vis_utils import plot_model 
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

def Siamase1(model,opt='sgd',shape=(128,128,3)):
    im_in = Input(shape=shape)
    
    x1 = model(im_in)
    x1 = GlobalAveragePooling2D()(x1)
    x1 = Dense(512, activation='tanh', name='fc_00')(x1)
    feat_x = Dense(128, activation='tanh', name='fc_out')(x1)
    feat_x = Lambda(lambda x: K.l2_normalize(x,axis=1))(feat_x)
    
    model_top = Model(inputs = [im_in], outputs = feat_x)
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

def MobileNet_FT(opt='sgd',shape=(128,128,3),output_fc=False):
    model = MobileNet(include_top=False, weights='imagenet', input_tensor=None, input_shape=shape, pooling=None)
    m1,m2 = Siamase1(model, opt=opt, shape=shape)
    if output_fc:
        return m2
    else:
        return m1
    
def DenseLMCL(x, units, name='fc_final', s=24, m=0.2):
    W = Dense(units, use_bias=False, kernel_constraint=constraints.unit_norm(), name=name)
    normalized_x = Lambda(lambda x: tf.nn.l2_normalize(x, -1))(x)
    cos_theta = W(normalized_x)
    output = Lambda(lambda x: x * s)(cos_theta)
    
    def lmcl_loss(y_true, y_pred, s=s, m=m):
        y_pred -= s * m * y_true  # y_true: one-hot vector
        return tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
    
    return output, lmcl_loss

def MobileNet_LMCL(output_fc=False,opt='adam',shape=(128,128,3),units=500):
    im_in=Input(shape=shape,name='im_in')
    model=MobileNet(include_top=False, weights='imagenet', input_tensor=None, input_shape=shape, pooling=None)(im_in)
    model=GlobalAveragePooling2D()(model)
    #model=Dropout(0.25)(model)
    model=Dense(512, activation='tanh', name='fc_00')(model)
    model=Dense(128, activation='tanh', name='fc_out')(model)
    
    fc_out=model
    lmcl_out,lmcl_loss=DenseLMCL(model,units=units,name='fc_final')
    
    if not output_fc:
        model = Model(im_in, lmcl_out)
        model.compile(optimizer=opt, loss=lmcl_loss, metrics=['acc'])
    else:
        model = Model(im_in, fc_out)
        
    #model.summary()
    return model

def main():
    model=MobileNet_FT()
    model.summary()
    #plot_model(model,to_file='model_siamase.png',show_shapes=True)
    
	
if __name__=='__main__':
    main()

'''

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
    
    x = Convolution2D(512, (1, 1), padding='same', activation='tanh')(x)
    #out = Activation('relu')(x)
    
    model=Model(img_input, x)
    return model

def SqueezeNet_Cons(shape=(112,112,3)):
    modelsqueeze=SqueezeNet(shape=shape)
    return Siamase1(modelsqueeze,shape=shape)

def sn_vap(load=1,savepath='sn.h5'):
    gen=data_loader.gen_vap
    val_gen=data_loader.val_gen_vap
    model=models.SqueezeNet()
    if load==1:
        model.load_weights(savepath)
    else:
        model.fit_generator(gen, steps_per_epoch=30, epochs=50, validation_data = val_gen, validation_steps=20)
        model.save(savepath)
    return model

def mn_vgg(load=1,savepath='mn.h5'):
    gen=data_loader.gen_vgg
    val_gen=data_loader.val_gen_vgg
    model=models.MobileNet_FT()
    print('MobileNet_Fine_Tuned Loaded.')
    if load==1:
        model.load_weights(savepath)
        print('Weights Loaded.')
    else:
        try:
            model.fit_generator(gen, steps_per_epoch=30, epochs=80, validation_data = val_gen, validation_steps=20, callbacks=callbacks())
        except KeyboardInterrupt:
            print('KeyboardInterrupt Received. Weights Saved.')
        finally:
            model.save_weights(savepath)
    return model

	
'''