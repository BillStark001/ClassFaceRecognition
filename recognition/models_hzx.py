from functools import partial

import keras
import tensorflow as tf
from keras import backend as K
from keras import constraints, initializers, regularizers
from keras.engine.topology import Layer
from keras.layers import Activation, Dense, Lambda
from keras.models import Model

import models


"""
    another form of implementation
"""

class AMLogits(Layer):

    def __init__(self, units, kernel_initializer='glorot_uniform',
                kernel_regularizer=None, kernel_constraint=None, **kwargs):
        super(AMLogits, self).__init__(**kwargs)
        self.units = units
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
    
    
    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name='kernel')
        self.kernel = K.l2_normalize(self.kernel, 0)
    

    def call(self, inputs, **kwargs):
        inputs = K.l2_normalize(inputs, -1)
        output = K.dot(inputs, self.kernel)
        return output


    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)


def LMCL(y_true, y_pred, scale, margin):
    y_pred = y_true * (y_pred - margin) + (1 - y_true) * y_pred
    y_pred *= scale
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)

def AMSoftmax(x, units, s=30, m=0.2, name='fc_final'):
    W = Dense(units, use_bias=False, kernel_constraint=constraints.unit_norm(), name=name)
    normalized_x = Lambda(lambda x: K.l2_normalize(x, -1))(x)
    cos_theta = W(normalized_x)
    output = Lambda(lambda x: x * s)(cos_theta) ## NOTE: ......

    def LMCL(y_true, y_pred):
        y_pred -= s * m * y_true  # y_true: one-hot vector
        return K.categorical_crossentropy(y_true, y_pred, from_logits=True)
    
    return output, LMCL


def eval_acc(y_true, y_pred):
    y_pred = K.softmax(y_pred)
    return keras.metrics.categorical_accuracy(y_true, y_pred)

def ModelFRv1(base_model, input_shape, embedding_size=128, training=True, classes=1024, 
        finetune=True, optimizer='adam', scale=30, margin=0.2):
    from keras.regularizers import l2
    from keras.layers import PReLU

    inputs = base_model.input

    x = Dense(512, kernel_regularizer=l2(5e-4), name='fc_1')(base_model.output)
    x = PReLU(name='prelu_1')(x)
    x = Dense(embedding_size, name='fc_embed_%d' % embedding_size)(x)
    x = PReLU(name='prelu_2')(x)

    if training:
        if finetune:
            for layer in base_model.layers:
                layer.trainable = False

        output = AMLogits(classes, name='amlogits_%d' % classes)(x)
        loss = partial(LMCL, scale=scale, margin=margin)
        model = Model(inputs, output)
        model.compile(optimizer=optimizer, loss=loss, metrics=[eval_acc])
    else:
        output = x
        model = Model(inputs, output)

    return model

def ModelFRM1(input_shape, embedding_size=128, training=True, classes=1024, 
            finetune=True, optimizer='adam', scale=30, margin=0.1):
    if training:
        print('FRM1: m=%g s=%g' % (margin, scale))
    base_model = models.MobileNetV2(input_shape=input_shape, include_top=False, expansion=4)
    return ModelFRv1(base_model, input_shape, embedding_size, training, classes, 
            finetune, optimizer, scale, margin)

def get_model(weights='FRM1a-000.h5', embed_size=128, shape=(112, 112, 3)):
    """
        get model for test
        default parameters are used
    """
    model = ModelFRM1(shape, embed_size, training=False)
    model.load_weights(weights, by_name=True)
    return model
