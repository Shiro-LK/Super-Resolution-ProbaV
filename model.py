# -*- coding: utf-8 -*-
"""
Created on Wed May 22 17:43:33 2019

@author: Shiro
"""

import cv2
import numpy as np
import os
import pandas as pd
import math
from skimage import io 
from skimage.transform import rescale
import skimage
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, Conv2DTranspose, BatchNormalization, Conv2D, Activation, add
from keras.layers import GlobalMaxPooling2D, Flatten, PReLU
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import random
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.optimizers import SGD, Adam, Nadam
random.seed(20)
import time
import numba
from numba import prange #parallise loop

def relu_advanced(x):
    return K.relu(x, max_value=1.0)

def relu_advanced2(x):
    return K.relu(x, max_value=100.0)
  
from keras.losses import mse, binary_crossentropy 

class PRELU(PReLU):
    def __init__(self, **kwargs):
        self.__name__ = "PRELU"
        super(PRELU, self).__init__(**kwargs)
        
        
##################################
#### LOSS FUNCTION ##############
#################################

def MSE(y_true, y_pred):

    diff = K.square(y_pred - y_true)
    error = K.sum(diff, axis=1)

    return K.mean(error)

#def MSE2(y_true, y_pred):
#
#    diff = K.square(y_pred - y_true)
#    error = K.sum(diff, axis=(1,2,3))
#
#    return K.log(K.mean(error))/K.log(100.0)

def PSNR(y_true, y_pred):
    diff = K.square(y_pred - y_true)
    error = K.mean(diff, axis=-1)

    return  -10*K.log(error)/ K.log(10.0)   # first dim is the batch

def custom_loss(y_true, y_pred):
    sr = y_pred[:,:,:,0]
    sr_clear = y_pred[:,:,:,1]
    hr = y_true[:,:,:,0]
    hr_clear = y_true[:,:,:,1]
    dim = K.int_shape(y_pred)[1]
    diff = hr - sr
    denominateur = K.sum(hr_clear, axis=(1,2))

    b = K.sum( diff * hr_clear, axis=(1,2))/denominateur #batchsize dim
    #print(K.int_shape(y_pred), K.int_shape(y_pred[:,:,:]))
    b = K.expand_dims(b, axis=-1)
    b = K.expand_dims(b, axis=-1)
    b = K.repeat_elements(b, dim, axis=1 )
    b = K.repeat_elements(b, dim, axis=-1 )
    
    cMSE = K.sum(np.square( (diff-b)*hr_clear), axis=(1,2))/denominateur

    #cMSE = K.sum(np.square( (diff)*hr_clear), axis=(1,2))/denominateur
    cPSNR = -10*K.log(cMSE)/K.log(10.0)
    loss1 = 46.5 / cPSNR
    ce = binary_crossentropy(K.reshape(hr_clear, (-1, dim*dim)), K.reshape(sr_clear, (-1, dim*dim)))
    
    #print(K.int_shape(loss1), K.int_shape(ce))
    return loss1 + 0.5*ce


###########################
### NETWORK   #############
###########################
      
def SRCNN(input_shape, depth_multiplier=1, multi_output=False):
    """
        # last 1.0237 filter1 | 1.015 filter 7 | 1.021 filter 3 | 1.013 filter 5 | filter 9 

    """
    inputs = Input(input_shape, name="inputs")
    conv1 = Convolution2D(filters=64*depth_multiplier, kernel_size=9, padding="same", name="conv1", activation="relu")(inputs)
    #conv1 = BatchNormalization(name='bn_conv1')(conv1)
    
    mapping = Convolution2D(filters=32*depth_multiplier, kernel_size=1, padding="same", name="mapping", activation="relu")(conv1)
    #mapping = BatchNormalization(name='bn_mapping')(mapping)
    
    if multi_output:
        out = Convolution2D(filters=2, kernel_size=5, padding="same", name="output", activation="sigmoid")(mapping)
    else:
        out = Convolution2D(filters=1, kernel_size=5, padding="same", name="output", activation="sigmoid")(mapping)
    return Model(inputs, out)

def SRCNNv2(input_shape, depth_multiplier=1, multi_output=False):
    """
        conv 9-64 puis 7-64 puis 5-32 puis 7-1 -> 1.006 120 epoch
        conv 9-128 puis 7-64 puis 5-32 puis 7-16 puis 9-1 -> 1.007 130 epoch
    """
    inputs = Input(input_shape, name="inputs")
    conv1 = Convolution2D(filters=64, kernel_size=9, padding="same", name="conv1", activation="relu")(inputs)
    conv2 = Convolution2D(filters=64, kernel_size=7, padding="same", name="conv2", activation="relu")(conv1)
    #conv3 = Convolution2D(filters=64, kernel_size=3, padding="same", name="conv3", activation="relu")(conv2)

    mapping = Convolution2D(filters=32, kernel_size=5, padding="same", name="mapping", activation="relu")(conv2)
    #mapping2 = Convolution2D(filters=16, kernel_size=7, padding="same", name="mapping2", activation="relu")(mapping)
    
    
    if multi_output:
        out = Convolution2D(filters=2, kernel_size=5, padding="same", name="output", activation="sigmoid")(mapping)
    else:
        out = Convolution2D(filters=1, kernel_size=5, padding="same", name="output", activation="sigmoid")(mapping)
    return Model(inputs, out)

def SRCNNv3(input_shape, depth_multiplier=1, multi_output=False):
    """
            
    """
    inputs = Input(input_shape, name="inputs")
    deconv = Conv2DTranspose(filters=32, kernel_size=7, strides=3, padding="same", name="deconv", activation="relu")(inputs)
    conv1 = Convolution2D(filters=64*depth_multiplier, kernel_size=9, padding="same", name="conv1", activation="relu")(deconv)
    #conv1 = BatchNormalization(name='bn_conv1')(conv1)
    
    mapping = Convolution2D(filters=32*depth_multiplier, kernel_size=1, padding="same", name="mapping", activation="relu")(conv1)
    #mapping = BatchNormalization(name='bn_mapping')(mapping)
    #out = Convolution2D(filters=1, kernel_size=5, padding="same", name="output", activation="sigmoid")(mapping)
    out = Convolution2D(filters=1, kernel_size=5, padding="same", name="output", activation=relu_advanced)(mapping)
    
    if multi_output:
        out2 = Convolution2D(filters=1, kernel_size=5, padding="same", name="output2", activation=relu_advanced)(mapping)
        gmp = GlobalMaxPooling2D()(mapping)
        #flatten = Flatten()(gmp)
        out3 = Dense(1, activation=relu_advanced2, name = "output3")(gmp)
    
        return Model(inputs, [out, out2, out3])
    return Model(inputs, out)

def SRCNNex(input_shape, depth_multiplier=1, multi_output=False): 
    inputs = Input(input_shape, name="inputs")
    conv1 = Convolution2D(filters=64, kernel_size=9, padding="same", name="conv1", activation="relu")(inputs)
    mapping = Convolution2D(filters=32, kernel_size=5, padding="same", name="mapping", activation="relu")(conv1)
    
    if multi_output:
        out = Convolution2D(filters=2, kernel_size=5, padding="same", name="output", activation="sigmoid")(mapping)
    else:
        out = Convolution2D(filters=1, kernel_size=5, padding="same", name="output", activation="sigmoid")(mapping)
    return Model(inputs, out)




def SRVGG16(input_shape, depth_multi=1, multi_output=False): # 1.006
    inputs = Input(input_shape, name="inputs")
    conv1 = Convolution2D(filters=64, kernel_size=9, padding="same", name="conv1", activation="relu")(inputs)

    # Block 1
    x = Convolution2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='block1_conv1')(conv1)
    x = Convolution2D(filters=64, kernel_size=(3, 3),activation='relu', padding='same', name='block1_conv2')(x)

    # Block 2
    x = Convolution2D(filters=128, kernel_size=(3, 3), activation='relu',padding='same', name='block2_conv1')(x)
    x = Convolution2D(filters=128, kernel_size=(3, 3), activation='relu',padding='same', name='block2_conv2')(x)

    mapping =  Convolution2D(filters=32, kernel_size=5, padding="same", name="mapping", activation="relu")(x)
    if multi_output:
        out = Convolution2D(filters=2, kernel_size=5, padding="same", name="output", activation="sigmoid")(mapping)
    else:
        out = Convolution2D(filters=1, kernel_size=5, padding="same", name="output", activation="sigmoid")(mapping)
    return Model(inputs, out)


def FSRCNN(input_shape, depth_multi=1, multi_output=False, scale=3):
    inputs = Input(input_shape, name="inputs")
    conv1 = Convolution2D(filters=56, kernel_size=5, padding="same", name="conv1", activation="relu")(inputs)
   
    conv2 = Convolution2D(filters=12, kernel_size=1, padding="same", name="conv2", activation="relu")(conv1)
    conv3 = conv2
    for i in range(4):
        conv3 = Convolution2D(filters=12, kernel_size=3, padding="same", name="conv3_"+str(i), activation="relu")(conv3)
    conv4 = Convolution2D(filters=56, kernel_size=1, padding="same", name="conv4", activation="relu")(conv3)

    #mapping = Convolution2D(filters=32, kernel_size=1, padding="same", name="mapping", activation="relu")(conv4)
    if multi_output:
        out = Conv2DTranspose(filters=2, kernel_size=5, strides=scale, padding="same", name="output", activation="sigmoid")(conv4)
    else:
        out = Conv2DTranspose(filters=1, kernel_size=5, strides=scale, padding="same", name="output", activation="sigmoid")(conv4)

    return Model(inputs, out)    

def FSRCNN_(input_shape, depth_multi=1, multi_output=False, scale=3):
    inputs = Input(input_shape, name="inputs")
    conv1 = Convolution2D(filters=56, kernel_size=5, padding="same", name="conv1")(inputs)
    conv1 = PReLU(name="conv1_prelu")(conv1)
    conv2 = Convolution2D(filters=12, kernel_size=1, padding="same", name="conv2")(conv1)
    conv3 = PReLU(name="conv2_prelu")(conv2)
    for i in range(4):
        conv3 = Convolution2D(filters=12, kernel_size=3, padding="same", name="conv3_"+str(i))(conv3)
        conv3 = PReLU(name="conv3_" +str(i) +"_prelu")(conv3)
    conv4 = Convolution2D(filters=56, kernel_size=1, padding="same", name="conv4")(conv3)
    conv4 = PReLU(name="conv4_prelu")(conv4)

    #mapping = Convolution2D(filters=32, kernel_size=1, padding="same", name="mapping", activation="relu")(conv4)
    if multi_output:
        out = Conv2DTranspose(filters=2, kernel_size=5, strides=scale, padding="same", name="output", activation="sigmoid")(conv4)
    else:
        out = Conv2DTranspose(filters=1, kernel_size=5, strides=scale, padding="same", name="output", activation="sigmoid")(conv4)

    return Model(inputs, out)    



def FSRCNNv2(input_shape, depth_multi=1, multi_output=False, scale=3, finetuned=False):
    inputs = Input(input_shape, name="inputs")
    conv1 = Convolution2D(filters=56, kernel_size=5, padding="same", name="conv1", activation="elu")(inputs)
   
    conv2 = Convolution2D(filters=12, kernel_size=1, padding="same", name="conv2", activation="elu")(conv1)
    conv3 = conv2
    for i in range(4):
        conv3 = Convolution2D(filters=12, kernel_size=3, padding="same", name="conv3_"+str(i), activation="elu")(conv3)
    conv4 = Convolution2D(filters=56, kernel_size=1, padding="same", name="conv4", activation="elu")(conv3)

    #mapping = Convolution2D(filters=32, kernel_size=1, padding="same", name="mapping", activation="relu")(conv4)
    if finetuned == False:
        if multi_output:
            out = Conv2DTranspose(filters=2, kernel_size=5, strides=scale, padding="same", name="output", activation="sigmoid")(conv4)
        else:
            out = Conv2DTranspose(filters=1, kernel_size=5, strides=scale, padding="same", name="output", activation="sigmoid")(conv4)

    else:
        deconv = Conv2DTranspose(filters=32, kernel_size=5, strides=scale, padding="same", name="deconv", activation="elu")(conv4)
        conv1 = Convolution2D(filters=64, kernel_size=9, padding="same", name="conv1", activation="relu")(deconv)
        mapping = Convolution2D(filters=32, kernel_size=5, padding="same", name="mapping", activation="relu")(conv1)
        
        if multi_output:
            out = Convolution2D(filters=2, kernel_size=5, padding="same", name="output", activation="sigmoid")(mapping)
        else:
            out = Convolution2D(filters=1, kernel_size=5, padding="same", name="output", activation="sigmoid")(mapping)
    
#        # Block 1
#        x = Convolution2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='block1_conv1')(deconv)
#        x = Convolution2D(filters=64, kernel_size=(3, 3),activation='relu', padding='same', name='block1_conv2')(x)
#    
#        # Block 2
#        x = Convolution2D(filters=128, kernel_size=(3, 3), activation='relu',padding='same', name='block2_conv1')(x)
#        x = Convolution2D(filters=128, kernel_size=(3, 3), activation='relu',padding='same', name='block2_conv2')(x)
#    
#        mapping =  Convolution2D(filters=32, kernel_size=5, padding="same", name="mapping", activation="relu")(x)
#        if multi_output:
#            out = Convolution2D(filters=2, kernel_size=5, padding="same", name="output", activation="sigmoid")(mapping)
#        else:
#            out = Convolution2D(filters=1, kernel_size=5, padding="same", name="output", activation="sigmoid")(mapping)
    return Model(inputs, out)    
  

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a', padding="same")(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c', padding="same")(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a', padding="same")(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c', padding="same")(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1', padding="same")(input_tensor)
    shortcut = BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x

def SRResnet(input_shape):
    inputs = Input(shape=input_shape, name="inputs")
    bn_axis=3
    x = Conv2D(64, (7, 7),
                      strides=(1, 1),
                      padding='same',
                      kernel_initializer='he_normal',
                      name='conv1')(inputs)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)


    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    
    mapping =  Convolution2D(filters=32, kernel_size=5, padding="same", name="mapping", activation="relu")(x)
    out = Convolution2D(filters=1, kernel_size=5, padding="same", name="output", activation="sigmoid")(mapping)
    return Model(inputs, out)