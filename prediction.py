#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 21:53:21 2019

@author: shiro
"""
from pathlib import Path
import cv2
import numpy as np
import os
import skimage
import keras
from keras import backend as K
from keras.models import Model

random.seed(20)
import time
import gc
import numba
from keras.losses import binary_crossentropy

from numba import prange #parallise loop

from functions import load_data, preprocess_data, cPSNR_callback, score_scene, predict_results, get_validation_results


K.clear_session()

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def custom_loss(y_true, y_pred):
    sr = y_pred[:,:,:,0]
    sr_clear = y_pred[:,:,:,1]
    hr = y_true[:,:,:,0]
    hr_clear = y_true[:,:,:,1]
    dim = K.int_shape(sr)[1]
    diff = hr - sr
    denominateur = K.sum(hr_clear, axis=(1,2))

    b = K.sum( diff * hr_clear, axis=(1,2))/denominateur #batchsize dim

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
keras.losses.custom_loss = custom_loss


from sklearn.model_selection import train_test_split 
from model import FSRCNN, SRCNN, SRCNNex, SRCNNv2, SRVGG16
from keras.models import load_model 
# load data           
data_test = load_data("data/test.txt")
datas = load_data("data/train.txt")

data_train, data_val = train_test_split(datas, test_size=0.1, shuffle=True, random_state=42)

## preprocess data

scale = 3
resize=True
with_clearance=True
type_clearance="concat"
version=4
multi_output=True
k=9
train = preprocess_data(data_train, istrain=True,version=version,k=k)
val  = preprocess_data(data_val, istrain=True, version=1,k=k)

all_train = train = preprocess_data(data_train, istrain=False,version=version,k=k)
test = preprocess_data(data_test, istrain=False, version=version,k=k)
name_model ="SRVGG16_v4_withclearance_concat_multi_k9"#"SRCNNv1"#"FSRCNNv1_withclearance_sum"
model = load_model(name_model+".hdf5")

#get_validation_results(model, train, with_clearance=with_clearance,  multi_output=multi_output,
#                       type_clearance=type_clearance, scale=scale, resize=resize, version=version)
#get_validation_results(model, val4, with_clearance=with_clearance, multi_output=multi_output,
 #                      type_clearance=type_clearance, scale=scale, resize=resize, version=version)
directory = "results_train_" + name_model
predict_results(model, test, directory, with_clearance=with_clearance, multi_output=multi_output,
                       type_clearance=type_clearance, scale=scale, resize=resize, version=version)