# -*- coding: utf-8 -*-

import math
import keras
from keras import backend as K
from keras.models import Model
import random
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, TensorBoard, ReduceLROnPlateau
from keras.optimizers import SGD, Adam, Nadam
random.seed(20)
import time

from numba import prange #parallise loop
from generator import batch_generator_SRCNN
from functions import load_data, preprocess_data, cPSNR_callback
K.clear_session()
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def compute_steps(data, batch_size, version=1):
    n=0
    if version == 1:
        for LR_QM, _,_,_ in data:
            n += len(LR_QM)
    elif version == 2 or version == 4:
        n = len(data)
    elif version == 3:
        for LR_QM, _,_,_ in data:
            n += len(LR_QM)

    return math.ceil(n/batch_size) 
           

    

        
from sklearn.model_selection import train_test_split 
from model import FSRCNN, FSRCNN_, SRCNN, SRCNNex, SRCNNv2, SRVGG16, FSRCNNv2, SRCNNv3, SRResnet
from model import custom_loss, PSNR, MSE
# load data           
data_test = load_data("data/test.txt")
datas = load_data("data/train.txt")

data_train, data_val = train_test_split(datas, test_size=0.1, shuffle=True, 
                                        random_state=42)

## preprocess data
k=5
scale = 3
resize=True
with_clearance= False
type_clearance= "sum"
version=4
version_val = 1 if version !=4 else 4

all_data = preprocess_data(datas, istrain=True,version=version, k = k)
train = preprocess_data(data_train, istrain=True,version=version, k = k)
val  = preprocess_data(data_val, istrain=True, version=version_val, k = k)

#name_mod = "SRCNNv1_v1_withclearance_concat_multi_loss2.hdf5"
#name_mod = "SRCNNv1_v4_withclearance_concat.hdf5"
name_mod = "SRCNNex_v4_noclearance_k5.hdf5" #"SRVGG16_v4_noclearance_k9.hdf5" # "FSRCNNv2_v4_noclearance_k9.hdf5" #
checkpoint = ModelCheckpoint(name_mod, verbose=2, 
                             monitor='val_cPSNR_'+str(version_val ), save_best_only=True, save_weights_only=False, mode='min')
batch_size=16
#load model and parameters
c = 1 if version != 4 else  k
channel = c*1 if with_clearance == False or type_clearance=="sum" else c*2

#opt = Adam(0.001)
opt = Nadam(0.001)  #SRresnet 0.0005


## multi output
multi_output=True
#model = SRCNN((128*scale, 128*scale ,channel), 1, multi_output) 
model = SRCNNex((128*scale, 128*scale ,channel), 1, multi_output) 
#model = SRCNN((128*scale, 128*scale ,channel), 1, multi_output) 
#model = SRVGG16((128*scale, 128*scale ,channel), 1, multi_output) 
#model = FSRCNNv2((128, 128, channel), 1, multi_output, scale=scale)
model.summary()
if multi_output:
    model.compile(loss=custom_loss, optimizer=opt)
else:
    #model = SRCNN((128*scale, 128*scale ,channel), 1, multi_output) 
    model.compile(loss=MSE, optimizer=opt)


#weights_ = "SRVGG16_v4_withclearance_concat_multi_k9.hdf5"#"FSRCNN_v4_noclearance_k9_.hdf5"#"vgg16_weights_tf_dim_ordering_tf_kernels.h5"#"SRVGG16_v4_withclearance_concat_multi.hdf5"#"SRCNNv1_v4_withclearance_concat_multi_k9.hdf5"#"SRVGG16_v4_withclearance_concat_multi.hdf5"
#model.load_weights(weights_, by_name=True, skip_mismatch=True)

#model.load_weights("vgg16_weights_tf_dim_ordering_tf_kernels.h5", by_name=True, skip_mismatch=True)
#model.load_weights("SRVGG16_200.hdf5", by_name=True, skip_mismatch=True)

#model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5", by_name=True, skip_mismatch=True)
lr_decay = ReduceLROnPlateau(monitor='val_cPSNR_'+str(version_val), factor=0.5, patience=5, 
                                 verbose=1, mode='min', epsilon=0.0001, cooldown=0, min_lr=0.000000001)    


gen_train = batch_generator_SRCNN(train, batch_size=batch_size, with_clearance=with_clearance, type_clearance=type_clearance,
                                    version=version, shuffle=True, scale=scale, data_aug=False, resize=resize, multi_output=multi_output)
gen_val = batch_generator_SRCNN(val, batch_size=batch_size, with_clearance=with_clearance, type_clearance=type_clearance,
                                    version=version, shuffle=False, scale=scale, data_aug=False, resize=resize, multi_output=multi_output)


metrics = cPSNR_callback(val, with_clearance=with_clearance, type_clearance=type_clearance, version=version_val, 
                         scale=scale, resize=resize, name="val_", multi_output=multi_output)

# train model
steps_train=compute_steps(train, batch_size,  version=version)
#steps_val=compute_steps(val, batch_size, version=version)

model.fit_generator(gen_train, steps_per_epoch=steps_train, epochs=150, verbose=1, 
                    callbacks=[metrics, lr_decay, checkpoint])
                    #validation_data=gen_val, validation_steps=steps_val)


# MULTI OUTPUT
# SRCNN v4 0.99143 concat 9 withclear
# SRCNN v4 0.99229 sum  9 withclear
# SRCNN v4 0.98961 no clear 9
# SRCNN v1  0.99745 concat  withclear     
# SRCNN v1  0.99896 sum with clear
# SRCNN v1  0.99625 noclear                    
                    
# SRCNN ex v4 0.988 concat 9 withclear
# SRCNN ex v4 0.990 noclear                                      
# SRVGG16 v4 0.988 concat
# SRVGG16 v4 0.9896 noclear
# FSRCNN v4 0.99134 noclear   elu
# FSRCNNv2 v4 0.99134 noclear   elu
                    
