#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 16:01:23 2019

@author: shiro
"""
import cv2
import numpy as np
import os
from pathlib import Path

import skimage
import keras
from keras.callbacks import Callback
import random
import math
random.seed(20)
import time
import gc
import numba
from numba import prange #parallise loop
from generator import batch_generator_SRCNN, batch_generator_SRCNN_validation, load_image2D, batch_generator_SRCNN_test
## Load all data 
def load_data(filename):
    with open(filename, "r") as f:
        temp = [line.replace("\\","/").split() for line in f]
    
    return temp

def preprocess_data(data, istrain=True,  version=1, k=9):
    """
        @istrain : train/validation  data => True  | testing data => False
        
    """
    data_preprocess = []
    if version == 1 or version == 2 or version==3 or version==4: 
        """
            version 1 : take independently max clearance img
            version 2 : for training, averagin the max clearance and image before feeding network
        """
        for path, v in data:
            norm = float(v)
            
                
            if istrain :
                LR_QM, SM, HR = get_scene(path, istrain,  version,k)
                data_preprocess.append([LR_QM, norm, SM, HR])
            else: # test data
                LR_QM, SM = get_scene(path, istrain, version,k)
                data_preprocess.append([LR_QM, norm, SM])
                

    return data_preprocess

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

## load one scene data
def get_scene(path, istrain=True, version=1, k=9):
    """
        version 1 and version 2 : return all images with max clearance value
        version 3 : istrain = True return all images
        version 4 : take top k images with max clearance (merge channel images)
    
    """
    names = ['LR000.png', 'LR001.png', 'LR002.png', 'LR003.png', 'LR004.png', 'LR005.png', 
         'LR006.png', 'LR007.png', 'LR008.png', 'LR009.png', 'LR010.png', 'LR011.png', 
         'LR012.png', 'LR013.png', 'LR014.png', 'LR015.png', 'LR016.png', 'LR017.png', 
         'LR018.png', 'LR019.png', 'LR020.png', 'LR021.png', 'LR022.png', 'LR023.png', 
         'LR024.png', 'LR025.png', 'LR026.png', 'LR027.png', 'LR028.png', 'LR029.png', 
         'LR030.png', 'LR031.png', 'LR032.png', 'LR033.png', 'LR034.png',  
         'QM000.png', 'QM001.png', 'QM002.png', 'QM003.png', 
         'QM004.png', 'QM005.png', 'QM006.png', 'QM007.png', 'QM008.png', 'QM009.png', 
         'QM010.png', 'QM011.png', 'QM012.png', 'QM013.png', 'QM014.png', 'QM015.png', 
         'QM016.png', 'QM017.png', 'QM018.png', 'QM019.png', 'QM020.png', 'QM021.png', 
         'QM022.png', 'QM023.png', 'QM024.png', 'QM025.png', 'QM026.png', 'QM027.png', 
         'QM028.png', 'QM029.png', 'QM030.png', 'QM031.png', 'QM032.png', 'QM033.png', 
         'QM034.png', 'HR.png', 'SM.png']

    if path is not None:
        LR_QM = []
        clearance = []
        max_clearance = 0
        
        if istrain:
            HR = os.path.join(path, names[-2])
        SM = os.path.join(path, names[-1])
        
        if version == 1 or version == 2:
            """ 
                version 1 and version 2 are different in the function batch generator
                add only image with max number of clear pixel
            """
            for ids, lr in enumerate(names[0:35]):
                lr_path = os.path.join(path, lr)
                if os.path.isfile(lr_path):
                    qm_path = os.path.join(path, names[35+ids])
                    

                    clearMap = skimage.img_as_float64( cv2.imread(qm_path, -1) )
                    clearValue = clearMap.sum()
                    clearance.append(clearValue)
                    max_clearance = max(max_clearance, clearValue)
                else:
                    break
                
            for ids, cl in enumerate(clearance):
                if cl == max_clearance:
                    lr_path = os.path.join(path, names[ids])
                    qm_path = os.path.join(path, names[35+ids])
                    LR_QM.append((lr_path, qm_path))
                
        elif version == 3:
            """
                add all images for training / for validation/testing max number of clear pixel or average depending of batch generator function
            """

            for ids, lr in enumerate(names[0:35]):
                lr_path = os.path.join(path, lr)
                if os.path.isfile(lr_path):
                    qm_path = os.path.join(path, names[35+ids])
                    LR_QM.append((lr_path, qm_path))
                        
        elif version == 4:
            """
                take for training and testing top k = 9  images with the best max clearance
            """

            for ids, lr in enumerate(names[0:35]):
                lr_path = os.path.join(path, lr)
                if os.path.isfile(lr_path):
                    qm_path = os.path.join(path, names[35+ids])
                    

                    clearMap = skimage.img_as_float64( cv2.imread(qm_path, -1) )
                    clearValue = clearMap.sum()
                    clearance.append(clearValue)
 
                else:
                    break
            clearance = np.array(clearance)
            ids_sort = np.argsort(clearance)[::-1]
            while len(ids_sort) < k:
                ids_sort = np.concatenate((ids_sort, ids_sort), axis=-1)
            for ids in ids_sort[:k]:

                lr_path = os.path.join(path, names[ids])
                qm_path = os.path.join(path, names[35+ids])
                LR_QM.append((lr_path, qm_path))

            
        if istrain:
            return [LR_QM, SM, HR]
        else:
            return [LR_QM, SM]
        
## METRIC FUNCTION FOR ONE SCENE
@numba.autojit
def score_scene(sr, hr, clearhr, norm, num_crop=6):

    max_x, max_y = np.array(hr.shape) - num_crop
    sr_ = sr[num_crop//2:-num_crop//2, num_crop//2:-num_crop//2]
    
    np.place(clearhr, clearhr==0, np.nan)
    
    zSR = np.zeros((num_crop + 1, num_crop + 1), np.float64)
    for x_off in prange(0, num_crop+1):
        for y_off in prange(0, num_crop+1):
            
            clearHR_ = clearhr[x_off : x_off + max_x, y_off : y_off + max_y]

            hr_ = hr[x_off:x_off + max_x, y_off:y_off + max_y]

            diff = (hr_- sr_)* clearHR_

            b = np.nanmean(diff)


            ## compute cMSE
            cMSE = np.nanmean( (diff-b)**2) 

            cPSNR = -10.0*np.log10(cMSE)
   
            zSR[x_off, y_off] = norm/cPSNR

    return zSR.min()


class cPSNR_callback(Callback):
    def __init__(self, data, with_clearance=True, type_clearance="sum", version=1, scale=3, resize=True, batch_size=8, name="val_", multi_output=False):
        #print("shape callback validation data:", validation_data.shape)
        self.data = data
        self.with_clearance= with_clearance
        self.type_clearance = type_clearance
        self.version = version
        self.scale = scale
        self.resize= resize
        self.batch_size = batch_size
        self.name = name
        self.multi_output=multi_output
    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}): 
        """
            metric computed on the error between the images with max clearance values 
        
        """
        generator = batch_generator_SRCNN_validation(data = self.data, with_clearance = self.with_clearance,
                                                     type_clearance=self.type_clearance, version=self.version, 
                                                     scale=self.scale, resize=self.resize)
        beg = time.time()
        if self.version != 4:
            cPSNR = np.zeros( (len(self.data),) ) # average all single super resolution
            
            loss = np.zeros( (len(self.data),))
            
            cPSNR_average_before = np.zeros( (len(self.data),) ) # average multi image before applying single super resolution
            for ids, (X, HR, clearHR, N) in enumerate(generator):
                # remove last dim ( x, y , 1)
                HR = HR[:,:,0]
                clearHR = clearHR[:,:,0]
                
                if self.multi_output:
                    img_predict = self.model.predict(X, batch_size=self.batch_size)[:,:,:,0]
                    average_img = img_predict.mean(axis=0)
                    average_img = average_img[:,:]
                else:
                    img_predict = self.model.predict(X, batch_size=self.batch_size)
                    average_img = img_predict.mean(axis=0)
                    average_img = average_img[:,:,0]
                #print(average_img.shape, np.sum(np.square(average_img-HR) ).shape)
                cPSNR[ids] = score_scene(average_img, HR, clearHR, N, num_crop=6)
                loss[ids] = np.mean(np.square(average_img-HR) )#-10*np.log10(np.mean(np.square(average_img-HR) ))

                ## average multi img before super resolution
                x = np.expand_dims(X.mean(axis=0), axis=0)
                if self.multi_output:
                    img_predict2 = self.model.predict(x, batch_size=self.batch_size)[:,:,:,0]
                    cPSNR_average_before[ids] = score_scene(img_predict2[0,:,:], HR, clearHR, N, num_crop=6)
                else:
                    img_predict2 = self.model.predict(x, batch_size=self.batch_size)

                    cPSNR_average_before[ids] = score_scene(img_predict2[0,:,:,0], HR, clearHR, N, num_crop=6)
            
            
            logs[self.name+'cPSNR_'+str(self.version)] = cPSNR.mean()
            logs[self.name+'loss_mean_'+str(self.version)] = loss.mean()
            logs[self.name+'cPSNR_before_'+str(self.version)] = cPSNR_average_before.mean()
            print('\r - %s: %s | - %s: %s | - %s : %s | - time : %s sec' % ( self.name+'cPSNR_'+str(self.version), str(round(logs[self.name+'cPSNR_'+str(self.version)],5) ) , self.name+'cPSNR_before_'+str(self.version), 
                  str(round(logs[self.name+'cPSNR_before_'+str(self.version)] ,5) ),  self.name+'loss_mean_'+str(self.version), str(round(logs[self.name+'loss_mean_'+str(self.version)] ,5) ), str(time.time()-beg)), end=10*' '+'\n')
        else:
            ## version 4 is image with concatenante channel like (384 384 k) = input
            cPSNR = np.zeros( (len(self.data),) ) # average all single super resolution
            loss = np.zeros( (len(self.data),))
            #img by img
            for ids, (X, HR, clearHR, N) in enumerate(generator):
                # remove last dim ( x, y , 1)
                #print(X.shape, HR.shape)
                HR = HR[:,:,0]
                clearHR = clearHR[:,:,0]
                
                
                if self.multi_output:
                    img_predict = self.model.predict(X, batch_size=self.batch_size)[:,:,:,0]
                    average_img = img_predict.mean(axis=0)
                    average_img = average_img[:,:]
                else:
                    img_predict = self.model.predict(X, batch_size=self.batch_size)
                #print(img_predict.shape)
 
                    average_img = img_predict.mean(axis=0)
                    average_img = average_img[:,:,0]
                loss[ids] = -10*np.log10( np.mean(np.square(average_img-HR) ) )
                cPSNR[ids] = score_scene(average_img, HR, clearHR, N, num_crop=6)
                
                
            
            #print(cPSNR)
            logs[self.name+'cPSNR_'+str(self.version)] = cPSNR.mean()
            logs[self.name+'loss_mean_'+str(self.version)] = loss.mean()
            print('\r - %s: %s | - %s : %s | - time : %s sec' % ( self.name+'cPSNR_'+str(self.version), str(round(logs[self.name+'cPSNR_'+str(self.version)],5) ) , self.name+'loss_mean_'+str(self.version), 
                  str(round(logs[self.name+'loss_mean_'+str(self.version)] ,5) ), str(time.time()-beg)), end=10*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
    

## VALIDATION AND TEST FUNCTION 
        
def save_prediction(pred, names, directory):
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)
    #io.use_plugin('freeimage')
    p = os.path.join(directory , names+'.png')

    im = skimage.img_as_uint(pred)
    #io.imsave(arr=im, fname= p, plugin="freeimage")
    cv2.imwrite(p, im,  [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
def get_validation_results(model, data, with_clearance=True, type_clearance="sum", batch_size=4, 
                           scale=3, resize=True, version=1, multi_output=False):
    generator = batch_generator_SRCNN_validation(data = data, with_clearance = with_clearance,
                                                     type_clearance=type_clearance, version=version, 
                                                     scale=scale, resize=resize)
    beg = time.time()
    if version != 4:
        cPSNR = np.zeros( (len(data),) ) # average all single super resolution
        for ids, (X, HR, clearHR, N) in enumerate(generator):
            # remove last dim ( x, y , 1)
            HR = HR[:,:,0]
            clearHR = clearHR[:,:,0]
            
            if multi_output:
                img_predict = model.predict(X, batch_size=batch_size)[:,:,:,0]
                average_img = img_predict.mean(axis=0)
            else:
                img_predict = model.predict(X, batch_size=batch_size)

 
                average_img = img_predict.mean(axis=0)
    
                average_img = average_img[:,:,0]
            cPSNR[ids] = score_scene(average_img, HR, clearHR, N, num_crop=6)
            
           
        
        results = cPSNR.mean()

        print('\r - cPSNR: %s |  - time : %s sec' % ( str(round(results,5) ) ,   str(time.time()-beg)), end=10*' '+'\n')
    else:
        ## version 4 is image with concatenante channel like (384 384 k) = input
        cPSNR = np.zeros( (len(data),) ) # average all single super resolution
        for ids, (X, HR, clearHR, N) in enumerate(generator):
            # remove last dim ( x, y , 1)
            #print(X.shape, HR.shape)
            HR = HR[:,:,0]
            clearHR = clearHR[:,:,0]
            
            
            if multi_output:
                img_predict = model.predict(X, batch_size=batch_size)[:,:,:,0]
                average_img = img_predict.mean(axis=0)
            else:
                img_predict = model.predict(X, batch_size=batch_size)

 
                average_img = img_predict.mean(axis=0)
    
                average_img = average_img[:,:,0]
            cPSNR[ids] = score_scene(average_img, HR, clearHR, N, num_crop=6)
            
           
        res = cPSNR.mean()
        del generator
        gc.collect()
        #logs['cPSNR_before'] = cPSNR_average_before.mean()
        print('\r - cPSNR: %s | - time : %s sec' % ( str(round(res,5) ) ,    str(time.time()-beg)), end=10*' '+'\n') 
    return

## predict on test data and save them in directory
def predict_results(model, data, directory, with_clearance=True, batch_size=4, type_clearance="sum", scale=3, resize=True, version=1,  multi_output=False):
    generator = batch_generator_SRCNN_test(data = data, with_clearance = with_clearance,
                                                     type_clearance=type_clearance, version=version, 
                                                     scale=scale, resize=resize)
    if version != 4:
        for ids, (X, clearHR, N) in enumerate(generator):
            p = Path(data[ids][2])
            names = p.parts[-2]
            # remove last dim ( x, y , 1)
            clearHR = clearHR[:,:,0]
            
            if multi_output:
                img_predict = model.predict(X, batch_size=batch_size)[:,:,:,0]
                average_img = img_predict.mean(axis=0)
    
            else:
                img_predict = model.predict(X, batch_size=batch_size)

 
                average_img = img_predict.mean(axis=0)
    
                average_img = average_img[:,:,0]
            save_prediction(average_img, names, directory)
    else:
        ## version 4 is image with concatenante channel like (384 384 k) = input
        
        for ids, (X, clearHR, N) in enumerate(generator):
            # remove last dim ( x, y , 1)
            p = Path(data[ids][2])
            names = p.parts[-2]
            clearHR = clearHR[:,:,0]
            
            if multi_output:
                img_predict = model.predict(X, batch_size=batch_size)[:,:,:,0]
                average_img = img_predict.mean(axis=0)
            else:
                img_predict = model.predict(X, batch_size=batch_size)
            #print(img_predict.shape)
 
                average_img = img_predict.mean(axis=0)
                average_img = average_img[:,:,0]
            save_prediction(average_img, names, directory)
        del generator
        gc.collect()
       
    return


        
        
      
         