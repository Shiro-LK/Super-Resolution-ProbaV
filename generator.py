#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 22:04:30 2019

@author: shiro
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
from keras.layers import Input, Dense, Convolution2D, Conv2DTranspose
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import random
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.optimizers import SGD, Adam, Nadam
random.seed(20)
import time
import gc
import numba
from numba import prange #parallise loop
def load_image2D(path, expand=False):
    img = skimage.img_as_float64( cv2.imread(path, -1) )
    #height, width = img.shape
    #if scale > 1:
    #    img = cv2.resize(img,  (height*scale, width*scale), interpolation = cv2.INTER_CUBIC)
    if expand:
        img = np.expand_dims(img, axis=2)
    return img

def change_contrast(img, img2, img3, img4):
    alpha = np.random.uniform(0.9,1.0)
    beta = np.random.uniform(0,0.1)
    img = alpha * img + beta
    img[img<0] = 0.0
    img[img>1.0] = 1.0
    
    if img2 is not None:
        img2 = alpha * img2 + beta
        img2[img2<0] = 0.0
        img2[img2>1.0] = 1.0
        
    img3 = alpha * img3 + beta
    img3[img3<0] = 0.0
    img3[img3>1.0] = 1.0
    
    if img4 is not None:
        img4 = alpha * img4 + beta
        img4[img4<0] = 0.0
        img4[img4>1.0] = 1.0
    
    return img, img2, img3, img4
def data_augmentation(x, clear_x, y, clear_y):
    """
        apply the same transformation for each of these 4 images
        clear_x, and clear_y can be None
    """
    hasAug = np.random.uniform()
    if hasAug <=0.5: # no augmentation
        
        return x, clear_x, y, clear_y
    else:
        # check contrast
#        contr = np.random.uniform()
#        if contr < 0.4:
#            x, clear_x, y, clear_y = change_contrast(x, clear_x, y, clear_y) 
        # check if flip flap occured
        transf = np.random.uniform(0,1.0, 2) # flip/flap/rotation
        ## flip flap
        if transf[0] <= 0.5:
            x = cv2.flip(x, 0)
            y = cv2.flip(y, 0)
            if clear_x is not None:
                clear_x = cv2.flip(clear_x, 0)
                
            if clear_y is not None:
                clear_y = cv2.flip(clear_y, 0)
        else:
            x = cv2.flip(x, 1)
            y = cv2.flip(y, 1)
            if clear_x is not None:
                clear_x = cv2.flip(clear_x, 1)
                
            if clear_y is not None:
                clear_y = cv2.flip(clear_y, 1)
            
        # rotation
        if transf[1] < 0.25: # 90 degre
            x = cv2.transpose(x)
            x = cv2.flip(x, 1)
            
            y = cv2.transpose(y)
            y = cv2.flip(y, 1)
            if clear_x is not None:
                clear_x = cv2.transpose(clear_x)
                clear_x = cv2.flip(clear_x, 1)
                
            if clear_y is not None:
                clear_y = cv2.transpose(clear_y)
                clear_y = cv2.flip(clear_y, 1)
                
        elif transf[1] < 0.5:#rotation -90
            x = cv2.transpose(x)
            x = cv2.flip(x, 0)
            
            y = cv2.transpose(y)
            y = cv2.flip(y, 0)
            if clear_x is not None:
                clear_x = cv2.transpose(clear_x)
                clear_x = cv2.flip(clear_x, 0)
                
            if clear_y is not None:
                clear_y = cv2.transpose(clear_y)
                clear_y = cv2.flip(clear_y, 0)
                
        elif transf[1] < 0.75:
            x = cv2.flip(x, -1)
            y = cv2.flip(y, -1)
            if clear_x is not None:
                clear_x = cv2.flip(clear_x, -1)
                
            if clear_y is not None:
                clear_y = cv2.flip(clear_y, -1)
                
                
        return x, clear_x, y, clear_y
def batch_transform(img1, img2=None, type_clearance="sum", version=1):
    """
        transform array of img into batch of array of img given the type of clearance we want to apply
    """
    if img2 is None:
        if version == 4:
            return img1
        return np.expand_dims(img1, axis=-1)
    else:
        if type_clearance == "sum":
            im = img1 * img2
            if version == 4:
                return im
            return np.expand_dims(im, axis=-1)
        else:
            if version != 4:
                img1 = np.expand_dims(img1, axis=-1)
                img2 = np.expand_dims(img2, axis=-1)
            return np.concatenate((img1, img2), axis=-1)
def batch_resize(img1, img2=None, dim=(384, 384), type_clearance="sum", version=1):
    """
        @img1, img2 : can be a list of img or an array of img (num_img, n, n)
        if version is 4 , then img1 and img2 are dim (n, n, k)
    """
    if type_clearance not in ["sum", "concat"]:
        print("type clearance is 'sum' or 'concat' only, check your parameters !")
        raise("Error in parameters")
    
    if version != 4:
        resized = []

    
        for i in range(len(img1)):
            if img2 is None:
                resized.append(cv2.resize(img1[i], dim, interpolation=cv2.INTER_CUBIC))
            else:
                if type_clearance == "sum":
                    resized.append(cv2.resize(img1[i]*img2[i], dim, interpolation=cv2.INTER_CUBIC))
                    
                elif type_clearance == "concat":
                    im1 = cv2.resize(img1[i], dim, interpolation=cv2.INTER_CUBIC)
                    im2 = cv2.resize(img2[i], dim, interpolation=cv2.INTER_CUBIC)
                    im1 = np.expand_dims(im1, axis=-1)
                    im2 = np.expand_dims(im2, axis=-1)
                    resized.append( np.concatenate((im1, im2), axis=-1) )
                    
        resized = np.array(resized)
        
        if type_clearance == 'sum':
            resized = np.expand_dims(resized, axis=-1)
    else:
        if img2 is None:
            resized = cv2.resize(img1, dim, interpolation=cv2.INTER_CUBIC)
            
        else:
            if type_clearance == "sum":
                resized = cv2.resize(img1*img2, dim, interpolation=cv2.INTER_CUBIC)
            elif type_clearance == "concat":
                im = np.concatenate( (img1, img2), axis = -1)
                resized = cv2.resize(im, dim, interpolation=cv2.INTER_CUBIC )
            
        
    return resized

def batch_generator_SRCNN(data, batch_size=16, with_clearance=True, type_clearance="sum", version=1, shuffle=True, scale=3, data_aug=False, resize=True, multi_output=False):
    """
    with_clearance : it means that our input is multiply by the clearance map
    type_clearance : "sum" or "concat"
    version 1: For training, each image are considered independant (we do not take in account the multi image)
               For testing, (istrain=False) we return a set of image corresponding to one batch (we do not care about the batchsize), no shuffle
    
    version 2 : For training, each image are average with the max clearance (that reduce the number of iteration)
    
    """
    if type_clearance not in ["sum", "concat"]:
        print("type clearance is 'sum' or 'concat' only, check your parameters !")
        raise("Error in parameters")
        
    if version == 1 or version==3  :
        if version == 1:
            print("preprocessing data of all clearance max images \n")
        elif version == 3:
            print("preprocessing data of all data images \n")
        X = []
        Y = []
        Y_clear = []
        N = []
        for LR_QM, norm, SM, HR in data:
            for lrqm in LR_QM:
                X.append(lrqm)
                Y.append(HR)
                Y_clear.append(SM)
                N.append(norm)
                
        
        
        n = len(X)
        steps = math.ceil(n/batch_size)
        while True:
            if shuffle:
                mapPos = list(zip(X, Y, Y_clear, N))
                random.shuffle(mapPos)
                X, Y, Y_clear, N = zip(*mapPos)
                del mapPos
            
            for i in range(steps):
                if with_clearance:
                    batch_X_clear = []
                batch_X_img = []
                
                
                batch_Y_img = []
                batch_Y_clear_img = []
                batch_N = []
                
                begin = i*batch_size
                end = begin + batch_size
                
                if end >= n:
                    end = n

   
                for idx in range(begin, end) :
                    
                     ## load Y data
                    img_y = load_image2D(Y[idx])
                    img_y_clear = load_image2D(Y_clear[idx])
                    
                    
                    ## load X data
                    img_x = load_image2D(X[idx][0])
                    
                    if with_clearance:
                        clear = load_image2D(X[idx][1])
                        if data_aug:
                            img_x, clear, img_y, img_y_clear =  data_augmentation(img_x, clear, img_y, img_y_clear)
                        batch_X_clear.append(clear)
                    else:
                        if data_aug:
                            img_x, _, img_y, _ =  data_augmentation(img_x, None, img_y, None)
                    batch_Y_img.append( np.expand_dims(img_y, axis=-1)  )
                    batch_X_img.append(img_x)
                    
                    batch_Y_clear_img.append( np.expand_dims(img_y_clear, axis=-1) )
                    batch_N.append(N[idx])
                    
                    
                    
                if with_clearance:    
                    batch_X_clear = np.array(batch_X_clear)
  
                batch_X_img   = np.array(batch_X_img)
                batch_Y_img   = np.array(batch_Y_img)
                batch_Y_clear_img   = np.array(batch_Y_clear_img)
                batch_N   = np.array(batch_N).reshape((-1,1))

                batch, h, w = batch_X_img.shape
                if h*scale != batch_Y_img.shape[1] or batch_Y_img.shape[1] != w*scale: 
                    print("error in upscaling :\n upscale : {0} \n Y : {1}".format(h*scale, batch_Y_img.shape[1]))
                    raise("Error")
                    
                if len(batch_Y_img) != len(batch_X_img) :
                    raise("error in dimension of N, or batch_X_img or batch_Y_img")
                    
                if with_clearance:
                    if resize:
                        batch_X = batch_resize(batch_X_img , batch_X_clear, (h*scale, w*scale), type_clearance = type_clearance)
                    else:
                        batch_X = batch_transform(batch_X_img , batch_X_clear, type_clearance = type_clearance)
                    
                    
                    #print(batch_X.shape, batch_X_clear.shape, batch_X_img.shape, batch_X_clear.dtype, batch_X_img.dtype )
                    
                    #yield batch_X, batch_Y_img
                else:
                    if resize:
                        batch_X = batch_resize(batch_X_img, None, (h*scale, w*scale))
                    else:
                        batch_X = batch_transform(batch_X_img , None,  type_clearance = type_clearance)
                #print(batch_X.shape, batch_Y_img.shape)
                if multi_output:
                    yield  batch_X, np.concatenate((batch_Y_img, batch_Y_clear_img), axis=-1)
                else:
                    yield  batch_X, batch_Y_img
    elif version == 2 :
        if version == 2:
            print("preprocessing data version 2, average imageset\n")

        ## load all image from same imageset (with max clearance) and average them 
        ## before apply transformation and feeding NN
        X = []
        Y = []
        X_clear = []
        Y_clear = []
        N = []
        for LR_QM, norm, SM, HR in data:
            x_average = []
            x_clear_average = []
            for lrqm in LR_QM:
                x_average.append( load_image2D(lrqm[0]) )
                x_clear_average.append( load_image2D(lrqm[1]) )
                
            x_average = np.array(x_average)
            x_clear_average = np.array(x_clear_average)
            
            X.append( x_average.mean(axis=0) )
            X_clear.append( x_clear_average.mean(axis=0) )
            
            Y.append( load_image2D(HR) )
            Y_clear.append( load_image2D(SM) )
            N.append(norm)
            
        
        
        n = len(X)
        steps = math.ceil(n/batch_size)
        while True:
            if shuffle:
                mapPos = list(zip(X, X_clear, Y, Y_clear, N))
                random.shuffle(mapPos)
                X, X_clear, Y, Y_clear, N = zip(*mapPos)
                del mapPos
            
            for i in range(steps):
                if with_clearance:
                    batch_X_clear = []
                batch_X_img = []
                
                
                batch_Y_img = []
                batch_Y_clear_img = []
                batch_N = []
                
                begin = i*batch_size
                end = begin + batch_size
                
                if end >= n:
                    end = n

   
                for idx in range(begin, end) :
                    
                    
                    if with_clearance:

                        if data_aug:
                            img_x, clear, img_y, img_y_clear =  data_augmentation(X[idx], X_clear[idx], Y[idx], Y_clear[idx])
                        else:
                            img_x = X[idx]
                            img_y = Y[idx]
                            img_y_clear = Y_clear[idx]
                            clear = X_clear[idx]
                        batch_X_clear.append(clear)
                    else:
                        if data_aug:
                            img_x, _, img_y, _ =  data_augmentation(img_x, None, img_y, None)
                        else:
                            img_x = X[idx]
                            img_y = Y[idx]
                            
                    batch_Y_img.append( img_y  )
                    batch_X_img.append( img_x )
                    batch_Y_clear_img.append( img_y_clear )
                    batch_N.append(N[idx])
                    
                if with_clearance:    
                    batch_X_clear = np.array(batch_X_clear)
  
                batch_X_img   = np.array(batch_X_img)
                batch_Y_img   = np.expand_dims( np.array(batch_Y_img), axis=-1)
                batch_Y_clear_img = np.expand_dims( np.array(batch_Y_clear_img), axis=-1)
                batch_N = np.array(batch_N)
                
                batch, h, w = batch_X_img.shape
                if h*scale != batch_Y_img.shape[1] or batch_Y_img.shape[1] != w*scale: 
                    print("error in upscaling :\n upscale : {0} \n Y : {1}".format(h*scale, batch_Y_img.shape[1]))
                    raise("Error")
                    
                
                if with_clearance:
                    batch_X_clear = np.array(batch_X_clear)
                    if resize:
                        batch_X = batch_resize(batch_X_img , batch_X_clear, (h*scale, w*scale), type_clearance = type_clearance)
                    else:
                        batch_X = batch_transform(batch_X_img , batch_X_clear, type_clearance = type_clearance)
                    
                else:
                   
                    if resize:
                        batch_X = batch_resize(batch_X_img, None, (h*scale, w*scale))
                    else:
                        batch_X = batch_transform(batch_X_img , None,  type_clearance = type_clearance)
                #print(batch_X.shape, batch_Y_img.shape)
                if multi_output:
                    yield  batch_X, np.concatenate((batch_Y_img, batch_Y_clear_img), axis=-1)
                else:
                    yield  batch_X, batch_Y_img
                
    elif version == 4:
        print("preprocessing data version 4, concat imageset top k\n")
        ## load all image from same imageset (with max clearance) and average them 
        ## before apply transformation and feeding NN
        X = []
        Y = []
        X_clear = []
        Y_clear = []
        N = []
        for LR_QM, norm, SM, HR in data:
            x_concat = []
            x_clear_concat = []
            for lrqm in LR_QM:
                x_concat.append( load_image2D(lrqm[0]) )
                x_clear_concat.append( load_image2D(lrqm[1]) )
                
            x_concat = np.array(x_concat) # dim k 128 128
            x_clear_concat = np.array(x_clear_concat)
            
            x_concat = np.moveaxis(x_concat, 0, -1)
            x_clear_concat = np.moveaxis(x_clear_concat, 0, -1) # dim 128 128 k
            
            X.append( x_concat )
            X_clear.append( x_clear_concat )
            
            Y.append( load_image2D(HR) )
            Y_clear.append( load_image2D(SM) )
            N.append(norm)
            
        
        
        n = len(X)
        steps = math.ceil(n/batch_size)
        while True:
            if shuffle:
                mapPos = list(zip(X, X_clear, Y, Y_clear, N))
                random.shuffle(mapPos)
                X, X_clear, Y, Y_clear, N = zip(*mapPos)
                del mapPos
            
            for i in range(steps):
                if with_clearance:
                    batch_X_clear = []
                batch_X_img = []
                
                
                batch_Y_img = []
                batch_Y_clear_img = []
                batch_N = []
                
                begin = i*batch_size
                end = begin + batch_size
                
                if end >= n:
                    end = n

   
                for idx in range(begin, end) :
                    
                    
                    if with_clearance:
                        # data augmentation
                        if data_aug:
                            img_x, clear, img_y, img_y_clear =  data_augmentation(X[idx], X_clear[idx], Y[idx], Y_clear[idx])
                        else:
                            img_x = X[idx]
                            img_y = Y[idx]
                            img_y_clear = Y_clear[idx]
                            clear = X_clear[idx]
                        # resize the image
                        h,w,depth = img_x.shape
                        if resize:
                            img_x = batch_resize(img_x , clear, (h*scale, w*scale), type_clearance = type_clearance, version=version)
                        else:
                            img_x = batch_transform(img_x , clear, type_clearance = type_clearance, version=version)
                            
                        #batch_X_clear.append(clear)
                    else:
                        if data_aug:
                            img_x, _, img_y, img_y_clear =  data_augmentation(img_x, None, img_y, Y_clear[idx])
                        else:
                            img_x = X[idx]
                            img_y = Y[idx]
                            img_y_clear = Y_clear[idx]
                        h,w,depth = img_x.shape
                        if resize:
                            img_x  = batch_resize(img_x , None, (h*scale, w*scale), version=version)
                        else:
                            img_x  = batch_transform(img_x  , None,  type_clearance = type_clearance, version=version)
                    
                    #print("batch shape :", img_x.shape)
                    
                    batch_Y_img.append( img_y  )
                    batch_X_img.append( img_x )
                    batch_Y_clear_img.append( img_y_clear )
                    batch_N.append(N[idx])
                    
                batch_X_img   = np.array(batch_X_img)
                batch_Y_img   = np.expand_dims( np.array(batch_Y_img), axis=-1)
                batch_Y_clear_img   = np.expand_dims( np.array(batch_Y_clear_img), axis=-1)
                batch_N = np.array(batch_N)

                    
                
#                if with_clearance:
#                    #batch_X_clear = np.array(batch_X_clear)
#                    if resize:
#                        batch_X = batch_resize(batch_X_img , batch_X_clear, (h*scale, w*scale), type_clearance = type_clearance, version=version)
#                    else:
#                        batch_X = batch_transform(batch_X_img , batch_X_clear, type_clearance = type_clearance, version=version)
#
#                else:
#                   
#                    if resize:
#                        batch_X = batch_resize(batch_X_img, None, (h*scale, w*scale), version=version)
#                    else:
#                        batch_X = batch_transform(batch_X_img , None,  type_clearance = type_clearance, version=version)
#                #print(batch_X.shape, batch_Y_img.shape)
                #print(batch_X_img.shape, batch_Y_img.shape)
                if multi_output:
                    yield  batch_X_img, np.concatenate((batch_Y_img, batch_Y_clear_img), axis=-1)
                else:
                    yield  batch_X_img, batch_Y_img
                
                
def batch_generator_SRCNN_validation(data, with_clearance=True, type_clearance="sum", version=1, scale=3, resize=True):
    """
        version 1 & 2 & 3: 
        @return : upscale batch composed of images from the same set with its target features
                    (batch of images , targets, targets_clear, Norm). 
                    the batch of images is of dimension (number max clearance, n, n, 1)
        version 4: 
        @return : upscale batch composed of images from the same set with its target features
                    (batch of images , targets, targets_clear, Norm). 
                    the batch of images is of dimension (1, n, n, k)
    """
    

    if version == 1 or version == 2 or version==3: 
        for LR_QM, norm, SM, HR in data:
            X = []
            X_clear = []
            Y = load_image2D(HR, expand=True)
            Y_clear = load_image2D(SM, expand=True)
            N = norm
            
            for (lr, qm) in LR_QM:
                img = load_image2D(lr)
                h,w = img.shape
                if with_clearance:
                    cl = load_image2D(qm)
    
                    X_clear.append(cl) 
                X.append(img)
            h,w = X[0].shape
            X_clear = np.array(X_clear)
            
            if resize:
                if with_clearance:
                    x = batch_resize(X, X_clear, (h*scale, w*scale), type_clearance=type_clearance ) 
                else:
                    x = batch_resize(X, None, (h*scale, w*scale) ) 
            else:
                X = np.array(X)
                
                if with_clearance:
                    
                    x = batch_transform(X, X_clear, type_clearance=type_clearance)
                else:
                    x = batch_transform(X, None, type_clearance=type_clearance)
            #print(x.shape)
            yield x, Y, Y_clear, N
            
    elif version == 4:
        for LR_QM, norm, SM, HR in data:

            X = []
            X_clear = []
            Y = load_image2D(HR, expand=True)
            Y_clear = load_image2D(SM, expand=True)
            N = norm
            
            for i, (lr, qm) in enumerate(LR_QM): # k-times
                img = load_image2D(lr)
                h,w = img.shape
                if with_clearance:
                    cl = load_image2D(qm)
    
                    X_clear.append(cl)
                X.append(img)
            h,w = X[0].shape
            X_clear = np.array(X_clear)
            X_clear = np.moveaxis(X_clear, 0, -1)
            
            X = np.array(X)
            X = np.moveaxis(X, 0, -1) # (n,n, k)
            
            if resize:
                if with_clearance:
                    x = batch_resize(X, X_clear, (h*scale, w*scale), type_clearance=type_clearance, version=version ) 
                else:
                    x = batch_resize(X, None, (h*scale, w*scale), version=version ) 
            else:

                
                if with_clearance:
                    
                    x = batch_transform(X, X_clear, type_clearance=type_clearance, version=version)
                else:
                    x = batch_transform(X, None, type_clearance=type_clearance, version=version)
            
            x = np.expand_dims(x, axis=0)
            yield x, Y, Y_clear, N
            
            
def batch_generator_SRCNN_test(data, with_clearance=True, type_clearance="sum", version=1, scale=3, resize=True, k=9):
    """
        version 1 & 2 & 3: 
        @return : upscale batch composed of images from the same set with its target features
                    (batch of images , targets, targets_clear, Norm). 
                    the batch of images is of dimension (number max clearance, n, n, 1)
        version 4: 
        @return : upscale batch composed of images from the same set with its target features
                    (batch of images , targets, targets_clear, Norm). 
                    the batch of images is of dimension (1, n, n, k)
    """
    

    if version == 1 or version == 2 or version==3: 
        for LR_QM, norm, SM in data:
            X = []
            X_clear = []
            #Y = load_image2D(HR, expand=True)
            Y_clear = load_image2D(SM, expand=True)
            N = norm
            
            for (lr, qm) in LR_QM:
                img = load_image2D(lr)
                h,w = img.shape
                if with_clearance:
                    cl = load_image2D(qm)
    
                    X_clear.append(cl) 
                X.append(img)
            h,w = X[0].shape
            X_clear = np.array(X_clear)
            
            if resize:
                if with_clearance:
                    x = batch_resize(X, X_clear, (h*scale, w*scale), type_clearance=type_clearance ) 
                else:
                    x = batch_resize(X, None, (h*scale, w*scale) ) 
            else:
                X = np.array(X)
                
                if with_clearance:
                    
                    x = batch_transform(X, X_clear, type_clearance=type_clearance)
                else:
                    x = batch_transform(X, None, type_clearance=type_clearance)
            #print(x.shape)
            yield x, Y_clear, N
            
    elif version == 4:
        for LR_QM, norm, SM in data:

            X = []
            X_clear = []
            #Y = load_image2D(HR, expand=True)
            Y_clear = load_image2D(SM, expand=True)
            N = norm
            
            for i, (lr, qm) in enumerate(LR_QM): # k-times
                img = load_image2D(lr)
                h,w = img.shape
                if with_clearance:
                    cl = load_image2D(qm)
    
                    X_clear.append(cl)
                X.append(img)
            h,w = X[0].shape
            X_clear = np.array(X_clear)
            X_clear = np.moveaxis(X_clear, 0, -1)
            
            X = np.array(X)
            X = np.moveaxis(X, 0, -1) # (n,n, k)
            
            if resize:
                if with_clearance:
                    x = batch_resize(X, X_clear, (h*scale, w*scale), type_clearance=type_clearance, version=version ) 
                else:
                    x = batch_resize(X, None, (h*scale, w*scale), version=version ) 
            else:

                
                if with_clearance:
                    
                    x = batch_transform(X, X_clear, type_clearance=type_clearance, version=version)
                else:
                    x = batch_transform(X, None, type_clearance=type_clearance, version=version)
            
            x = np.expand_dims(x, axis=0)
            yield x, Y_clear, N