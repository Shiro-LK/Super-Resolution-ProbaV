# -*- coding: utf-8 -*-


import cv2
import numpy as np
import os
import pandas as pd
import math
from skimage import io 
from skimage.transform import rescale
import skimage
import numba
from numba import prange
import time
from pathlib import Path
# MAX 35 IMG

## Create TXT FILE for loading 
def import_norm_data(filename="data/norm.csv"):
    dic = {}
    file = pd.read_csv(filename, sep=" ", header= None, names=["name", "value"])
    for i, (name, value) in file.iterrows():
        dic[name] = value
    return dic

def seperate_NIR_RED(filename):
    with open(filename, "r") as f:
        temp = [line.replace("\\","/").split() for line in f]
    f_NIR = open(filename.replace(".txt", "_NIR.txt"), "w")
    f_RED = open(filename.replace(".txt", "_RED.txt"), "w")    
    
    for line in temp:
        if line[0].find("NIR") != -1:
            f_NIR.write(line[0]+" " + line[1] + "\n")
        else:
            f_RED.write(line[0]+" " + line[1] + "\n")
            
    f_NIR.close()
    f_RED.close()
    
def create_data(path, normalize_data):
    max_ = 0
    f_train = open(path+"train.txt", "w")
    f_test = open(path+"test.txt", "w")
    folders1 = os.listdir(path)
    for fold1 in folders1:
        p1 = os.path.join(path, fold1)
        if os.path.isdir(p1): # test/train fold
            folders2 = os.listdir(p1)
            
            for fold2 in folders2: 
                p2 = os.path.join(p1, fold2)
                if os.path.isdir(p2): # NIR RED fold
                    folders3 = os.listdir(p2)
                    
                    for fold3 in folders3:
                        p3 = os.path.join(p2, fold3)
                        if os.path.isdir(p3): #name imgset folders
                            if fold1 == "train": 
                                f_train.write(p3 + " " + str(normalize_data[fold3]) + "\n")
                            elif fold1 == "test":
                                f_test.write(p3 + " " + str(normalize_data[fold3]) + "\n")
                        max_ = max(max_, len(os.listdir(p3)))
                                
    print(max_)
    f_train.close()
    f_test.close()

## Load all data 
def load_data(filename, istrain=True):
    with open(filename, "r") as f:
        temp = [line.replace("\\","/").split() for line in f]
        
    data = []
    for path, v in temp:
        norm = float(v)
        
            
        if istrain:
            LR, QM, SM, HR = get_scene(path, istrain)
            data.append([LR, QM, norm, SM, HR])
        else:
            LR, QM, SM = get_scene(path, istrain)
            data.append([LR, QM, norm])
    return data

## load one scene data
def get_scene(path, istrain=True):
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
        LR = []
        QM = []
        if istrain:
            HR = os.path.join(path, names[-2])
        SM = os.path.join(path, names[-1])
        for lr in names[0:35]:
            lr_path = os.path.join(path, lr)
            if os.path.isfile(lr_path):
                LR.append(lr_path)
            else:
                break
            
        for qm in names[35:70]:
            qm_path = os.path.join(path, qm)
            if os.path.isfile(qm_path):
                QM.append(qm_path)
            else:
                break
            
        if istrain:
            return [LR, QM, SM, HR]
        else:
            return [LR, QM,  SM]
        
## METRIC FUNCTION FOR ONE SCENE
@numba.autojit
def score_scene(sr, hr, clearhr, norm, num_crop=6):
    """
        score for one scene
    """
    zSR = []
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

@numba.autojit
def baseline_predict_scene(LR, QM, before=True, interpolation=cv2.INTER_CUBIC):
    """
        baseline version 1 :
            average images with the maximum number of clearance pixel
            if before is true, average the image then apply the resize and return the resize image
            else resize the images and return the average
    """
     # load clearance map
    
    n = len(QM)
    clearance = np.zeros( (n,) )
   
    #for cl in QM:
    for i in prange(n):
        cl = QM[i]
        img_cl =  skimage.img_as_float64( cv2.imread(cl , -1) ).astype(np.bool)
        if img_cl is None:
            print("error")
        if len(np.unique(img_cl)) > 2:
            print(np.unique(img_cl))
            raise("Error during loading clearance map !!!! ")
        #img_cl = img_cl/255 # normalize value 0-1
        clearance[i] = np.sum(img_cl)

    maxcl = clearance.max()
    maxclears = [i for i in prange(len(clearance)) if clearance[i] == maxcl] # save index of image with max clearance
    
    if before:
        img_predict = np.zeros( (128, 128), dtype=np.float64)
        #for ids in maxclears:
        for i in prange(len(maxclears)):
            ids = maxclears[i]
            im = skimage.img_as_float64( cv2.imread(LR[ids], -1) ) 
            img_predict += im
        img_predict = img_predict/len(maxclears)
        
        im_rescale =   cv2.resize(img_predict, (384, 384), interpolation = interpolation)# rescale(im, scale=3, order=3, mode='edge', anti_aliasing=False, multichannel=False)#
        return im_rescale
    else:
    
        # upscale 
        
        img_predict = np.zeros( (384, 384), dtype=np.float64)
        
        #for ids in maxclears:
        for i in prange(len(maxclears)):
            ids = maxclears[i]
            im = skimage.img_as_float64( cv2.imread(LR[ids], -1) ) 
            im_rescale =   cv2.resize(im, (384, 384), interpolation = interpolation)# rescale(im, scale=3, order=3, mode='edge', anti_aliasing=False, multichannel=False)#
            img_predict += im_rescale
        img_predict = img_predict/len(maxclears)
        
        return img_predict

@numba.autojit
def baseline_predict_scenev2(LR, QM, interpolation=cv2.INTER_CUBIC):
    """
        baseline version 2 :
            average image with the maximum number of clearance pixel of one imageset
    """
     # load clearance map
    n = len(QM)
    clearance = np.zeros( (n,) )
    
    #for cl in QM:
    for i in prange(n):
        cl = QM[i]
        img_cl =  skimage.img_as_float64( cv2.imread(cl , -1) ).astype(np.bool)        
        if img_cl is None:
            print("error")
        if len(np.unique(img_cl)) > 2:
            print(np.unique(img_cl))
            raise("Error during loading clearance map !!!! ")
        #img_cl = img_cl/255 # normalize value 0-1
        clearance[i] = np.sum(img_cl)
        
    maxcl = clearance.max()
    maxclears = [i for i in prange(len(clearance)) if clearance[i] == maxcl] # save index of image with max clearance
    
    dim = len(maxclears)
    clearance_map = np.zeros( (dim, 128, 128), dtype=np.float64 )
    im =  np.zeros( (dim, 128, 128), dtype=np.float64)
    for i in prange(dim):
        ids = maxclears[i]
        cl = QM[ids]
        clearance_map[i] = skimage.img_as_float64( cv2.imread(cl , -1) )
        im[i] = skimage.img_as_float64( cv2.imread(LR[ids], -1) ) 
        
   
    img = im * clearance_map # pixel with no clearance equal 0
    
    clear = clearance_map.sum(axis=0)
    np.place(clear, clear==0, np.nan)
    img_predict = np.sum(img, axis=0)/clear
    
    # average value of maxclearance and replace nan value by them
    img_average = img.mean(axis=0)
    img_predict[ np.isnan(img_predict) ] = img_average[np.isnan(img_predict)]
    
    
    # upscale img
    img_resize= cv2.resize(img_predict, (384, 384), interpolation = interpolation)

    return img_resize

@numba.autojit
def baseline_predict_scenev3(LR, QM, interpolation=cv2.INTER_CUBIC):
    """
        baseline version 2 :
            average image with the maximum number of clearance pixel of one imageset
    """
     # load clearance map
    n = len(QM)
    clearance = np.zeros( (n,) )
    
    #for cl in QM:
    for i in prange(n):
        cl = QM[i]
        img_cl =  skimage.img_as_float64( cv2.imread(cl , -1) ).astype(np.bool)        
        if img_cl is None:
            print("error")
        if len(np.unique(img_cl)) > 2:
            print(np.unique(img_cl))
            raise("Error during loading clearance map !!!! ")
        #img_cl = img_cl/255 # normalize value 0-1
        clearance[i] = np.sum(img_cl)
        
    maxcl = clearance.max()
    
    max_clearance_value = clearance.argsort()[::-1]
    maxclears = [i for i in prange(len(clearance)) if clearance[i] == maxcl] # save index of image with max clearance
    
    dim = len(maxclears)
    clearance_map = np.zeros( (dim, 128, 128), dtype=np.float64 )
    im =  np.zeros( (dim, 128, 128), dtype=np.float64)
    for i in prange(dim):
        ids = maxclears[i]
        cl = QM[ids]
        clearance_map[i] = skimage.img_as_float64( cv2.imread(cl , -1) )
        im[i] = skimage.img_as_float64( cv2.imread(LR[ids], -1) ) 
        
   
    img = im * clearance_map # pixel with no clearance equal 0
    
    clear = clearance_map.sum(axis=0)
    np.place(clear, clear==0, np.nan)
    img_predict = np.sum(img, axis=0)/clear
    
    # replace nan value by value in image where the clearance is available
    nan_map = clear.copy()
    nan_map[~np.isnan(nan_map)] = 0.0
    nan_map[np.isnan(nan_map)] = 1.0
    for ids in max_clearance_value:
        if clearance[ids] == maxcl:
            pass
        else:
            cl = QM[ids]
            img_temp =  skimage.img_as_float64( cv2.imread(LR[ids], -1) ) 
            clear_temp = skimage.img_as_float64( cv2.imread(cl , -1) )
            temp = clear_temp*nan_map
            np.place(temp, temp==0, np.nan)
            temp = temp*img_temp
            img_predict[np.isnan(img_predict)] = temp[np.isnan(img_predict)] 
            nan_map[:, :] = nan_map[:,:] - (nan_map*clear_temp)

    # average value of maxclearance and replace nan value by them
    img_average = img.mean(axis=0)
    img_predict[ np.isnan(img_predict) ] = img_average[np.isnan(img_predict)]
    
    
    # upscale img
    img_resize= cv2.resize(img_predict, (384, 384), interpolation =interpolation)


    return img_resize

@numba.autojit
def baseline_predict(data, istrain=True, evaluate=True, version=1, interpolation=cv2.INTER_CUBIC):
    num = len(data)
    predicted = np.zeros( (num, 384, 384) ) # number of images in the dataset to check
    zsub = np.zeros((num,))

    
    if istrain:
        for i in prange( num ):
            LR, QM, norm, SM, HR = data[i]
            if version == 1:
                img_predict = baseline_predict_scene(LR, QM, interpolation=interpolation)
            elif version == 2:
                img_predict = baseline_predict_scenev2(LR, QM, interpolation=interpolation)
                
            elif version == 3:
                img_predict = baseline_predict_scenev3(LR, QM, interpolation=interpolation)
            else:
                raise("methode not implemented ! ")
            
            # save img
            predicted[i] = img_predict
            # evaluate
            
            if evaluate:
                num_crop = 6
                clearHR =  skimage.img_as_float64( cv2.imread(SM, -1 ) )
                hr = skimage.img_as_float64( cv2.imread(HR, -1) )
                zSR = score_scene(img_predict, hr, clearHR, norm, num_crop=num_crop)

                zsub[i] = zSR
        if evaluate:
            print("evaluation \n number of elements : {0} \n Z = {1}".format(len(zsub), zsub.mean()))
        return predicted

        
        
def baseline_predict_test(data, dirs = "results_baseline", interpolation=cv2.INTER_CUBIC):
    num = len(data)
    for i in range( num ):
            LR, QM, norm = data[i]
            p = Path(LR[0])
            img_predict = baseline_predict_scene(LR, QM, interpolation=interpolation)
            #print(img_predict.shape)
            # save img
            #predicted[i] = img_predict
            #names[i] = p.parts[-2]
            save_prediction(img_predict, p.parts[-2], directory=dirs)
    
    
def save_prediction(pred, names, directory):
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)
    #io.use_plugin('freeimage')
    p = os.path.join(directory,names+'.png')

    im = skimage.img_as_uint(pred)
    #io.imsave(arr=im, fname= p, plugin="freeimage")
    cv2.imwrite(p, im,  [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
#norm = import_norm_data()
#print(norm)
#
#create_data(path="data\\", normalize_data=norm)
    
#data_test = load_data(os.path.join("data","test.txt"), istrain=False)
#datas = load_data(os.path.join("data","train.txt"), istrain=True)

#begin = time.time()
#predict = baseline_predict(datas, istrain=True, evaluate=True, version=1)
#print(time.time()-begin)


#begin = time.time()
#baseline_predict_test(data_test)
#print(time.time()-begin)
