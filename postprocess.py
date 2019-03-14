#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 17:57:36 2019

@author: xavier
"""

import numpy as np
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy
import scipy.misc
colors_map = [
    	[0, 0, 0],
    	[128, 0, 0],
    	[0, 128, 0],
    	[128, 128, 0],
    	[0, 0, 128],
    	[128, 0, 128],
    	[0, 128, 128],
    	[128, 128, 128],
    	[64, 0, 0],
    	[192, 0, 0],
    	[64, 128, 0],
    	[192, 128, 0],
    	[64, 0, 128],
    	[192, 0, 128],
    	[64, 128, 128],
    	[192, 128, 128],
    	[0, 64, 0],
    	[128, 64, 0],
    	[0, 192, 0],
    	[128, 192, 0],
    	[0, 64, 128],
    	[0, 0, 255]
]


def run(img_name = "2007_009788"):
    img = Image.open("../../../psa/psa/VOC2012/JPEGImages/"+img_name+".jpg")
    img = np.asarray(img)
    pixel_label = Image.open("../../../psa/psa//VOC2012/SegmentationClass/"+img_name+".png")
    padded_size = (int(np.ceil(img.shape[0]/8)*8), int(np.ceil(img.shape[1]/8)*8))
    print(img.shape)
    npy = img_name + "predict.npy"
    arr = np.load(npy)
    arr = [np.reshape(arr[:,i],(int(padded_size[0]/8),int(padded_size[1]/8))) for i in range(np.shape(arr)[1])]
    arr = np.array(arr)
    arr2 = np.argmax(arr,axis= 0)
    predict = np.max(arr,axis= 0)
    print(np.shape(arr))
    #img = Image.fromarray(arr2).resize((h, w), Image.LANCZOS)
    plt.imshow(arr2, cmap=cm.gray)
    scipy.misc.toimage(arr2, cmin = 0, cmax = 255, pal = colors_map, mode = 'P').save('./predict_result/'+img_name+'.png')
    
    
    #plt.show()

