import numpy as np
#np.set_printoptions(threshold=np.nan)
import tqdm
import os
import sys
import tensorflow as tf
from PIL import Image
import scipy

#VOC2012  [1.aeroplane 2.bicycle 3.bird 4.boat 5.bottle 6.bus 7.car 8.cat 9.chair 10.cow 11.dining table 12.dog 13.horse 14.motorbike 15.person 16.potted plant 17.sheep 18.sofa 19.train 20.tv/monitor]

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
    	[255, 255, 255]
]
voc2012_classes = {0:"background", 1:"aeroplane", 2:"bicycle", 3:"bird", 4:"boat", 5:"bottle", 6:"bus", 7:"car", 8:"cat", 9:"chair", 10:"cow", 11:"diningtable", 12:"dog", 13:"horse", 14:"motorbike", 15:"person", 16:"pottedplant", 17:"sheep", 18:"sofa", 19:"train", 20:"tvmonitor"}


x = tf.placeholder(tf.float32, shape=(None, None))
y = tf.placeholder(tf.float32, shape=(None, None))
def compute():
    temp = tf.multiply(x, y)
    inter = tf.reduce_sum(temp)
    union = tf.reduce_sum(tf.subtract(tf.add(x, y), temp))
    return inter, union

IN_, UN_ = compute()
SPACE = 35
union_ = np.zeros(21, dtype=np.float32)
inter_ = np.zeros(21, dtype=np.float32)

#f = []
#for (dirpath, dirnames, filenames) in os.walk('f_out_cam_pred0'):#'f_out_rw0'):
#    f.extend(filenames)

file_list = [f.split('.')[0] for f in os.listdir("../../../psa/psa/VGG_HA_CRF")]
segmentation_class = os.listdir("../../../psa/psa/VOC2012/SegmentationClass")


exclude_photoes = ['2010_004960','0000_00000'] 


Iter = 1
count = 0
for s in segmentation_class:
    if count == Iter:
        break
    #count = count + 1
    print(s)
    print(s.split('.')[0] )
    name = s.split('.')[0]
    if  (not name in file_list)  or (name in exclude_photoes):                
        continue
    #CAM_p = Image.open('f_out_rw0/'+s)



    Ha_crf_p = np.load('./../../../psa/psa/VGG_HA_CRF/'+s.split('.')[0]+'.npy')
    #temp = Image.fromarray(Ha_crf_p[0]).resize((h, w), Image.LANCZOS)
    #Ha_crf = np.asarray(temp)    
    #width, height = Predict_p.size
    #Predict = np.array( Predict_p )
    #Predict = np.where(Predict==21, 0, Predict)
    Ha_crf_p = Ha_crf_p[()]
    print( Ha_crf_p )
    width, height = None, None
    for key in Ha_crf_p:
        #print(key)
        #print(Ha_crf_p[key])
        #print( Ha_crf_p[key].shape )
        width, height = Ha_crf_p[key].shape
        H_image = scipy.misc.toimage(np.asarray(Ha_crf_p[key]), cmin = 0, cmax = 255, pal = colors_map, mode = 'P')#.show()
        
    Ha_crf = Ha_crf_p

    #print(Predict)

    

    GT_p = Image.open('../../../psa/psa/VOC2012/SegmentationClass/'+s)
    print("( {} ) --> ( {},{} )".format(GT_p.size, height, width))
    #GT_p.resize((width,height), Image.BILINEAR) 
    GT_p = GT_p.resize((height,width) ) 
    #GT_p.show()
    GT = np.array( GT_p )
    GT = np.where(GT==255, 0, GT )
    #print(GT.shape)
    #print(GT)


    #print(Ha_crf.keys())  
    sess = tf.Session()
    for i in Ha_crf.keys():
        gt = np.where((GT==(i)),1,0)
        #ha_crf = np.where((Ha_crf[i]>=0.9),1,0)
        ha_crf = Ha_crf_p[i]
        in_, un_ = sess.run([IN_,UN_], feed_dict={x:ha_crf, y:gt}) 

        union_[i] = union_[i] + un_
        inter_[i] = inter_[i] + in_

    
    GT_p.close()

IoU = inter_ / union_

for k, v in zip(voc2012_classes.values(), IoU):
    print('{:{}}: {}'.format(k, SPACE, v))

mIoU = np.mean(IoU)
print('{:{}}: {}'.format('mIoU', SPACE, mIoU))



