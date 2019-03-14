import numpy as np
import pickle
import scipy
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
from scipy import sparse
from scipy.sparse import csr_matrix
import scipy.misc
import random
import matplotlib.cm as cm

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

# CAM (confidence region)
# 2007_000032.npy
# 0:    [281, 500]
# 14:   [281, 500]

# get dictionary


img_name = "2009_003088"
img_name = "2007_007387"
img_name = "2007_009788"
img_name = "2009_005031"
img_name = "2007_000063"
img_name = "2010_004119"
img_name = "2009_001937"
img_name = "2007_000480"

def make_data(img_name):
    #os.chdir("/home/xavier/Desktop")
    cam = np.load("../../../psa/psa/VGG_LA_CRF/"+img_name+".npy").item()
    img = Image.open("../../../psa/psa/VOC2012/JPEGImages/"+img_name+".jpg")
    pixel_label = Image.open("../../../psa/psa/VOC2012/SegmentationClass/"+img_name+".png")
    img = np.asarray(img)
   
    print(cam.keys())
    #plt.imshow(cam[0], cmap=cm.gray)
    #plt.imshow(cam[15], cmap=cm.gray)
    #scipy.misc.imsave(os.path.join("test2" + '.png'), cam[0])
    #print(cam[14])
    padded_size = (int(np.ceil(img.shape[0]/8)*8), int(np.ceil(img.shape[1]/8)*8))#to make image size can divided by 8
    p2d = (0, padded_size[1] - img.shape[1], 0, padded_size[0] - img.shape[0])# decide how many zeros need to paded to make the images in the same size
    orig_shape = img.shape
    cams = np.zeros((22, orig_shape[0], orig_shape[1]), np.float32)
    for k, v in cam.items():
        cams[k] = v#cams[k+1] = v
    #cams[0] = (1 - np.max(cams[1:], (0), keepdims=False))**16
    #cams[0] = np.power(1 - np.max(cams[1:], axis = 0, keepdims=False),16)#Background score
    #cam[1]=np.where(cam[1]>.99,cam[1],0)
    #scipy.misc.imsave(os.path.join("la_crfs_1-2" + '.png'), cam[1])#save img
    
    
    # [21, 288, 504]
    cams = np.pad(cams, ((0, 0), (0, p2d[3]), (0, p2d[1])), mode='constant')#padding zeroes
    cams = torch.from_numpy(cams)
    # [21, 36, 63]
    cams = F.avg_pool2d(cams, 8, 8).cpu().numpy()
    b, w, h = cams.shape
    #mask = np.zeros((w,h))
    """
    # get key and value
    for key, value in cam.items():
        # key
        print(key)
        # value
        print(value)
    """
    # merge cam
    #cam_ = np.sum(cams, axis=0)
    #mask = np.where(cam_!=0, 1, 0)
    #=====================================================================
    # faetures 1 affinity map
    #f = np.load(os.path.join("/home/xavier/Desktop/psa-master/aff_map",img_name + ".npy"))
    # features 2 rgbxy
    f = np.zeros((w, h, 5))
    #f = np.load(os.path.join("/home/xavier/Desktop/psa-master/aff_map",img_name + ".npy"))
    # downsample img
    # [36, 63, 3]  = [w ,h ,3]
    #'''
    img = Image.fromarray(img).resize((h, w), Image.LANCZOS)
    plt.imshow(img, cmap=cm.gray)
    #scipy.misc.imsave(os.path.join("test2" + '.png'), img)
    img = np.asarray(img)
    print(np.shape(img))
    
    # get rgb features
    
    f[:, :, :3] = img/255. #normalize    
    # get xy feature
    for i in range(w):
        for j in range(h):            
            #f[i, j, 3] = float(i)
            #f[i, j, 4] = float(j)
            f[i, j, 3] = float(i)/w
            f[i, j, 4] = float(j)/h
    #'''
    #=====================================================================
    # get label and to one-hot
    #label = Image.open("/home/xavier/Desktop2007_000032.png")
    #scipy.misc.toimage(np.asarray(label), cmin = 0, cmax = 255, pal = colors_map, mode = 'P').save('a.png')
    # downsample label
    # [36, 63, 3]
    
    def crf2mask(high_crf,low_crf,thre4confident = .95):
        max_value = np.zeros(np.shape(high_crf[0]))
        label = np.zeros(np.shape(high_crf[0]))
        for key in low_crf.keys():
            if key>0:
                low_crf[key] = np.where(low_crf[key] > thre4confident, low_crf[key], 0)
                label = np.where(max_value < low_crf[key], key, label)#record class of max value
                max_value = np.where(max_value < low_crf[key] , low_crf[key], max_value)# record max_value            
                
        
        #plt.imshow(max_value, cmap=cm.gray)
        #bg = np.zeros(np.shape(high_crf[0]))
        high_crf[0] = np.where(high_crf[0] > thre4confident, high_crf[0], 0)
        label = np.where(max_value < high_crf[0] , 21, label)#record background       
        scipy.misc.toimage(label, cmin = 0, cmax = 255, pal = colors_map, mode = 'P').save("./train_label/"+img_name+'small.png')
        mask = np.where(label!=0,1,0)
        label = np.where(label==21 , 0, label)#record class of max value        
        return label, mask
    
    la_crf = np.load("../../../psa/psa/VGG_LA_CRF/"+img_name+".npy").item()
    ha_crf = np.load("../../../psa/psa/VGG_HA_CRF/"+img_name+".npy").item()
    print(la_crf)

    for key in la_crf:
        temp = Image.fromarray(la_crf[key]).resize((h, w), Image.LANCZOS)
        la_crf[key] = np.asarray(temp)
    
    temp = Image.fromarray(ha_crf[0]).resize((h, w), Image.LANCZOS)
    ha_crf[0] = np.asarray(temp)    
    
    label,mask = crf2mask(ha_crf ,la_crf)
    #scipy.misc.toimage(label, cmin = 0, cmax = 255, pal = colors_map, mode = 'P').save('label.png')
    
    '''
    label = label.resize((h, w), Image.BILINEAR)
    label = np.asarray(label)
    label = np.where(label==255,0,label)
    #scipy.misc.toimage(label, cmin = 0, cmax = 255, pal = colors_map, mode = 'P').save(  'bilinear.png')
    label = np.reshape(label, (-1))
    len_ = label.shape[0]
    ally = [[0 for x in range(21)] for y in range(len_)]
    mask_ = np.reshape(mask, (-1))
    print(type(label))
    print(np.shape(label))
    '''
    #scipy.misc.toimage(label, cmin = 0, cmax = 255, pal = colors_map, mode = 'P').save( 'colormap.png')
    
    '''
    y = []
    for i in range(len_):
        if mask_[i] ==1:
            ally[i][int(label[i])]=1
            y.append(ally[i])
    
    ally = np.asarray(ally)
    y = np.asarray(y)
    print("ally", ally.shape)
    print("y", y.shape)
    print("ty", y.shape)
    '''
    pixel_label = Image.open("../../../psa/psa/VOC2012/SegmentationClass/"+img_name+".png")
    pixel_label = np.asarray(pixel_label)
    pixel_label = np.where(pixel_label==255,0,pixel_label)#erasing boundary
    pixel_label = Image.fromarray(pixel_label)
    pixel_label = pixel_label.resize((h, w), Image.NEAREST)
    pixel_label = np.asarray(pixel_label)
    #plt.imshow(pixel_label, cmap=cm.gray)
    #scipy.misc.toimage(label, cmin = 0, cmax = 255, pal = colors_map, mode = 'P').save("./train_label/"+img_name+'small.png')
    scipy.misc.toimage(pixel_label, cmin = 0, cmax = 255, pal = colors_map, mode = 'P').save("./label/"+img_name+'small2.png')
    scipy.misc.toimage(mask, cmin = 0, cmax = 255, pal = colors_map, mode = 'P').save("./mask/"+img_name+'small.png')
    print("h:",h," w:",w)
    print("np.shape(pixel_label):", np.shape(pixel_label) )
    pixel_label = np.reshape(pixel_label,(-1))
    print("shape:", len(pixel_label))
    coll = {}#counting number of label
    for i in range(len(pixel_label)):
        if pixel_label[i] not in coll:
            coll[pixel_label[i]] = 1
        else:
            coll[pixel_label[i]] += 1
    print(coll)    
    #f[:, :, :3] = img
    
    # filt non-confidence region
    x = []# train_idx
    y = []# test_idx
    #num_train_node = np.sum(mask)
    val_idx = []
    val_ratio = .1
    mask = np.reshape(mask,(-1))
    pixel_label = np.where(pixel_label==255,0,pixel_label)
    label = np.reshape(label ,(-1))
    for i in range(len(mask)):
            if mask[i] == 1 :
                if random.random() > (1-val_ratio):#30% to be validation
                    val_idx.append(i)
                else:
                    x.append(i)# store train_idx
            else:
                y.append(i)# test idx
    ally = [[0 for x in range(21)] for y in range(len(mask))]# all labels
    ally = np.asarray(ally)
    print("len4mask:",len(mask))
    print("len4ally:",len(ally))
    for i in range(len(mask)):
        if mask[i] ==1:
            ally[i][int(label[i])]=1# sudo label        
        else:
            ally[i][int(pixel_label[i])]=1# groundTrue        
    #x = np.asarray(x)
    print("np.shape(ally)",np.shape(y))            
    #print(x)
    # to csr_matrix
    allx = np.reshape(f, (w*h, np.shape(f)[2]))
    rgbxy = np.reshape(f, (w*h, np.shape(f)[2]))
    allx = sparse.csr_matrix(allx)
    #print(allx)figh
    print("allx", allx.shape)
    # save as allx
    
    pickle.dump(allx, open("./data/ind."+img_name+".rgbxy", "wb"))
    pickle.dump(allx, open("./data/ind."+img_name+".allx", "wb"))
    # save as x
    pickle.dump(x, open("./data/ind."+img_name+".x", "wb"))
    # save as tx
    pickle.dump(val_idx, open("./data/ind."+img_name+".tx", "wb"))
    # save as y
    pickle.dump(y, open("./data/ind."+img_name+".y", "wb"))
    # save as ty
    pickle.dump(y, open("./data/ind."+img_name+".ty", "wb"))
    # save as ally
    pickle.dump(ally, open("./data/ind."+img_name+".ally", "wb"))
    
    # graph
    # [2268, 2268]
    graph = pickle.load(open("../../../psa/psa/VGG_RAM__/"+img_name+".pkl", "rb"))
    #graph_ = sorted(graph.reshape(-1))
    #thres = graph_[int(w*h*w*h*0.4)]
    def in_radius(r,index,num_col,neighbor):
        row = index//num_col
        col = index % num_col
        nei_row = neighbor//num_col
        nei_col = neighbor % num_col
        return (row-nei_row)**2+(col-nei_col)**2 < r**2
        
    thres = 0
    print("graph", graph.shape)
    #num_r,num_c = w,h#number of row, number of column
    pixel_label = np.reshape(pixel_label,(-1))
    graph_ = {}# graph[i]={indexes of i's neighbor }
    #=== test: assuming that we have ground truth affinity  ==================================
    '''
    for i in range(w*h):
        temp = []    
        for j in range(w*h):
            if in_radius(20,i,num_c,j) and pixel_label[i] == pixel_label[j]:#wether they are neighbor
                temp.append(j)
             #   temp_.append(j+w*h)
        graph_[i] = temp
    '''
    #=====================================    
    
    for i in range(w*h):
        temp = []
        #temp_ = []
        for j in range(w*h):
            if graph[i, j] > thres:#wether they are neighbor
                temp.append(j)
             #   temp_.append(j+w*h)
        graph_[i] = temp
     
    #    graph_[i+w*h] = temp_#?
    #print(graph_)
    # save as graph
    pickle.dump(graph_, open("./data/ind."+img_name+".graph", "wb"))
    
    
    with open("./data/ind."+img_name+".test.index", "w") as w_:
        for i in range(w*h):
            w_.write(str(i+w*h)+"\n")
        w_.close()


#make_data(img_name)
#===================================================
file_list = [f.split('.')[0] for f in os.listdir("../../../psa/psa/VGG_HA_CRF")]
segmentation_class = [f.split('.')[0] for f in os.listdir("../../../psa/psa/VOC2012/SegmentationClass")]

complete_list = [f.split('.')[1] for f in os.listdir("./data/")]
#complete_list = []

count = 20
i = 1
f_list = []
for name in segmentation_class:
    #print(name)
    if name in file_list  and  (not name in complete_list):                
        f_list.append(name)        
        
print(len(f_list))
#for n in f_list[:count]:
for n in f_list[:]:
    make_data(n)
