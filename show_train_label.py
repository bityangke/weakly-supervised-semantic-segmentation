import numpy as np
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


file_list = [f.split('.')[0] for f in os.listdir("../../../psa/psa/VGG_HA_CRF")]
segmentation_class = [f.split('.')[0] for f in os.listdir("../../../psa/psa/VOC2012/SegmentationClass")]
count = 0
fig = plt.figure(figsize=(63,47)) 
for name in segmentation_class:
    if name in file_list:
        if count < 450:
            count = count + 1
            continue
        image = Image.open("../../../psa/psa/VOC2012/JPEGImages/" + name + ".jpg")
        ground_truth = Image.open("../../../psa/psa/VOC2012/SegmentationClass/" + name + ".png")
        train_label_truth = Image.open("./train_label/" + name + "small.png")
        

        rows = 3
        columns = 5

        i = count % 5
        t = fig.add_subplot(rows,columns, i+1)
        t.set_title("image "+ str(count) + ":")
        t.set_axis_off()
        plt.imshow(image)
        t = fig.add_subplot(rows,columns,  columns+i+1)
        t.set_title("prediction "+ str(count) + ":")
        t.set_axis_off()
        plt.imshow(train_label_truth)
        t = fig.add_subplot(rows,columns, 2*columns+i+1)
        t.set_title("ground truth "+ str(count) + ":")
        t.set_axis_off()
        plt.imshow(ground_truth)
        
        if (count+1) % 5 == 0:
            plt.text(-600,-1150,s="For prediction: unknown-category points are in black.", bbox=dict(facecolor='yellow', alpha=0.1), fontsize=14)
            plt.savefig("./show_train_label/" + str(count-4) + "_to_" + str(count+1) + ".png")
            #plt.show()
            plt.close()
            print(count)
            fig = plt.figure(figsize=(63,47)) 
            #break 

        count = count + 1
        ground_truth.close()
        train_label_truth.close()

if (count+1) % 5 != 0:
    plt.text(-600,-1150,s="For prediction: unknown-category points are in black.", bbox=dict(facecolor='yellow', alpha=0.1), fontsize=14)
    plt.savefig("./show_train_label/" + str(count -count%5 +1) + "_to_" + str(count) + ".png")
    plt.show()
    plt.close()



        
    
    
