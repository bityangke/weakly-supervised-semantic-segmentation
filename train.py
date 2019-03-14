from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy, normalize
from models import GCN

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--sigma", type=float, default=1., help="gaussian filter parameer")
parser.add_argument("--lamda", type=float, default=1., help="loss function parameter ( of background points)")
args = parser.parse_args()
args.cuda = torch.cuda.is_available()


np.random.seed(args.seed)
#np.random.seed(time.time())
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)



img_name = "2007_009788"
img_name = "2007_000480"
def train_run(img_name):
    adj, features, labels, idx_train, idx_val, idx_test, rgbxy = load_data(dataset = img_name)
    
    # Model and optimizer
    model = GCN(nfeat=features.shape[1],
                nhid=80,
                nclass=labels.max().item() + 1,
                dropout=0.5)
    optimizer = optim.Adam(model.parameters(),
                           lr=0.01, weight_decay=5e-4)
    
    print("a\n")

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    
    
    def train(epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train1 = F.nll_loss(output[idx_train], labels[idx_train])
        '''
        print(labels.shape)
        #print(output.shape)
        c = torch.where(labels.cpu() == 0,torch.ones(labels.shape),torch.zeros(labels.shape))
        print(c.shape)
        background_points = torch.sum( c )
        total_points = labels.shape[0] 
        foreground_points = total_points - background_points

        l = labels[idx_train].cpu()
        ba = l.shape[0]
        deep = output.shape[1]
        labels_one_hot = torch.zeros((ba,deep))
        #labels_one_hot[:,0] = torch.where(l == 0, torch.ones(ba)/background_points, torch.zeros(ba))
        #labels_one_hot[:,0] = torch.where(l == 0, torch.ones(ba)*total_points/background_points, torch.zeros(ba))
        labels_one_hot[:,0] = torch.where(l == 0, torch.ones(ba), torch.zeros(ba))
        for i in range(1,deep):
            #labels_one_hot[:,i] = torch.where(l ==  i, torch.ones(ba)/foreground_points*foreground_inputs, torch.zeros(ba))
            #labels_one_hot[:,i] = torch.where(l ==  i, torch.ones(ba)*total_points/foreground_points, torch.zeros(ba))
            labels_one_hot[:,i] = torch.where(l ==  i, torch.ones(ba), torch.zeros(ba))
        loss_func = torch.nn.BCELoss()

        #print(output[idx_train].shape)
        #print(labels_one_hot.shape)
        soft = torch.nn.Softmax(dim=1) 
        loss_train1 = loss_func( soft(output[idx_train]).cpu(), labels_one_hot)
        '''
        '''
        rowsum =  torch.sum(output[idx_train],1)
        r_inv = torch.pow(rowsum, -1)
        r_inv[torch.isinf(r_inv)] = 0
        r_mat_inv = torch.diag(r_inv) 
        mx = torch.mm(r_mat_inv, output[idx_train])
        loss_train1 = loss_func( mx.cpu(), labels_one_hot)
        '''


        #print("b= {}, f = {}".format(background_points,foreground_points))
        l = output[idx_test].shape[0]
        W = torch.zeros((l,l,rgbxy.shape[1])).cuda()
        ones = torch.unsqueeze(torch.ones((l,5)),0)

        rgbxy_2d = rgbxy[idx_test]
        rgbxy_3d = []
        for i in range(l):
            rgbxy_3d.append(rgbxy_2d)
        rgbxy_3d = tuple(rgbxy_3d)
        rgbxy_3d = torch.stack(rgbxy_3d, dim=1).cuda()
       # ones_t = ones.transpose(ones, 1,2)
        
        '''       
        for i in range(l):
            for j in range(l):
                W[i][j] = rgbxy[idx_test[i]]-rgbxy[idx_test[j]]
        '''
        W = rgbxy_3d.transpose(0,1) - rgbxy_3d
        #'''
        sigma = args.sigma
        W = torch.exp(-torch.norm(W, dim=2)/2/sigma)
        d = torch.sum(W,dim=1)
        n_cut = 1
        for k in range(output.shape[1]):
            s = output[idx_test,k]
            n_cut = n_cut + torch.mm(torch.mm(torch.unsqueeze(s,0), W), torch.unsqueeze(1-s, 1)) / (torch.dot(d,s))
        
        lamda = args.lamda
        print(n_cut)
        loss_train = loss_train1.cuda() + lamda*n_cut
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
    
        if not False:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(features, adj)
    
        #loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        loss_val = loss_train1
        acc_val = accuracy(output[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

    
    def test():
        model.eval()
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
            
        #np.save(img_name+'predict',output.detach().cpu().numpy())
        print("save prediction!---"+img_name+'predict')
        with open("retport.txt",'a') as f:
            f.write(img_name + "\t")
            f.write("test acc:" + str(acc_test) + "\n")
            
    
    # Train model
    t_total = time.time()
    for epoch in range(200):
        train(epoch)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    
    # Testing
    test()
    from postprocess import run
    run(img_name)

#============================================
import os
#train_run(img_name)

file_list = [f.split('.')[0] for f in os.listdir("../../../psa/psa/VGG_HA_CRF")]
segmentation_class = [f.split('.')[0] for f in os.listdir("../../../psa/psa//VOC2012/SegmentationClass")]

#complete_list = [f.split('.')[0] for f in os.listdir("./predict_result/")]
complete_list = []
#segmentation_class = os.listdir("data")
count = 100#20
i = 0
exclude_photoes = ['2010_004960','0000_00000'] 
for name in segmentation_class[200:]:
    print(name)
    #if  name.split('.')[0] in file_list:        
    #if  name in file_list:        
    if name in file_list  and  (not name in complete_list) and (not name in exclude_photoes):                
        train_run(name)
        i=i+1
#'''
    if i==count:
        break
#'''

