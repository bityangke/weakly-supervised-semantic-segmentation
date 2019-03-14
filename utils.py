import numpy as np
import scipy.sparse as sp
import torch

import pickle as pkl
import networkx as nx

from scipy.sparse.linalg.eigen.arpack import eigsh
import sys

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    '''
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    #features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    '''
    #=======================
    import pickle
    graph = pickle.load(open("../../../psa/psa/VGG_RAM__/"+dataset+".pkl", "rb"))
    adj = sp.coo_matrix(graph,
                        dtype=np.float32)
    #=======================
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    #features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))  
    #adj = Laplacian(adj + sp.eye(adj.shape[0]))
    '''
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    #adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj = torch.FloatTensor(adj.toarray())

    idx_train = torch.arange(140,dtype=torch.long)
    idx_val = torch.arange(200,500, dtype=torch.long)
    idx_test = torch.arange(500,1000, dtype=torch.long)
    '''
    #===========================
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph', 'rgbxy']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    
    train_idx, test_idx, val_idx, _, allx, ally, graph, rgbxy = tuple(objects)# xxx_idx format is list    
    
    #adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    labels = torch.LongTensor(np.where(ally)[1])
    #y_train = np.zeros(labels.shape)
    #y_val = np.zeros(labels.shape)
    #y_test = np.zeros(labels.shape)
    #train_mask = sample_mask(train_idx, labels.shape[0])
    #test_mask = sample_mask(test_idx, labels.shape[0])
    #val_mask = sample_mask(val_idx, labels.shape[0])
    idx_train = torch.tensor(train_idx)
    idx_val = torch.tensor(val_idx)
    idx_test = torch.tensor(test_idx)
    #features = sp.coo_matrix(allx).tolil()
    #features = normalize(allx)#!!!!!!!!!!!!!!!!!!!!!!!!!!!
    rgbxy = torch.FloatTensor(np.array(rgbxy.todense()))       
    features = allx#!!!!!!!!!!!!!!!!!!!!!!!!!!!
    features = torch.FloatTensor(np.array(features.todense()))       
    #adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj = torch.FloatTensor(adj.toarray())
    #return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
    #===========================    
    return adj, features, labels, idx_train, idx_val, idx_test, rgbxy


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def Laplacian(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
