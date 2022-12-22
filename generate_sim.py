import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import pickle
import os
import pdb
import scipy.io as scio
import torch.nn as nn
import h5py
from torch.nn import Parameter
import math
import random
import copy
torch.multiprocessing.set_sharing_strategy('file_system')

def cal_similarity( S1):
    batch_size = S1.size(0)

    S_pair = S1 * 1.
    pro = S_pair * 1.

    size = batch_size
    top_size = 3000
    m, n1 = pro.sort()

    pro[torch.arange(size).view(-1, 1).repeat(1, top_size).view(-1), n1[:, :top_size].contiguous().view(
        -1)] = 0.
    pro[torch.arange(size).view(-1), n1[:, -1:].contiguous().view(
        -1)] = 0.
    pro = pro / (pro.sum(1).view(-1, 1) + 1e-6)
    pro_dis = pro.mm(pro.t())
    pro_dis = pro_dis * 4000
    S_i = (S_pair * (1 - 0.3) + pro_dis * 0.3)
    return S_i


if __name__ == '__main__':
    feature_path = ''
    with open(feature_path + '/flickr_vgg16_features.pkl', 'rb') as f:
        all_data = pickle.load(f)
        train_x, train_L = all_data['train_data'], all_data['train_L']

    all_data = h5py.File(feature_path + '/mir25_crossmodal.h5')
    tag_fea = np.load(feature_path + '/taglist1386_glove.npy')
    tag_fea = torch.Tensor(tag_fea)
    tag_fea = F.normalize(tag_fea)


    tag_sim = tag_fea.mm(tag_fea.t())
    tag_sim = 1. * (tag_sim > 0.) * tag_sim

    train_y = np.asarray(all_data['train_y'])

    train_x, train_y, train_L = torch.Tensor(train_x), torch.Tensor(train_y), torch.Tensor(train_L)
    train_y = torch.Tensor(train_y)
    train_y_new = train_y.mm(tag_sim.float())
    F_I = F.normalize(train_x)
    F_T = F.normalize(train_y_new)
    S_T = F_T.mm(F_T.t())
    S_I = F_I.mm(F_I.t())

    mt = 0.01
    tm = 0.2
    im = 0.2
    sel = (S_T > tm) * (S_I > im) * 1.
    S_ = sel + (1 - sel) * (S_I * (1 - mt) + S_T * mt)

    S_ = cal_similarity(S_)
    with open('vgg16_based_sim_flickr.pkl', 'wb') as f:
        pickle.dump(S_, f)
