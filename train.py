import logging
import time
import argparse
import train_div
import torch
import os
import random
import numpy as np

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main(opt):
    sess = train_div.Session(opt)
    num_epoch = 0
    if opt.EVAL == True:
        sess.load_checkpoints()
        sess.eval()

    else:
        for epoch in range(opt.num_epochs):
            # train the Model
            if epoch < 40:
                sess.train(epoch)
    sess.load_checkpoints()
    num_epoch = 0
    for epoch in range(opt.num_epochs):
        # train the Model
        if epoch < 5:
            sess.train(epoch, True)
        # eval the Model
        sess.eval(step=epoch + 1, num_epoch=num_epoch, adapt=True)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='nus')
    parser.add_argument('--data_path', default='')
    parser.add_argument('--sim_path', default='')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of training epochs.')
    parser.add_argument('--batch_size', default=96, type=int, help='Size of a training mini-batch.')
    parser.add_argument('--learning_rate', default=0.002, type=float, help='Initial learning rate.')
    parser.add_argument('--weight_decay', default=0.0005, type=float, help='weight_decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='weight_decay')
    parser.add_argument('--EVAL', default=False, type=bool, help='train or test')
    parser.add_argument('--EVAL_INTERVAL', default=2, type=float, help='train or test')

    parser.add_argument('--bit', default=64, type=int, help='128, 64, 32, 16')

    parser.add_argument('--data_set', default='nus', type=str, help='loss1-alpha')
    parser.add_argument('--dw', default=1, type=float, help='loss1-alpha')
    parser.add_argument('--gamma', default=0.8, type=float, help='margin')
    parser.add_argument('--beta', default=0., type=float, help='beta')
    parser.add_argument('--cw', default=1, type=float, help='loss2-beta')

    parser.add_argument('--K', default=1.5, type=float, help='pairwise distance resize')

    parser.add_argument('--a1', default=0.01, type=float, help='1 order distance')
    parser.add_argument('--a2', default=0.3, type=float, help='2 order distance') 
    opt = parser.parse_args()
    print(opt)

    main(opt)
