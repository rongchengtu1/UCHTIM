import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from metric import compress, calculate_top_map, calculate_map, p_topK, calculate_map_1
from models import ImgNet, TxtNet
import os.path as osp
import torchvision.transforms as transforms
# from load_data import get_loader, get_loader_wiki
import dataset_my as dp
import numpy as np
import pdb
import time
import logging
import pickle
import random
import cv2
import scipy.io as scio
import h5py
import pickle
import os
import argparse
import logging

class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample

def calc_dis(query_L, retrieval_L, query_dis, top_k=32):
    num_query = query_L.shape[0]
    map = 0
    for iter in range(num_query):
        gnd = (np.dot(query_L[iter, :], retrieval_L.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = query_dis[iter]
        ind = np.argsort(hamm)[:top_k]
        gnd = gnd[ind]
        tsum = np.int(np.sum(gnd))
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map = map + np.mean(count / (tindex))
    map = map / num_query
    return map


class Session:
    def __init__(self, opt):
        self.opt = opt

        data_path = opt.data_path
        sim_path = opt.sim_path
        self.data_set = opt.data_set
        ### data processing

        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        train_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                               transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0)),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.RandomApply([color_jitter], p=0.7),
                                               transforms.RandomGrayscale(p=0.2),
                                               GaussianBlur(3),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                               ])
        transformations = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        all_dta = h5py.File(data_path + '/mir25_crossmodal_10ktrain.h5')
        txt_feat_len = np.asarray(all_dta['test_y']).shape[1]
        self.database_dataset = dp.DatasetPorcessing(
                np.asarray(all_dta['data_set']), np.asarray(all_dta['dataset_y']), np.asarray(all_dta['dataset_L']), transformations)
        self.train_dataset = dp.DatasetPorcessing(
                np.asarray(all_dta['train_data']), np.asarray(all_dta['train_y']), np.asarray(all_dta['train_L']), transformations, True)
        self.test_dataset = dp.DatasetPorcessing(
                np.asarray(all_dta['test_data']), np.asarray(all_dta['test_y']), np.asarray(all_dta['test_L']), transformations)
        with open(sim_path, 'rb') as f:
             gs2 = pickle.load(f)
             gs2 = 2.0 * gs2 - 1.
        self.gs = gs2


        self.database_labels = self.database_dataset.labels
        self.test_labels = self.test_dataset.labels
        self.train_labels = self.train_dataset.labels
        self.image_codes = torch.rand((self.train_labels.size(0), opt.bit)).float()
        self.text_codes = torch.rand((self.train_labels.size(0), opt.bit)).float()

        # Data Loader (Input Pipeline)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                        batch_size=opt.batch_size,
                                                        shuffle=True,
                                                        num_workers=4,
                                                        drop_last=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                       batch_size=opt.batch_size,
                                                       shuffle=False,
                                                       num_workers=4)

        self.database_loader = torch.utils.data.DataLoader(dataset=self.database_dataset,
                                                           batch_size=opt.batch_size,
                                                           shuffle=False,
                                                           num_workers=4)

        self.CodeNet_I = ImgNet(code_len=opt.bit, txt_feat_len=txt_feat_len)
        self.CodeNet_T = TxtNet(code_len=opt.bit, txt_feat_len=txt_feat_len)

        self.opt_I = torch.optim.SGD([{'params': self.CodeNet_I.features.parameters(), 'lr': opt.learning_rate * 0.05},
         {'params': self.CodeNet_I.classifier.parameters(), 'lr': opt.learning_rate * 0.05},
         {'params': self.CodeNet_I.hashlayer.parameters(), 'lr': opt.learning_rate}], lr=opt.learning_rate, momentum=opt.momentum,
                                     weight_decay=opt.weight_decay)
        self.opt_T = torch.optim.SGD(self.CodeNet_T.parameters(), lr=opt.learning_rate, momentum=opt.momentum,
                                     weight_decay=opt.weight_decay)



        self.best = 0
        self.best_i2t = 0
        self.best_t2i = 0
        # pdb.set_trace()
        logger = logging.getLogger('train')
        logger.setLevel(logging.INFO)
        stream_log = logging.StreamHandler()
        stream_log.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        stream_log.setFormatter(formatter)
        logger.addHandler(stream_log)
        self.logger = logger
    def loss_denoise(self, code_sim, S, weight_s):
        loss = (weight_s * (code_sim - S).pow(2)).mean()
        return loss
    def loss_cal(self, code_I, code_T, S, weight_s, I):
        B_I = F.normalize(code_I)
        B_T = F.normalize(code_T)

        BI_BI = B_I.mm(B_I.t())
        BT_BT = B_T.mm(B_T.t())
        BI_BT = B_I.mm(B_T.t())

        diagonal = BI_BT.diagonal()
        all_1 = torch.rand((BT_BT.size(0))).fill_(1).cuda()
        loss_pair = F.mse_loss(diagonal, 1.5 * all_1)

        loss_dis_1 = self.loss_denoise(BT_BT * (1 - I), S * (1 - I), weight_s)
        loss_dis_2 = self.loss_denoise(BI_BT * (1 - I), S * (1 - I), weight_s)
        loss_dis_3 = self.loss_denoise(BI_BI * (1 - I), S * (1 - I), weight_s)

        loss_cons = F.mse_loss(BI_BT, BI_BI) + \
                    F.mse_loss(BI_BT, BT_BT) + \
                    F.mse_loss(BI_BI, BT_BT) + \
                    F.mse_loss(BI_BT, BI_BT.t())

        loss = loss_pair + (loss_dis_1 + loss_dis_2 + loss_dis_3) * self.opt.dw \
               + loss_cons * self.opt.cw


        return loss, (loss_pair, loss_dis_1, loss_dis_2, loss_dis_3, loss_cons, loss_cons)


    def train(self, epoch, start_denoise=False):
        self.CodeNet_I.cuda().train()
        self.CodeNet_T.cuda().train()
        top_mAP = 0.0
        num = 0.0
        self.logger.info('Epoch [%d/%d], alpha for ImgNet: %.3f, alpha for TxtNet: %.3f' % (
            epoch + 1, self.opt.num_epochs, self.CodeNet_I.alpha, self.CodeNet_T.alpha))
        for idx, (img, _, txt, labels, index) in enumerate(self.train_loader):
            img = Variable(img.cuda())
            txt = Variable(torch.FloatTensor(txt.numpy()).cuda())

            batch_size = img.size(0)
            I = torch.eye(batch_size).cuda()


            fea_i, code_I = self.CodeNet_I(img)
            fea_t, code_T = self.CodeNet_T(txt)

            S0 = self.gs[index, :][:, index].cuda()
            if start_denoise:
                B_I = F.normalize(code_I)
                B_T = F.normalize(code_T)
                code_sim = (B_I.mm(B_T.t()) + B_I.mm(B_I.t()) + B_T.mm(B_T.t())) / 3.
                select_pos = (torch.abs(code_sim - S0) > self.opt.gamma) * 1.
                I = torch.eye(code_sim.size(0)).cuda()

                weight_s = (1 - torch.abs(S0)) * select_pos + 1 - select_pos

                S = (1 - select_pos) * S0 + select_pos * torch.sign(code_sim - S0)
                S = S * (1 - I) + I
            else:
                weight_s = 1.
                S = S0

            loss0, all_los0 = self.loss_cal(code_I, code_T, S.detach(), weight_s, I)
            loss = loss0
            self.opt_I.zero_grad()
            self.opt_T.zero_grad()
            loss.backward(retain_graph=True)
            self.opt_I.step()
            self.opt_T.step()

            loss1, loss2, loss3, loss4, loss5, loss6 = all_los0

            top_mAP += calc_dis(labels.cpu().numpy(), labels.cpu().numpy(), -S0.cpu().numpy())

            num += 1.
            if (idx + 1) % (len(self.train_loader)) == 0:
                self.logger.info(
                    'Epoch [%d/%d], Iter [%d/%d] '
                    'Loss1: %.4f '
                    'Loss5: %.4f Loss6: %.4f '
                    'Total Loss: %.4f '
                    'mAP: %.4f'
                    % (
                        epoch + 1, self.opt.num_epochs, idx + 1,
                        len(self.train_loader) // self.opt.batch_size,
                        loss1.item(),
                        code_T.abs().mean().item(),
                        code_I.abs().mean().item(),
                        loss.item(),
                        top_mAP / num))

    def eval(self, step=0, num_epoch=0, last=False, adapt=False):
        # Change model to 'eval' mode (BN uses moving mean/var).
        self.CodeNet_I.eval().cuda()
        self.CodeNet_T.eval().cuda()
        if self.opt.EVAL == False:
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.database_loader, self.test_loader, self.CodeNet_I,
                                                              self.CodeNet_T)

            # MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=5000)
            # MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=5000)
            MAP_I2Ta = calculate_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L)
            MAP_T2Ia = calculate_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L)
            K = [1, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
            self.logger.info('--------------------Evaluation: Calculate top MAP-------------------')
            # self.logger.info('MAP@5000 of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2T, MAP_T2I))
            self.logger.info('MAP@All of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2Ta, MAP_T2Ia))
            # self.logger.info('MAP@All of Image to Image: %.3f, MAP of Text to Text: %.3f' % (MAP_I2Ia, MAP_T2Ta))
            self.logger.info('--------------------------------------------------------------------')
            if MAP_I2Ta + MAP_T2Ia > self.best:
                num_epoch = 0

                if not adapt:
                    self.save_checkpoints(step=step, best=True)
                self.best = MAP_T2Ia + MAP_I2Ta
                self.best_i2t = MAP_I2Ta
                self.best_t2i = MAP_T2Ia
                self.logger.info("#########is best:%.3f #########" % self.best)
            else:
                num_epoch += 1
        if self.opt.EVAL:
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.database_loader, self.test_loader, self.CodeNet_I,
                                                              self.CodeNet_T)
            MAP_I2Ta = calculate_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L)
            MAP_T2Ia = calculate_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L)
            self.logger.info('--------------------Evaluation1: Calculate top MAP-------------------')
            self.logger.info('MAP@All of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2Ta, MAP_T2Ia))
            self.logger.info('--------------------------------------------------------------------')

        return num_epoch

    def save_checkpoints(self, step, path='',
                         best=False):
        ckp_path = path + '/model_ckp_ablation_text_' + self.data_set + '/our_model_bit_'+str(int(self.opt.bit)) + '.pth'
        if not os.path.exists(path + '/model_ckp_ablation_text_' + self.data_set):
            os.makedirs(path + '/model_ckp_ablation_text_' + self.data_set)
        obj = {
            'ImgNet': self.CodeNet_I.state_dict(),
            'TxtNet': self.CodeNet_T.state_dict(),
            'step': step,
        }
        torch.save(obj, ckp_path)
        self.logger.info('**********Save the trained model successfully.**********')

    def load_checkpoints(self, path=''):
        ckp_path = path + '/model_ckp_ablation_text_' + self.data_set + '/our_model_bit_'+str(int(self.opt.bit)) + '.pth'
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.logger.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            self.logger.error('********** No checkpoint %s!*********' % ckp_path)
            return
        self.CodeNet_I.load_state_dict(obj['ImgNet'])
        self.CodeNet_T.load_state_dict(obj['TxtNet'])
        self.logger.info('********** The loaded model has been trained for %d epochs.*********' % obj['step'])





