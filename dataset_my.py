import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import pickle

class DatasetPorcessing(Dataset):
    def __init__(self, train_data, train_y, train_label, transform=None, is_train=False):
        self.train_data = train_data
        self.is_train = is_train
        self.transform = transform
        self.train_y = train_y
        self.labels = torch.tensor(train_label).float()
    def __getitem__(self, index):
        # print(self.img_filename[index])
        # img = Image.open(os.path.join(self.img_path, self.name_list[index]))
        # img = img.convert('RGB')
        img = Image.fromarray(self.train_data[index])
        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img)
        # cur_tag_label = self.all_tags_labels[self.name2id[self.name_list[index].split('/')[-1]]]
        # tags = torch.Tensor(cur_tag_label['tag'][0]).float()
        label = self.labels[index]
        y_vector = torch.Tensor(self.train_y[index]).float()
        if self.is_train:
            return img1, img2, y_vector, label, index
        else:
            return img1, y_vector, label, index
    def __len__(self):
        return self.labels.shape[0]

