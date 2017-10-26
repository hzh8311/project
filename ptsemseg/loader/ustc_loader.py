import os
import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.utils import data


class ustcLoader(data.Dataset):
    def __init__(self, root, split="train", is_transform=False, img_size=(300, 500)):
        self.root = root
        self.split = split
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.is_transform = is_transform
        if self.img_size == (300, 500):
            self.mean = np.load(os.path.join(self.root, 'mean.npy'))
            self.std = np.load(os.path.join(self.root, 'std.npy'))
        else:
            self.mean = np.load(os.path.join(self.root, 'mean224.npy'))
            self.std = np.load(os.path.join(self.root, 'std224.npy'))
        self.n_classes = 3
        self.files = collections.defaultdict(list)

        for split in ["train", "test", "val"]:
            if self.img_size == (300, 500):
                file_list = [_ for _ in os.listdir(root + split)
                            if not _.endswith('_gt.jpg') and _.startswith('p')]
            else: # 300x300 image -> 224x224 image
                file_list = [_ for _ in os.listdir(root + split)
                            if not _.endswith('_gt.jpg') and _.startswith('s')]
            self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_path = self.root + self.split + '/' + img_name
        lbl_path = self.root + self.split + '/' + img_name[:-4] + '_gt.jpg'

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(lbl_path, mode='L')
        lbl = (np.array(lbl, dtype=np.int32) / 128.).round().astype(np.int32)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        # print(np.unique(lbl.numpy()))
        return img, lbl

    def transform(self, img, lbl):
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= self.mean
        img /= self.std
        # img = m.imresize(img, (self.img_size[0], self.img_size[1]), 'nearest')
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        #  img = img.astype(float) / 255.0
        # NHWC -> NCWH
        img = img.transpose(2, 0, 1)
        lbl = lbl.astype(int)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def decode_segmap(self, temp, plot=False):
        background = [0, 0, 255]
        scratch = [0, 255, 0]
        foreground = [255, 0, 0]

        label_colours = np.array([background, scratch, foreground])
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

if __name__ == '__main__':
    local_path = '/data5/huangzh/project/data/'
    dst = ustcLoader(local_path, is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        # if i == 0:
            # img = torchvision.utils.make_grid(imgs).numpy()
            # img = np.transpose(img, (1, 2, 0))
            # img = img[:, :, ::-1]
            # plt.imshow(img)
            # plt.show()
            # plt.imshow(dst.decode_segmap(labels.numpy()[i]))
            # plt.show()
            # print(np.unique(labels[0].numpy()))
