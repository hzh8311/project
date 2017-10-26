import sys
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import scipy.misc as misc
import torch.nn.functional as F
import torchvision.models as models
import matplotlib
matplotlib.use('Agg')
import os
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.models import get_model
from ptsemseg.metrics import scores

def test(args):

    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    if args.model_path.find('linknet') != -1:
        loader = data_loader(data_path, is_transform=True, img_size=(224, 224))
    else:
        loader = data_loader(data_path, is_transform=True, img_size=(300, 500))
    n_classes = loader.n_classes

    # Setup Model
    model = get_model(args.arch, n_classes)

    print("=> loading checkpoint '{}'".format(args.model_path))
    checkpoint = torch.load(args.model_path)
    args.start_epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})".format(
                args.model_path, checkpoint['epoch']))
    model.eval()

    # Setup image
    testset = [_ for _ in os.listdir('data/test/') if _.startswith('s') and not _.endswith('_gt.jpg')]
    for im in testset:
        img = misc.imread(os.path.join('data/test', im))

        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= loader.mean
        img /= loader.std
        if (img.shape[0], img.shape[1]) != loader.img_size:
            img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]))
            img = img.astype(float) / 255.0
        # NHWC -> NCWH
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img).float()

        # print("=> read input image from : {}".format(args.img_path))

        # if torch.cuda.is_available():
        #     model.cuda(0)
        #     images = Variable(img.cuda(0))
        # else:
        images = Variable(img)

        outputs = model(images)
        pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=1)
        decoded = loader.decode_segmap(pred[0])
        # print np.unique(pred)
        save_dir = os.path.join(os.path.dirname(args.model_path), 'result')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        misc.imsave(os.path.join(save_dir, im), decoded)
    print "Segmentation Mask Saved at: {}".format(save_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('-m', '--model_path', nargs='?', type=str, default='fcn8s_pascal_1_26.pkl',
                        help='Path to the saved model')
    parser.add_argument('-a', '--arch', nargs='?', type=str, default='segnet', help='')
    parser.add_argument('-d','--dataset', nargs='?', type=str, default='ustc',
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('-i', '--img_path', nargs='?', type=str, default=None,
                        help='Path of the input image')
    parser.add_argument('-o', '--out_path', nargs='?', type=str, default=None,
                        help='Path of the output segmap')
    args = parser.parse_args()
    test(args)
