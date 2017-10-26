import os
import sys
import torch
# import visdom
import time
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib
matplotlib.use('Agg')

from torch.autograd import Variable
from torch.utils import data

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.loss import cross_entropy2d
from ptsemseg.metrics import scores
from lr_scheduling import *

def train(trainloader, model, criterion, flip=True):

    model.train()
    end = time.time()

    # Setup visdom for visualization
    # vis = visdom.Visdom()

    # loss_window = vis.line(X=torch.zeros((1,)).cpu(),
    #                        Y=torch.zeros((1)).cpu(),
    #                        opts=dict(xlabel='minibatches',
    #                                  ylabel='Loss',
    #                                  title='Training Loss',
    #                                  legend=['Loss']))


    bar = Bar('Processing', max=len(trainloader))
    for i, (images, labels) in enumerate(trainloader):
        date_time.update(time.time() - end)
        if torch.cuda.is_available():
            images = Variable(images.cuda(0))
            labels = Variable(labels.cuda(0))
        else:
            images = Variable(images)
            labels = Variable(labels)

        iter = len(trainloader)*epoch + i
        poly_lr_scheduler(optimizer, args.l_rate, iter)

        optimizer.zero_grad()
        outputs = model(images)

        loss = cross_entropy2d(outputs, labels, weight=weight)

        loss.backward()
        optimizer.step()

        # vis.line(
        #     X=torch.ones((1, 1)).cpu() * i,
        #     Y=torch.Tensor([loss.data[0]]).unsqueeze(0).cpu(),
        #     win=loss_window,
        #     update='append')

        if (i+1) % 20 == 0:
            print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, args.n_epoch, loss.data[0]))

    # test_output = model(test_image)
    # predicted = train_loader.decode_segmap(test_output[0].cpu().data.numpy().argmax(0))
    # target = train_loader.decode_segmap(test_segmap.numpy())

    # vis.image(test_image[0].cpu().data.numpy(), opts=dict(title='Input' + str(epoch)))
    # vis.image(np.transpose(target, [2,0,1]), opts=dict(title='GT' + str(epoch)))
    # vis.image(np.transpose(predicted, [2,0,1]), opts=dict(title='Predicted' + str(epoch)))
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    torch.save(model, "checkpoints/{}_{}_{}_{}.pkl".format(args.arch, args.dataset, args.feature_scale, epoch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='fcn8s',
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal',
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=300,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=500,
                        help='Height of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=100,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=8,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-5,
                        help='Learning Rate')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1,
                        help='Divider for # of features to use')
    args = parser.parse_args()
    train(args)
