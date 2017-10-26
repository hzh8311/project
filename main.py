import os
import sys
import time
import torch
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
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
from ptsemseg.utils import AverageMeter
from ptsemseg.loggers import Logger
from ptsemseg.models.utils import save_checkpoint, adjust_learning_rate
from lr_scheduling import *

sys.path.append("/data5/huangzh/DAN-torch/pose/progress")
from progress.bar import Bar as Bar


def main(args):
    global best_acc

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    train_loader = data_loader(
        data_path,
        'train',
        is_transform=True,
        img_size=(args.img_rows, args.img_cols))
    val_loader = data_loader(
        data_path,
        'val',
        is_transform=True,
        img_size=(args.img_rows, args.img_cols))
    n_classes = train_loader.n_classes
    trainloader = data.DataLoader(
        train_loader, batch_size=args.batch_size, num_workers=4, shuffle=True)
    valloader = data.DataLoader(
        val_loader, batch_size=args.batch_size, num_workers=4, shuffle=True)

    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)

    print("Create model {}-{}".format(args.arch, args.dataset))
    model = get_model(args.arch, n_classes)

    if torch.cuda.is_available():
        model.cuda(0)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.l_rate, momentum=0.99, weight_decay=5e-4)

    title = args.dataset + '-' + args.arch
    logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
    logger.set_names(
        ['Epoch', 'LR', 'Train Loss', 'Val Loss', 'Train MIOU', 'Val MIOU'])

    cudnn.benchmark = True
    print('Total params: %.2f M' % (sum(p.numel() for p in model.parameters()) /
                                    (1024 * 1024)))

    if args.evaluate:
        print('Evaluation only')
        score, class_iou = validate(valloader, model, cross_entropy2d,
                                    n_classes, args.flip)
        for k, v in score.items():
            print k, v

        for i in range(n_classes):
            print i, class_iou[i]

    lr = args.l_rate
    for epoch in range(args.n_epoch):

        lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule,
                                  args.gamma)
        print('Epoch: %d | LR: %.8f' % (epoch + 1, lr))
        train_loss, train_acc = train(trainloader, model, cross_entropy2d,
                                      optimizer, n_classes, args.flip)
        valid_loss, valid_acc = vaildate(valloader, model, cross_entropy2d,
                                         optimizer, n_classes, args.flip)

        logger.append(
            [epoch + 1, lr, train_loss, valid_loss, train_acc, valid_acc])

        is_best = valid_acc > best_acc
        best_acc = max(valid_acc, best_acc)
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict()
            },
            is_best,
            checkpoint=args.checkpoint)
    logger.close()
    logger.plot(['Train Acc', 'Val Acc'])
    savefig(os.path.join(args.checkpoint, 'log.eps'))


def train(trainloader, model, criterion, optimizer, num_classes=3, flip=True):
    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()
    model.train()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for i, (images, labels) in enumerate(trainloader):
        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            images = Variable(images.cuda(0))
            labels = Variable(labels.cuda(0))
        else:
            images = Variable(images)
            labels = Variable(labels)

        outputs = model(images)

        if flip:
            flip_images_var = Variable(
                torch.from_numpy(fliplr(images.clone().numpy())).float()
                .cuda(0))
            flip_outputs_var = model(flip_images_var)
            flip_outputs_var = flip_back(flip_outputs_var.data.cpu())

        loss = criterion(outputs, labels)

        pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=1)
        gt = labels.data.cpu().numpy()

        score, class_iou = scores(gt, pred, num_classes)
        acc = score.mean_iu

        losses.update(loss.data[0], images.size[0])
        acces.update(acc.data[0], images.size[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
            batch=i + 1,
            size=len(trainloader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            acc=acces.avg)
        bar.next()

    bar.finish()

    # test_output = model(test_image)
    # predicted = train_loader.decode_segmap(test_output[0].cpu().data.numpy().argmax(0))
    # target = train_loader.decode_segmap(test_segmap.numpy())

    # vis.image(test_image[0].cpu().data.numpy(), opts=dict(title='Input' + str(epoch)))
    # vis.image(np.transpose(target, [2,0,1]), opts=dict(title='GT' + str(epoch)))
    # vis.image(np.transpose(predicted, [2,0,1]), opts=dict(title='Predicted' + str(epoch)))
    # if not os.path.exists('checkpoints'):
    #     os.makedirs('checkpoints')
    # torch.save(model, "checkpoints/{}_{}_{}_{}.pkl".format(args.arch, args.dataset, args.feature_scale, epoch))
    return losses.avg, acces.avg


def validate(valloader, model, criterion, num_classes, flip=True):
    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()
    model.eval()

    bar = Bar('Processing', max=len(valloader))
    gts, preds = [], []
    for i, (images, labels) in enumerate(valloader):
        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            images = Variable(images.cuda(0), volatile=True)
            labels = Variable(labels.cuda(0), volatile=True)
        else:
            images = Variable(images)
            labels = Variable(labels)

        outputs = model(images)

        if flip:
            flip_images_var = Variable(
                torch.from_numpy(fliplr(images.clone().numpy())).float()
                .cuda(0))
            flip_outputs_var = model(flip_images_var)
            flip_outputs_var = flip_back(flip_outputs_var.data.cpu())

        pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=1)
        gt = labels.data.cpu().numpy()

        for gt_, pred_ in zip(gt, pred):
            gts.append(gt_)
            preds.append(pred_)

        loss = criterion(outputs, labels)
        score, class_iou = scores(gt, pred, num_classes)
        acc = score.mean_iu

        losses.update(loss.data[0], images.size[0])
        acces.update(acc.data[0], images.size[0])

        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
            batch=i + 1,
            size=len(trainloader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            acc=acces.avg)
        bar.next()

    bar.finish()

    score, class_iou = scores(gts, preds, n_class=n_classes)

    for k, v in score.items():
        print k, v

    for i in range(n_classes):
        print i, class_iou[i]

    # test_output = model(test_image)
    # predicted = train_loader.decode_segmap(test_output[0].cpu().data.numpy().argmax(0))
    # target = train_loader.decode_segmap(test_segmap.numpy())

    # vis.image(test_image[0].cpu().data.numpy(), opts=dict(title='Input' + str(epoch)))
    # vis.image(np.transpose(target, [2,0,1]), opts=dict(title='GT' + str(epoch)))
    # vis.image(np.transpose(predicted, [2,0,1]), opts=dict(title='Predicted' + str(epoch)))
    # if not os.path.exists('checkpoints'):
    #     os.makedirs('checkpoints')
    # torch.save(model, "checkpoints/{}_{}_{}_{}.pkl".format(args.arch, args.dataset, args.feature_scale, epoch))
    return losses.avg, acces.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument(
        '-a',
        '--arch',
        nargs='?',
        type=str,
        default='segnet',
        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
    parser.add_argument(
        '-d',
        '--dataset',
        nargs='?',
        type=str,
        default='ustc',
        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument(
        '--img_rows',
        nargs='?',
        type=int,
        default=300,
        help='Height of the input image')
    parser.add_argument(
        '--img_cols',
        nargs='?',
        type=int,
        default=500,
        help='Height of the input image')
    parser.add_argument(
        '-n',
        '--n_epoch',
        nargs='?',
        type=int,
        default=100,
        help='# of the epochs')
    parser.add_argument(
        '-b', '--batch_size', nargs='?', type=int, default=8, help='Batch Size')
    parser.add_argument(
        '--l_rate', nargs='?', type=float, default=1e-5, help='Learning Rate')
    parser.add_argument(
        '--feature_scale',
        nargs='?',
        type=int,
        default=1,
        help='Divider for # of features to use')
    parser.add_argument(
        '--schedule',
        type=int,
        nargs='+',
        default=[60, 90],
        help='Decrease learning rate at these epochs.')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.1,
        help='LR is multiplied by gamma on schedule.')
    parser.add_argument(
        '-f',
        '--flip',
        dest='flip',
        action='store_true',
        help='flip the input during validation')
    parser.add_argument(
        '-c',
        '--checkpoint',
        default='checkpoint',
        type=str,
        metavar='PATH',
        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument(
        '-e',
        '--evaluate',
        dest='evaluate',
        action='store_true',
        help='evaluate model on validation set')
    args = parser.parse_args()
    main(args)
