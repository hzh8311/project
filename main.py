import os
import sys
import torch
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

def main(args):
    global best

    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)

    model = get_model(args.arch, n_classes)

    if torch.cuda.is_available():
        model.cuda(0)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.l_rate, momentum=0.99, weight_decay=5e-4)

    for epoch in range(args.n_epoch):
        if
