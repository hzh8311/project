import torch
import numpy as np

path = './checkpoints/linknet/model_best.pth.tar'

c = torch.load(path)
print c.keys()
print c['state_dict']
