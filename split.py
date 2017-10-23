import numpy as np
import os
import cv2

ddir = './data'

split = ['train', 'val', 'test']

for s in split:
    path = os.path.join(ddir, s)
    fs = os.listdir(path)
    for f in fs:
        assert f.endswith('jpg')
        im = cv2.imread(os.path.join(path, f))
        w,h,c = im.shape
        im1 = im[0:w/2, 0:h/2, :]
        cv2.imwrite(os.path.join(path, 'p1_'+f), im1)
        im2 = im[w/2:, 0:h/2, :]
        cv2.imwrite(os.path.join(path, 'p2_'+f), im2)
        im3 = im[0:w/2, h/2:, :]
        cv2.imwrite(os.path.join(path, 'p3_'+f), im3)
        im4 = im[w/2:, h/2:, :]
        cv2.imwrite(os.path.join(path, 'p4_'+f), im4)
