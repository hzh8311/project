import numpy as np
import os
import cv2

ddir = './data'

split = ['train', 'val', 'test']

for s in split:
    path = os.path.join(ddir, s)
    fs = [_ for _ in os.listdir(path) if _.startswith('default')]
    for f in fs:
        assert f.endswith('jpg')
        im = cv2.imread(os.path.join(path, f))
        h,w,c = im.shape
        im1 = im[0:300, 0:300, :]
        cv2.imwrite(os.path.join(path, 's1_'+f), cv2.resize(im1, (224,224)))
        im2 = im[0:300, 300:600, :]
        cv2.imwrite(os.path.join(path, 's2_'+f), cv2.resize(im2, (224,224)))
        im3 = im[0:300, 600:900:, :]
        cv2.imwrite(os.path.join(path, 's3_'+f), cv2.resize(im3, (224,224)))
        im4 = im[300:600, 0:300:, :]
        cv2.imwrite(os.path.join(path, 's4_'+f), cv2.resize(im4, (224,224)))
        im5 = im[300:600, 300:600, :]
        cv2.imwrite(os.path.join(path, 's5_'+f), cv2.resize(im5, (224,224)))
        im6 = im[300:600, 600:900, :]
        cv2.imwrite(os.path.join(path, 's6_'+f), cv2.resize(im6, (224,224)))
