# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 14:31:32 2019

@author: Dell
"""

from urllib.request import urlretrieve
import os
import numpy as np
import tensorflow as tf
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt


def download():     # download tiger and kittycat image
    categories = ['tiger', 'kittycat']
    for category in categories:
        os.makedirs('./for_transfer_learning/data/%s' % category, exist_ok=True)
        with open('./for_transfer_learning/imagenet_%s.txt' % category, 'r') as file:
            urls = file.readlines()
            n_urls = len(urls)
            for i, url in enumerate(urls):
                try:
                    urlretrieve(url.strip(), './for_transfer_learning/data/%s/%s' % (category, url.strip().split('/')[-1]))
                    print('%s %i/%i' % (category, i, n_urls))
                except:
                    print('%s %i/%i' % (category, i, n_urls), 'no image')


def load_img(path):
    img = skimage.io.imread(path)
    img = img / 255.0
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))[None, :, :, :]   # shape [1, 224, 224, 3]
    return resized_img


def load_data():
    imgs = {'tiger': [], 'kittycat': []}
    for k in imgs.keys():
        dir = './for_transfer_learning/data/' + k
        for file in os.listdir(dir):
            if not file.lower().endswith('.jpg'):
                continue
            try:
                resized_img = load_img(os.path.join(dir, file))
            except OSError:
                continue
            imgs[k].append(resized_img)    # [1, height, width, depth] * n
            if len(imgs[k]) == 400:        # only use 400 imgs to reduce my memory load
                break
    # fake length data for tiger and cat
    tigers_y = np.maximum(20, np.random.randn(len(imgs['tiger']), 1) * 30 + 100)
    cat_y = np.maximum(10, np.random.randn(len(imgs['kittycat']), 1) * 8 + 40)
    return imgs['tiger'], imgs['kittycat'], tigers_y, cat_y

if __name__ == '__main__':
#     download()
#     train()
#    eval()