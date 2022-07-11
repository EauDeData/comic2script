from curses import noecho
from sys import flags
from tkinter.messagebox import RETRY
import cv2
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch
import numpy as np
import random as rng
import os
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans,vq

def layout_contours(path, image = False):

    # mean-std normalize the input image (batch-size: 1)
    if not image:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else: im = path
    sobeled = cv2.GaussianBlur(im, (5, 5), 1)
    sobeled = cv2.Sobel(sobeled, -1, 1, 1)
    thrs = cv2.adaptiveThreshold(sobeled, sobeled.max(), cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, sobeled.mean())
    im2 = cv2.erode(thrs, None, iterations=1)
    im2 = cv2.dilate(im2, None, iterations=1)
    im2 = cv2.GaussianBlur(im2, (7, 7), 5)

    contours, hierarchy = cv2.findContours(image=im2, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    image_copy = np.zeros_like(cv2.cvtColor(im.copy(), cv2.COLOR_GRAY2RGB))
    contours_mask = np.ones_like(cv2.cvtColor(im.copy(), cv2.COLOR_GRAY2RGB))*255

    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    hierarchy = hierarchy[0]
    for i, (c, h) in enumerate(zip(contours, hierarchy)):
        if h[3] > 0: continue
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        if boundRect[i][-1] * boundRect[i][-2] < 1500: contours_poly[i] = None

    for i in range(len(contours)):
        if type(contours_poly[i]) == type(None): continue
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(image_copy, contours_poly, i, color, -1) # Image Copy Acts As Mask
        cv2.drawContours(contours_mask, contours_poly, i, 0, 3)
    
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE) if not image else path, image_copy, ~contours_mask # Image and Mask


def load_kernels_bank(path):
    kernels = []
    for file in os.listdir(path):
        kernels.append(cv2.imread(path + file, cv2.IMREAD_GRAYSCALE).T)
    return kernels

def visualize(im, mask, cont):

    if len(im.shape) != 3: im = cv2.cvtColor(~im, cv2.COLOR_GRAY2RGB)
    final = 0.7 * im + 0.3 * mask + cont
    final = (final - final.min()) / (final.min() - final.max()) * 255
    
    plt.imshow(final.astype(np.uint8))
    plt.show()

    return None

def bin_mask(mask):
    return np.sum(mask, axis = 2).astype(bool)


def features(image, extractor = cv2.SIFT_create(), num_features = 50):


    keypoints, descriptors = extractor.detectAndCompute(image, None)

    voc,variance=kmeans(descriptors, num_features, 1)
    histogram=np.zeros(num_features,"float32")
    words,distance=vq(descriptors,voc)
    for w in words:
        histogram[w]+=1

    return keypoints, descriptors, histogram

def look_for_faces(im, kernels):

    im = normalize(im)
    convolved = []
    for k in kernels:
        k = normalize(k)
        con = normalize(cv2.filter2D(im, -1, k)).astype(np.uint8)
        thrs = cv2.adaptiveThreshold(con, con.max(), cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, con.mean())

        convolved.append(thrs)

    return sum(convolved)

def normalize(im, max_ = 255, type_ = int):
    return (max_ * (im - im.min()) / (im.max() - im.min())).astype(type_)

if __name__ == '__main__':

    path = '/home/adri/Desktop/cvc/data/comics/comicbookplus_data/?cid=1786/?dlid=23013/0/4.jpg'
    im, mask, cont = layout_contours(path)
    #visualize(im, mask, cont)
    im[~bin_mask(mask) ] = 0
    print(im.shape)
    img=cv2.drawKeypoints(im,features(im)[0],np.zeros_like(im),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img)
    plt.show()


    f = features(im)
    plt.bar(range(len(f[-1])), f[-1])
    plt.show()

