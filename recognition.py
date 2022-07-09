from sys import flags
import cv2
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch
import numpy as np
import random as rng

def layout_contours(path):

    # mean-std normalize the input image (batch-size: 1)
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    sobeled = cv2.GaussianBlur(im, (5, 5), 1)
    sobeled = cv2.Sobel(sobeled, -1, 1, 1)
    thrs = cv2.adaptiveThreshold(sobeled, sobeled.max(), cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, sobeled.mean())
    im2 = cv2.erode(thrs, None, iterations=1)
    im2 = cv2.dilate(im2, None, iterations=1)
    im2 = cv2.GaussianBlur(im2, (7, 7), 5)

    contours, hierarchy = cv2.findContours(image=im2, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    image_copy = cv2.cvtColor(im.copy(), cv2.COLOR_GRAY2RGB)

    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        if boundRect[i][-1] * boundRect[i][-2] < 1500: contours_poly[i] = None

    for i in range(len(contours)):
        if type(contours_poly[i]) == type(None): continue
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(image_copy, contours_poly, i, color, 3)

    plt.imshow(image_copy, cmap = 'gray')
    plt.show()
    
    return None, None


if __name__ == '__main__':

    path = '/home/adri/Desktop/cvc/data/comics/comicbookplus_data/?cid=1786/?dlid=23013/0/4.jpg'
    boxes, img = layout_contours(path)


