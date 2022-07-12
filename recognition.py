from curses import noecho
from sys import flags
from tkinter.messagebox import RETRY
from urllib.parse import _NetlocResultMixinStr
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
import torchvision
from scipy.cluster.vq import kmeans,vq
import torch
import clip
import matplotlib.patches as mpatches

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
    black2 = np.zeros_like(image_copy)
    contours_mask = np.ones_like(cv2.cvtColor(im.copy(), cv2.COLOR_GRAY2RGB))*255

    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    hierarchy = hierarchy[0]
    bbxs = []
    for i, (c, h) in enumerate(zip(contours, hierarchy)):
        if h[3] > 0: continue
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        if boundRect[i][-1] * boundRect[i][-2] < 1500: contours_poly[i] = None

    independent_masks = []
    for i in range(len(contours)):
        if type(contours_poly[i]) == type(None): continue
        bbxs.append(boundRect[i])
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(image_copy, contours_poly, i, color, -1) # Image Copy Acts As Mask
        cv2.drawContours(contours_mask, contours_poly, i, 0, 3)
        independent_masks.append( cv2.rectangle(black2.copy(),(boundRect[i][0], boundRect[i][1]), (boundRect[i][0] + boundRect[i][2], boundRect[i][1] + boundRect[i][3]), 255, -1)/255)
    
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE) if not image else path, image_copy, ~contours_mask, independent_masks, bbxs # Image and Mask


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


def labelVisualize(num_class,color_dict,img, labels):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]

    patches = [ mpatches.Patch(color=(color_dict[i][0] / 255, color_dict[i][1] / 255, color_dict[i][2] / 255), label=labels[i] ) for i in range(len(labels)) ]
    return img_out / 255, patches

def clip_heatmaps(img, masks, bbxs):

    #categories = ["background / texture", "A drawing of a person", "A drawing of group of people (comic style)", "some text", "A drawing of a face (comic syle)", 'Black Background']
    
    #categories = [ "a printed document with the Drawing Of A single person",  "a printed document with the Drawing Of A group of people", "a conversation between people"]
    #categories = [f"{i} - people dialogue" for i in range(6)]
    categories = [f"there's {i} people in this image" for i in range(4)] 


    # NOTE: 0 PEOPLE DIALOGUE DETECTS SOLO TEXT IN CONTRAST TO OTHER SENTENCES
    print(img.max())
    cagegories_count = np.zeros( (len(categories) + 1, img.shape[-2], img.shape[-1]))
    cagegories_count[len(categories), :, :] = 1.
    cetegories_probabilities = np.zeros((len(categories), img.shape[-2], img.shape[-1])) 

    dic = {x: (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)) for x, _, in enumerate(categories)}


    dic[len(categories)] = (0, 0, 0)
    dic[0] = (0, 255, 0)
    dic[1] = (0, 0, 255)
    dic[4] = (255, 0, 0)
    #dic[5] = (0, 0, 0)

    device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)
    text = clip.tokenize(categories).to(device)
    print(clip.available_models())
    print(preprocess)
    norm = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    for i, (x,y,w,h) in zip(masks, bbxs):

        im = img[x:x+w, y:y+h]  
        if im.shape[0] * im.shape[1] == 0: continue


        input_ = cv2.resize(np.stack([im, im, im]).transpose(1, 2, 0), (224, 224))
        image = norm(torch.from_numpy(input_.transpose(2, 0, 1)).to(device)).unsqueeze(0)
        image = image.to(device)
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu()
        argmax = probs.argmax(1).numpy()
        cagegories_count[argmax[0], i[:, :, 0]==1] += 10

    return np.argmax(cagegories_count, axis = 0).astype(float), dic, categories


'''


    image = preprocess(Image.fromarray(img[i:i+step, j:j+step]))
    if image.mean() == 0: continue
    image = image.unsqueeze(0).to(device)
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).squeeze().cpu().argmax().numpy()


'''


def normalize(im, max_ = 255, type_ = int):
    return (max_ * (im - im.min()) / (im.max() - im.min())).astype(type_)

if __name__ == '__main__':

    path = '/home/adri/Desktop/cvc/data/XAC_Data/Costums_i_tradicions/192_10403/2463209.jpg'
    im, mask, cont, masks, bbxs = layout_contours(path)
    #visualize(im, mask, cont)
    #im[~bin_mask(mask) ] = 0
    #print(im.shape)
    #img=cv2.drawKeypoints(im,features(im)[0],np.zeros_like(im),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #plt.imshow(img)
    #plt.show()


    #f = features(im)
    #plt.bar(range(len(f[-1])), f[-1])
    #plt.show()

    map_, cat, labels = clip_heatmaps(normalize(im, max_=1, type_=float), masks, bbxs)
    print(map_)
    colormap, patch = labelVisualize(len(cat), cat, map_, labels)
    imagesum = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) / 255
    print(imagesum.max(), imagesum.dtype)
    plt.imshow(( colormap*.5 + imagesum*.5))
    plt.legend(handles=patch, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

