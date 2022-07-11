import torch
import urllib
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import os
import random


preprocess = transforms.Compose([
    transforms.Resize((512, 512), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def list_comics(base = '/home/adri/Desktop/cvc/data/comics/comicbookplus_data/'):

    comics = []

    for (dir_path, dir_names, file_names) in os.walk(base):

        for file in file_names:
            try:
                if int(file.split('.')[0]) > 10 : continue
            except: continue
            if os.path.exists(dir_path + '/' +  file): comics.append(dir_path + '/' +  file)

    random.shuffle(comics)
    return comics


class ComicsSet(torch.utils.data.Dataset):
    def __init__(self, files) -> None:
        self.files = files
    
    def read_img(self, path):
        return Image.open(path).convert('RGB')
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        return preprocess(self.read_img(self.files[index]))

def split_train_test():
    all_comics = list_comics()
    split = 10 #int(0.9*len(all_comics))
    return ComicsSet(all_comics[:split]), ComicsSet(all_comics[:10]) # This is for overfitting porpouses, delete and do it properly


if __name__ == '__main__':
    print(split_train_test()[0][2].shape)