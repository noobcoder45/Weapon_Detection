from Models import resnet_fasterrcnn, resnet_fasterrcnn_v2, mobilenet_fasterrcnn
import cv2
import numpy as np
from numpy.core.defchararray import join, mod
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import os
import torch
from torch._C import device
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.rpn import AnchorGenerator
from PIL import Image
from torchvision import transforms as torchtrans
from sklearn.model_selection import train_test_split
import sys
import argparse

class Images(Dataset):
    def __init__(self,imgs_path,labels_path):

        self.imgs_path = imgs_path
        self.labels_path = labels_path
        self.img_name = [img for img in sorted(os.listdir(self.imgs_path))]
        self.label_name = [label for label in sorted(os.listdir(self.labels_path))]

    def __getitem__(self,idx):

        image_path = os.path.join(self.imgs_path,str(self.img_name[idx]))
        img = cv2.imread(image_path)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = img_rgb/255
        img_res = torch.as_tensor(img_res) 
        img_res = img_res.permute(2, 0, 1)

        label_name = self.img_name[idx][:-4] + "txt"
        label_path = os.path.join(self.labels_path,str(label_name))
        with open(label_path, 'r') as label_file:
            l_count = int(label_file.readline())
            box = []
            for i in range(l_count):
                box.append(list(map(int, label_file.readline().split())))

        target={}
        target["boxes"] = torch.as_tensor(box) 
        area = []
        for i in range(len(box)):

            a = (box[i][2] - box[i][0]) * (box[i][3] - box[i][1])
            area.append(a)
        target["area"] = torch.as_tensor(area) 
        labels = []
        for i in range(len(box)):
            labels.append(1)

        target["image_id"] = torch.as_tensor([idx]) 
        target["labels"] = torch.as_tensor(labels, dtype = torch.int64) 


        return img_res,target

    def __len__(self):
        return len(self.img_name)


def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Faster RCNN trainer')
    parser.add_argument('model', help='Base Model of RCNN')
    args = parser.parse_args()
    print(args)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    batch_size=5
    num_classes=2
    num_epoch = 10
    lr = 0.01
    momentum = 0.9

    IMAGES_PATH = './drive/MyDrive/input/Images'
    LABELS_PATH = './drive/MyDrive/input/Labels'
    VERSION = '1'
    MODEL_SAVE_PATH = './WeaponDetection'+ VERSION + ".pt"
    model = None
    
    if args.model == 'resnet':
        model = resnet_fasterrcnn(2,num_epoch,lr,momentum)

    elif args.model == 'mobilenet':
        model = mobilenet_fasterrcnn(2,num_epoch,lr,momentum)

    elif args.model == 'resnet2':
        model = resnet_fasterrcnn_v2(2,num_epoch,lr,momentum)
    else:
        print("Incorrect model enterred")
        sys.exit(1)

    gun_data = Images(IMAGES_PATH, LABELS_PATH)
    train_data = DataLoader(gun_data, batch_size=5,
                       shuffle=True, num_workers=0, collate_fn=collate_fn)

    
    model.train(train_data)
    model.save(MODEL_SAVE_PATH)