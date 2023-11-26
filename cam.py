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
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator
from PIL import Image
from torchvision import transforms as torchtrans
import subprocess
from torchvision.transforms import functional as F
import matplotlib.patches as patches
from PIL import Image


import math 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model= torch.load('./weapon_trained_model-1.0a.pt', map_location=device)
model.eval()

cap = cv2.VideoCapture('./video3.mp4')
cap.set(3, 640)
cap.set(4, 480)

classNames = ["Guns"]

while True: 
    ret, img= cap.read()
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_res = img_rgb/255
    img_res = torch.as_tensor(img_res).to(device)
    img_res = img_res.permute(2, 0, 1)

    predictions = model([img_res])
    boxes = predictions[0]['boxes']
    for i in range(boxes.shape[0]):
        box=boxes[0]
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 255), 3)    
    
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()