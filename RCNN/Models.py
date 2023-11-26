import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2, fasterrcnn_mobilenet_v3_large_fpn

class resnet_fasterrcnn():
  def __init__(self,num_classes,ephocs=30,lr=0.01, momentum=0.9):
    #creating faster rcnn model
    self.model = fasterrcnn_resnet50_fpn(pretrained=True) 
    in_features = self.model.roi_heads.box_predictor.cls_score.in_features
    self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    self.epochs = ephocs
    self.lr = lr
    self.momentum = momentum

  def train(self,train_data):
    param = [param for param in self.model.parameters() if param.requires_grad]

    optimizer = torch.optim.SGD(param,lr=0.01,momentum=0.9)

    for epoch in range(self.epochs):
      tot_loss = 0
      self.model.train()
      for img, target in train_data:
          loss_dict = self.model(img, target)
          loss = sum(loss for loss in loss_dict.values())
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          tot_loss_num += loss
      print("eppch:{},loss:{}".format(epoch, tot_loss))

  def save(self,file_path):
    torch.save(self.model, file_path)

class mobilenet_fasterrcnn():
  def __init__(self,num_classes,ephocs=30,lr=0.01, momentum=0.9):
    #creating faster rcnn model
    self.model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True) 
    in_features = self.model.roi_heads.box_predictor.cls_score.in_features
    self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    self.epochs = ephocs
    self.lr = lr
    self.momentum = momentum

  def train(self,train_data):
    param = [param for param in self.model.parameters() if param.requires_grad]

    optimizer = torch.optim.SGD(param,lr=0.01,momentum=0.9)

    for epoch in range(self.epochs):
      tot_loss = 0
      self.model.train()
      for img, target in train_data:
          loss_dict = self.model(img, target)
          loss = sum(loss for loss in loss_dict.values())
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          tot_loss_num += loss
      print("eppch:{},loss:{}".format(epoch, tot_loss))

  def save(self,file_path):
    torch.save(self.model, file_path)

class resnet_fasterrcnn_v2():
  def __init__(self,num_classes,ephocs=30,lr=0.01, momentum=0.9):
    #creating faster rcnn model
    self.model = fasterrcnn_resnet50_fpn_v2(pretrained=True) 
    in_features = self.model.roi_heads.box_predictor.cls_score.in_features
    self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    self.epochs = ephocs
    self.lr = lr
    self.momentum = momentum

  def train(self,train_data):
    param = [param for param in self.model.parameters() if param.requires_grad]

    optimizer = torch.optim.SGD(param,lr=0.01,momentum=0.9)

    for epoch in range(self.epochs):
      tot_loss = 0
      self.model.train()
      for img, target in train_data:
          loss_dict = self.model(img, target)
          loss = sum(loss for loss in loss_dict.values())
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          tot_loss_num += loss
      print("eppch:{},loss:{}".format(epoch, tot_loss))

  def save(self,file_path):
    torch.save(self.model, file_path)