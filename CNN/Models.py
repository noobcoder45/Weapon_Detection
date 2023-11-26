import torchvision
import torch

class model():
  def __init__(self,num_classes,device,ephocs=30,lr=0.01, momentum=0.9):
    pass

  def forward():
    pass
  


class base_cnn():
  def __init__(self,num_classes,device,ephocs=30,lr=0.01, momentum=0.9):
    #creating faster rcnn model
    self.model = model()

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