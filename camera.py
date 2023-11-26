import cv2
import numpy as np
import torch
from torch._C import device
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision import transforms as torchtrans
from torchvision.transforms import functional as F
import math 
import sys
import subprocess
import argparse

if __name__ == "__main__":
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	parser = argparse.ArgumentParser(description='Faster RCNN trainer')
	parser.add_argument('model', help='Base Model of RCNN')
	args = parser.parse_args()
	model = None
	if args.model == 'resnet_faster':
		model = torch.load('./RCNN/model/resnet.pt', map_location=device)

	elif args.model == 'mobilenet':
		model = torch.load('./RCNN/model/mobilenet.pt', map_location=device)

	elif args.model == 'resnet2':
		model = torch.load('./RCNN/model/resnet2.pt', map_location=device)

	elif args.model == 'CNN':
		model = torch.load('./CNN/model/cnn.pt')

	elif args.model == 'YOLO':
		subprocess.run(['python3', './YOLO/detect.py', arg1, arg2], check=True)
		
	else:
		print("Incorrect model enterred")
		sys.exit(1)


	model.eval()

	cap = cv2.VideoCapture(0)
	cap.set(3, 640)
	cap.set(4, 480)

	classNames = ["Guns"]

	while True: 
		ret, img= cap.read()
		
		img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
		img_res = img_rgb/255
		img_res = torch.as_tensor(img_rgb).to(device)
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
