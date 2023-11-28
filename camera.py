import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import os
from PIL import Image
import matplotlib.pyplot as plt
import subprocess
import matplotlib.patches as patches

# Assuming you have the necessary imports for your model and other libraries here

def main(input_path, model_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create a VideoWriter object for MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for MP4
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

    model = torch.load(model_path, map_location=device)
    model.eval()

    cap.set(3, 640)
    cap.set(4, 480)

    classNames = ["Guns"]

    while True:
        ret, img = cap.read()

        if not ret:
            break  # Break the loop if there are no more frames
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = img_rgb / 255
        img_res = torch.as_tensor(img_res).to(device)
        img_res = img_res.permute(2, 0, 1)

        predictions = model([img_res])
        boxes = predictions[0]['boxes']
        for i in range(boxes.shape[0]):
            box = boxes[0]
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 255), 3)

        out.write(img)
        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Object Detection from Webcam using Faster R-CNN")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input video file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output video file")

    args = parser.parse_args()
    main(args.input_path, args.model_path, args.output_path)

