# Weapon_Detection

## YOLO
Folder `YOLO` is a clone of [ulralytics/yolov5](https://github.com/ultralytics/yolov5.git) with some changes.
First clone the repository from [noobcode45/Weapon_Detection](https://github.com/noobcoder45/Weapon_Detection.git)

Download datasets and models from [https://drive.google.com/drive/folders/1U7CdoC5UQ9QmmMAO34Fh85uKQ2s1O4a2?usp=drive_link](https://drive.google.com/drive/folders/1U7CdoC5UQ9QmmMAO34Fh85uKQ2s1O4a2?usp=drive_link)
Put it in the main directory.

Extract the zip file.

Make sure the directory structure is
Weapon_Detection-main  
|_dataset  
    ...  
|_YOLO  

go to YOLO directory

Train the yolov5 model with `train.py` with command given below. Replace `[IMG]`, `[BATCH]` and `[EPOCH]` with Number of epochs respectively.
```
python3 train.py --img [IMG] --batch [BATCH] --epochs [EPOCH] --data ./data.yaml --cfg ./models/yolov5x.yaml --weights yolov5x.pt --name yolov5x_weapon
python3 train.py --img 480 --batch 4 --epochs 30 --data ./data.yaml --cfg ./models/yolov5x.yaml --weights yolov5x.pt --name yolov5x_weapon
```

For validation loss, run the below command with best model.
```
python val.py --weight exp/weights/best.pt --data ./data.yaml
```

Replace `[model]` with path for best model and `[source]` for image path and run the below command for detection.
```
python detect.py --weights [add path of best model] --source [add file path of image]
python detect.py --weights ./runs/train/yolov5x_weapon2/weights/best.pt --source ../Weapon-detection-1/test/images/33_jpeg.rf.9ff4cca04f3a1c2ca32fdc53f26c341d.jpg
```

Saved yolo model can be found with name SavedModel


## RCNN
First clone the repository from [noobcode45/Weapon_Detection](https://github.com/noobcoder45/Weapon_Detection.git)
Go to directory `RCNN`

For Training 
Open rcnn_training.ipynb in Google Colab, upload the dataset, and change the folder path accordingly
For Testing
Open testing_rcnn_models, upload models and test change the 1) Model path and 2) Test image path of your choice and run.
Saved models are also available by downloaded by running the cells.

## Camera
To use camera for RCNN, in the project directory run the following command
```
python3 camera.py --input_path [add path to input video (0 if webcam)] --model_path [path to .pt file] --output_path [path to save the output]
python3 camera.py --input_path ./video1.mp4 --model_path ./WeaponDetection_10_resnet2.0.pt --output_path ./output_video.mp4
```
For YOLO run the detect as mentioned aboce with --source flag set to 0


