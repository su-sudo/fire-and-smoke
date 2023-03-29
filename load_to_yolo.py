
#did'nt run it entirely since annotated image is not available


import cv2
#import numpy as np
#import tensorflow as tf
#from keras.models import load_model
import os
from ultralytics import YOLO

model = YOLO("yolov8x-seg.pt")
images =[]
classname = ['fire']
path = 'yolo dataset/archive (1)/Segmentation_Mask/completed'
for img in os.listdir('yolo dataset/archive (1)/Image/Fire'):
        img1 =cv2.imread(f'yolo dataset/archive (1)/Image/Fire/{img}')
        mask = cv2.imread(f'yolo dataset/archive (1)/Segmentation_Mask/Fire/{img}')
        image = cv2.bitwise_and(img1, mask)
        cv2.imwrite(os.path.join(path,img), image)
        images.append(image)
        print(len(images))
        if len(images) == 20:
             break
#cv2.imshow('image',image)
        #cv2.waitKey(0)
print("completed")

#Todo train yolo with images [] export model
#Todo since label is not found it is recommended to use transferlearning or label custom data
#train = model.
#result = model.predict(source=r"Image Dataset\Fire\0.jpg", classes='fire' ,show=True, save=True)
model.train(task='segment', mode='train', epochs=20,model='yolov8n-seg.pt',imgsz=640,data='data_custom.yaml')
#data_custom.yaml

#yolov8n-seg.pt is a segmentation model downloaded from yolo repo