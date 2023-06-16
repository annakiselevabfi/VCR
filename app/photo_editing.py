from roboflow import Roboflow
from torch.autograd import Variable
from EasyOCR import easyocr
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import torch
import pandas as pd
import os

app_path = "F:/Work/Yolov8withCrnn/app/"
plates_folder = "F:/Work/Yolov8withCrnn/app/notcringe/"

content = os.listdir(plates_folder)

for file in content:
    if file.endswith('.bmp') or file.endswith('.png'):
        # crop image
        img = cv2.imread(plates_folder+file)

        #modify image
        img = cv2.resize(img, (725, 190))
        
        # try this №1
        # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        # img = cv2.filter2D(img, -1, kernel)

        # try this №2
        # img = cv2.addWeighted(img, 4, cv2.blur(img, (30, 30)), -4, 128)

        img_modified = Image.fromarray(img)
        img_modified = ImageEnhance.Color(img_modified).enhance(0)
        img_modified = ImageEnhance.Sharpness(img_modified).enhance(2)
        img_modified = ImageEnhance.Contrast(img_modified).enhance(1.5)

        img_modified.save(f'{plates_folder}cringe-{file}')
