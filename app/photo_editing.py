from roboflow import Roboflow
from torch.autograd import Variable
from EasyOCR import easyocr
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import torch
import pandas as pd
import os

app_path = "D:/project_VCR/yolo/app/"
plates_folder = "D:\project_VCR\obrabotannye"

content = os.listdir(plates_folder)

for file in content:
    if file.endswith('.bmp') or file.endswith('.png'):
        # crop image
        img = cv2.imread(plates_folder+file)

        #modify image
        img = cv2.resize(img, (725, 190))
        img_modified = Image.fromarray(img)
        img_modified = ImageEnhance.Color(img_modified).enhance(0)
        img_modified = ImageEnhance.Sharpness(img_modified).enhance(2)
        img_modified = ImageEnhance.Contrast(img_modified).enhance(1.5)

        img_modified.save(f'{plates_folder}editing-{file}')
