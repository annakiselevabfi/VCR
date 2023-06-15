from torch.autograd import Variable
from PIL import Image, ImageEnhance, ImageGrab
from ultralytics import YOLO
import shutil
import easyocr
import cv2
import os
import numpy as np
import torch
import pandas as pd

app_path = "D:/project_VCR/yolo/app/"
car_list_route = "database/car_list.xlsx"
sam = "database/"
raw_image = "1.bmp"
yolov8_result = "yolo_result.png"
plate_image = "plate.png"
eng_syms = 'ABEKMHOPCTYX'
rus_syms = 'АВЕКМНОРСТУХ'

# get yolo predictors
yolov8_plates = YOLO(app_path + 'yolov8/yolo_plates.pt')
yolov8_plates.to('cuda')
yolov8_symbols = YOLO(app_path + 'yolov8/yolo_symbols.pt')
yolov8_plates.to('cuda')

# init EasyOCR reader
txt_reader = easyocr.Reader(['ru'])

# get database content
car_list = pd.read_excel(app_path + car_list_route)

# clear results folders
try:
    shutil.rmtree(app_path + 'plate_results')
except:
    print('Plate results weren`t found')

try:
    shutil.rmtree(app_path + 'symbols_results')
except:
    print('Symbols results weren`t found')

# get model predictions
print("Search for car plates...")
results = yolov8_plates.predict(
    source = app_path+ sam + raw_image,
    save_crop = True,
    imgsz=864,
    show = True,
    conf = 0.7,
    project = app_path,
    name = "plate_results",
    verbose = False
)

i = 0
for img_result in os.listdir(app_path + 'plate_results/crops/plate'):
    i+=1
    cropped_name = raw_image.split('.')[0]
    if (i > 1): cropped_name += str(i)
    cropped_name += '.jpg'

    # get cropped image
    print(app_path + 'plate_results/crops/plate/' + cropped_name)
    img = cv2.imread(app_path + 'plate_results/crops/plate/' + cropped_name)
    cropped_image = img.copy()

    #modify image
    cropped_image = cv2.resize(cropped_image, (725, 190))
    cropped_image_modified = Image.fromarray(cropped_image)
    cropped_image_modified = ImageEnhance.Color(cropped_image_modified).enhance(0)
    cropped_image_modified = ImageEnhance.Sharpness(cropped_image_modified).enhance(2)
    cropped_image_modified = ImageEnhance.Contrast(cropped_image_modified).enhance(1.5)

    # Save the cropped image
    cv2.imwrite(app_path + str(i) + '-' + plate_image, cropped_image)
    cropped_image_modified.save(app_path + str(i) + '-modified-' + plate_image)
    cropped_image_modified = np.asarray(cropped_image_modified)

    # read the text from image by EasyOCR
    txt_results = ['.*', '.*']
    try:
        txt_results[0] += txt_reader.readtext(cropped_image_modified, allowlist='АВЕКМНОРСТУХ 0123456789')[0][-2] 
    except: txt_results[0] += ""
    txt_results[0] += '.*'

    # read the text from image by Yolov8
    results = yolov8_symbols.predict(
        source = app_path + str(i) + '-modified-' + plate_image,
        save = True,
        imgsz=736,
        show = True,
        conf = 0.7,
        project = app_path,
        name = "symbols_results",
        verbose = False
    )[0].boxes
    
    symbols = []
    tmpSymbols = []
    avg_width = 0
    def sorting(e):
        return e[0]
    for res in results:
        class_id = res.cls[0].item()
        class_name = yolov8_symbols.names[int(class_id)]
        coords = res.xyxy[0].tolist()
        avg_width += coords[2] - coords[0]
        xcenter = (coords[0] + coords[2]) / 2
        tmpSymbols.append([xcenter, class_name])
    tmpSymbols.sort(key=sorting)
    avg_gap = 1.5 * (avg_width / len(tmpSymbols))
    for t in range(len(tmpSymbols)):
        if (t != 0 and tmpSymbols[t][0] - tmpSymbols[t-1][0] > avg_gap):
            symbols.append(" ")
        symbols.append(tmpSymbols[t][1])

    txt_results[1] += ''.join(symbols)
    txt_results[1] += '.*'
    for j in range(len(txt_results)):
        txt_results[j] = txt_results[j].replace(" ", ".*")
        txt_results[j] = txt_results[j].replace("_", "")  
        txt_results[j] = txt_results[j].replace(".**", "nothing")
        txt_results[j] = txt_results[j].replace("O", "(О|0)")
        txt_results[j] = txt_results[j].replace("0", "(О|0)")
        for k in range(len(eng_syms)):
            txt_results[j] = txt_results[j].replace(eng_syms[k], rus_syms[k])

    print(f"Search matching results from the database (EasyOCR) - {txt_results[0]}")
    car_results = car_list[car_list.number.str.match(txt_results[0])]
    print(car_results, "\n")

    print(f"Search matching results from the database (Yolov8) - {txt_results[1]}")
    car_results = car_list[car_list.number.str.match(txt_results[1])]
    print(car_results, "\n")



