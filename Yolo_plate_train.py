from ultralytics import YOLO

model = YOLO('D:/project_VCR/yolo/yolov8_plates/weights/best.pt')

if __name__ == '__main__': 
   # Training.
    results = model.predict(
        source = 'D:/project_VCR/yolo/yolov8_plates/1.bmp',
        save = True,
        imgsz=840,
        show = True,
        conf = 0.7)
    
    boxes = results[0].boxes
    box = boxes[0]  # returns one box
    print(boxes.cls)  # cls, (N, )
