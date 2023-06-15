from ultralytics import YOLO

model = YOLO('D:/project_VCR/yolo/yolov8n(anya)6/weights/best.pt')

if __name__ == '__main__': 
   # Training.
    results = model.predict(
        source = 'D:/project_VCR/yolo/obrabotannye2/cringe-13.bmp',
        save = True,
        imgsz=725,
        show = True,
        conf = 0.7)
    #results = model('D:/project_VCR/yolo/obrabotannye2/cringe-13.bmp')
    boxes = results[0].boxes
    box = boxes[0]  # returns one box
    print(boxes.cls)  # cls, (N, )
