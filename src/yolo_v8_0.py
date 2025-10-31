from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("models/yolov8n.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
