from ultralytics import YOLO

try:
    model = YOLO('yolov8n.pt')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
