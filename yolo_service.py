import bentoml
from bentoml.io import Image, JSON
from ultralytics import YOLO
import numpy as np

# Load the YOLOv8 model
model = YOLO("best.pt")

# Create a BentoML service
svc = bentoml.Service("yolo8_service")

@svc.api(input=Image(), output=JSON())
def detect(image):
    # Perform object detection using YOLOv8
    results = model.predict(image)  # Using the predict method for predictions
    
    # Check if there are any detected objects
    if results is None or len(results) == 0 or len(results[0].boxes) == 0:
        return []  # Return an empty list if no objects are detected

    detection_list = []

    # Extract boxes from the results
    boxes = results[0].boxes  # This is a Boxes object

    for result in results[0].boxes:  # Предполагается, что results - это список с boxes
        box = {
            "class": int(result.cls),  # Класс объекта
            "confidence": float(result.conf),  # Уверенность
            "x1": float(result.xyxy[0][0]),          # Координата x1
            "y1": float(result.xyxy[0][1]),          # Координата y1
            "x2": float(result.xyxy[0][2]),          # Координата x2
            "y2": float(result.xyxy[0][3]),          # Координата y2
        }
        detection_list.append(box)
    
    return detection_list
