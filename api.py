from fastapi import FastAPI,  File,  UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2,  numpy as np

app = FastAPI(title="YOLOv8 Object Detection API")
model = YOLO("yolov8n.pt")

@app.get("/")
def home():
    return {"message": "YOLOv8 API running", "docs": "/docs"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    results = model(img)
    detections = []
    for r in results[0].boxes:
        detections.append({
            "class": model.names[int(r.cls)],
            "confidence": round(float(r.conf), 3),
            "bbox": [round(float(x), 1) for x in r.xyxy[0].tolist()]
        })
    return JSONResponse({"total": len(detections), "detections": detections})