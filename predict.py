from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')
import cv2
# Load a model
model = YOLO("weights/yolo11l-pose.pt")  # load a custom model
# Predict with the model
results = model.predict(source="datasets/video",save=True)  # predict on an image
