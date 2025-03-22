from ultralytics import YOLO

# Load the YOLOv11s model
model = YOLO('yolo11s.pt')

# Train the model with specified parameters
model.train(
    data='data/data.yaml',  # Path to your data.yaml file
    epochs=100,             # Number of training epochs
    imgsz=640,              # Reduced image size to 640 (from 1280) to reduce memory
    batch=8,                # Reduced batch size to 8 (from 16) to reduce memory
    cache='ram',           # Cache dataset in RAM for faster training
    optimizer='AdamW',     # Optimizer to use (AdamW)
    lr0=0.001              # Initial learning rate
)