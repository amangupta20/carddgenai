from ultralytics import YOLO

# Load the YOLOv11s model
model = YOLO('yolo11m.pt')

# Train the model with specified parameters
model.train(
    data='data/data.yaml',  # Path to your data.yaml file
    epochs=100,             # Number of training epochs
    imgsz=640,              # Reduced image size to 640 (from 1280) to reduce memory
    batch=12,               # Increased batch size to 12 (from 8) - user request
    cache='ram',           # Cache dataset in RAM for faster training
    optimizer='AdamW',     # Optimizer to use (AdamW)
    lr0=0.001              # Initial learning rate
)