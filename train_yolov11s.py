from ultralytics import YOLO

# Load the YOLOv11s model with pre-trained weights
model = YOLO('runs/detect/train21/weights/best.pt')  # Load pre-trained weights

# Train the model with specified parameters
model.train(
    data='data/data.yaml',  # Path to your data.yaml file
    epochs=200,             # Number of training epochs (100 initially)
    imgsz=640,              # Reduced image size to 640 (from 1280) to reduce memory
    batch=8,               # Batch size 12
    cache='ram',           # Re-enable cache='ram' for faster local training
    optimizer='AdamW',     # Optimizer to use (AdamW)
    lr0=0.001,              # Initial learning rate
    # Data augmentation parameters (add these lines)
    degrees=10,             # Rotation degree
    translate=0.1,          # Translation ratio
    scale=0.5,              # Scale ratio
    shear=0.0,              # Shear ratio
    perspective=0.0,        # Perspective ratio
    hsv_h=0.015,            # Hue gain
    hsv_s=0.01,             # Saturation gain
    hsv_v=0.01              # Value gain
)
