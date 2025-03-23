from ultralytics import YOLO
import os

# Load your existing best model from the first run
existing_best_model = 'runs/detect/train21/weights/best.pt'  # Replace with actual path
print(f"Loading existing best model from: {existing_best_model}")

# Stage 1: Continue training from your original best model with light augmentation
model = YOLO(existing_best_model)

results_continued = model.train(
    data='data/data.yaml',
    epochs=150,  # More epochs since we have higher patience
    imgsz=640,
    batch=8,
    cache='ram',
    optimizer='AdamW',
    lr0=0.0005,  # Lower learning rate since we're continuing from a decent model
    lrf=0.0001,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=1,  # Minimal warmup
    cos_lr=True,
    patience=40,  # Much higher patience to prevent early stopping
    
    # Light augmentation since the original model had none
    augment=True,
    fliplr=0.5,     # Horizontal flip
    scale=0.2,      # Mild scale changes
    hsv_h=0.015,    # Slight hue variation
    hsv_s=0.2,      # Mild saturation changes
    hsv_v=0.2,      # Mild brightness changes
    translate=0.05, # Slight position shifts
    
    # Explicitly disable more aggressive augmentation
    mosaic=0.0,     # No mosaic
    mixup=0.0,      # No mixup
    
    # Unfreeze all layers for continued training
    freeze=[],
    
    # Validation and saving
    val=True,
    save=True,
    save_period=10,
    project='runs/car_damage',
    name='continued_aug_training',
)

# Get path to best model from continued training
continued_best_model = os.path.join('runs/car_damage/continued_aug_training/weights/best.pt')

# Stage 2: Fine-tune with higher resolution
print("\nStarting Stage 2: Fine-tuning with increased resolution")
model = YOLO(continued_best_model)

results_highres = model.train(
    data='data/data.yaml',
    epochs=75,      # More epochs
    imgsz=832,      # Higher resolution
    batch=4,        # Smaller batch size
    cache='ram',
    optimizer='SGD', # Switch to SGD for fine-tuning
    lr0=0.0001,     # Very low learning rate
    lrf=0.00001,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=0,
    cos_lr=True,
    patience=30,    # Higher patience here too
    
    # Very minimal augmentation in high-res stage
    augment=True,
    fliplr=0.5,     # Just horizontal flips
    
    # Validation and saving
    val=True,
    save=True,
    save_period=5,
    project='runs/car_damage',
    name='highres_finetuning',
)

print("Training complete. Best model saved at: runs/car_damage/highres_finetuning/weights/best.pt")