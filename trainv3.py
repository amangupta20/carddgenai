from ultralytics import YOLO

# Load your best model from the first stage
model = YOLO('runs/car_damage/continued_aug_training/weights/best.pt')

# Alternative second-stage approach without using ema
results_stage2 = model.train(
    data='/content/drive/MyDrive/CarDataset',
    epochs=75,
    imgsz=640,  # Stay at 640px since higher resolution didn't help
    batch=6,    # Middle ground batch size
    cache='ram',
    optimizer='AdamW',  # Stay with AdamW since it worked well
    lr0=0.0001,
    lrf=0.00001,
    momentum=0.9,
    weight_decay=0.0001,
    warmup_epochs=3,
    cos_lr=True,
    patience=30,
    
    # Very specific augmentation focus
    augment=True,
    scale=0.2,      # Small scale changes
    fliplr=0.5,     # Horizontal flips
    hsv_h=0.01,     # Very slight hue variation
    hsv_s=0.1,      # Very slight saturation
    
    # Validation settings
    val=True,
    save=True,
    save_period=5,
    project='runs/car_damage',
    name='final_stage',
)

print("Training complete. Best model saved at: runs/car_damage/final_stage/weights/best.pt")
