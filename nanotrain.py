from ultralytics import YOLO
import os
import time

# Create a timestamp for the run
timestamp = time.strftime("%Y%m%d_%H%M%S")

# Define project and run names
project_dir = "runs/car_damage_nano"
run_name = f"nano_run_{timestamp}"

print(f"Starting car damage detection training with YOLOv11-nano model")
print(f"Project directory: {project_dir}")
print(f"Run name: {run_name}")

# Load YOLOv11-nano model with pre-trained weights
model = YOLO('yolo11n.pt')  # Nano model has fewer parameters, may generalize better on small datasets

# First training stage with basic configuration and minimal augmentation
print("\n--- Starting Stage 1: Initial Training ---")
results_stage1 = model.train(
    data='data/data.yaml',           # Path to your data configuration
    epochs=250,                      # Train for more epochs with smaller model
    imgsz=640,                       # Standard image size
    batch=16,                        # Can use larger batch with smaller model
    cache='ram',                     # Cache data in RAM for faster training
    optimizer='AdamW',               # AdamW optimizer
    lr0=0.001,                       # Initial learning rate
    lrf=0.0001,                      # Final learning rate
    momentum=0.937,                  # Momentum for optimizer
    weight_decay=0.0005,             # Weight decay for regularization
    warmup_epochs=5,                 # Longer warmup for better stability
    cos_lr=True,                     # Use cosine learning rate scheduler
    patience=50,                     # Higher patience to avoid early stopping
    
    # Minimal augmentation strategy
    augment=True,                    # Enable augmentation
    fliplr=0.5,                      # Horizontal flips (good for car damages)
    scale=0.2,                       # Small scale variations
    hsv_h=0.015,                     # Minimal hue variation
    hsv_s=0.2,                       # Some saturation variation (lighting changes)
    hsv_v=0.2,                       # Some brightness variation
    
    # Prevent more aggressive augmentation
    mosaic=0.0,                      # Disable mosaic - can confuse damage patterns
    mixup=0.0,                       # Disable mixup
    
    # Avoid freezing layers since nano is small and we want full training
    freeze=[],                       # Train all layers
    
    # Validation and saving configuration
    val=True,                        # Perform validation
    save=True,                       # Save checkpoints
    save_period=20,                  # Save every 20 epochs
    project=project_dir,             # Project directory
    name=f"{run_name}_stage1",       # Run name
)

# Get path to best model from first stage
best_model_path = os.path.join(f"{project_dir}/{run_name}_stage1/weights/best.pt")

# Second training stage with focused fine-tuning
print("\n--- Starting Stage 2: Fine-tuning ---")
model = YOLO(best_model_path)        # Load best model from first stage

results_stage2 = model.train(
    data='data/data.yaml',           # Same data configuration
    epochs=100,                      # Additional epochs for fine-tuning
    imgsz=640,                       # Keep same image size
    batch=8,                         # Smaller batch size for fine-tuning
    cache='ram',                     # Cache in RAM
    optimizer='SGD',                 # Switch to SGD for fine-tuning
    lr0=0.0001,                      # Much lower learning rate
    lrf=0.00001,                     # Very low final learning rate
    momentum=0.9,                    # Slightly reduced momentum
    weight_decay=0.0001,             # Lower weight decay
    warmup_epochs=0,                 # No warmup needed for fine-tuning
    cos_lr=True,                     # Cosine LR scheduler
    patience=30,                     # Patience for early stopping
    
    # Very minimal augmentation for fine-tuning
    augment=True,                    # Keep augmentation enabled
    fliplr=0.5,                      # Just horizontal flips
    scale=0.1,                       # Minimal scale variation
    
    # Validation and saving
    val=True,                        # Perform validation
    save=True,                       # Save checkpoints
    save_period=10,                  # Save more frequently
    project=project_dir,             # Same project directory
    name=f"{run_name}_stage2",       # Stage 2 run name
)

best_final_model = os.path.join(f"{project_dir}/{run_name}_stage2/weights/best.pt")
print(f"\nTraining complete!")
print(f"Best model from stage 1: {best_model_path}")
print(f"Best model from stage 2: {best_final_model}")
print(f"\nTo use your model for prediction:")
print(f"    model = YOLO('{best_final_model}')")
print(f"    results = model.predict('path/to/image.jpg', conf=0.25)")