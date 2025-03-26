from ultralytics import YOLO
import os
import torch
import argparse
from pathlib import Path

# =====================================
# CONFIGURATION - EDIT THESE VARIABLES
# =====================================
# Dataset configuration
DATA_PATH = 'data/data.yaml'    # Path to data.yaml file

# Model configuration 
MODEL_PATH = 'yolo11s.pt'      # Base model or pretrained model path
PRETRAINED = None               # Set to a path to use a pretrained model instead of MODEL_PATH

# Training parameters
EPOCHS = 100                    # Number of training epochs
BATCH_SIZE = 6                  # Batch size for training
IMG_SIZE = 640                  # Image size for training
DEVICE = '0'                    # Device to use (cpu or GPU number)
WORKERS = 8                     # Number of workers for dataloader
RESUME = False                  # Whether to resume training from last checkpoint

# Output configuration
PROJECT = 'runs/car_damage'     # Project directory
NAME = 'main'                  # Experiment name
# =====================================

def parse_args():
    """Parse command-line arguments (only used if you want to override the variables above)"""
    parser = argparse.ArgumentParser(description='Train YOLOv11 model on car damage dataset')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=IMG_SIZE, help='Image size for training')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Model to use for training')
    parser.add_argument('--data', type=str, default=DATA_PATH, help='Path to data.yaml file')
    parser.add_argument('--device', type=str, default=DEVICE, help='Device to use for training (cpu or GPU number)')
    parser.add_argument('--workers', type=int, default=WORKERS, help='Number of workers for dataloader')
    parser.add_argument('--project', type=str, default=PROJECT, help='Project name')
    parser.add_argument('--name', type=str, default=NAME, help='Experiment name')
    parser.add_argument('--resume', action='store_true', default=RESUME, help='Resume training from last checkpoint')
    parser.add_argument('--pretrained', type=str, default=PRETRAINED, help='Path to pretrained model')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Print GPU memory info if available
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Available CUDA devices: {device_count}")
        for i in range(device_count):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"Memory allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
            print(f"Memory reserved: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
            print(f"Memory reserved (cached): {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
    else:
        print("CUDA is not available. Using CPU for training (not recommended).")
    
    # Check if data.yaml exists
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file not found: {args.data}")
    
    # Load model
    print(f"Loading {'pretrained ' + args.pretrained if args.pretrained else 'base ' + args.model} model...")
    model = YOLO('yolo11s.pt')

    # Training parameters optimized for 6GB VRAM
    # - Reduced batch size for 6GB VRAM
    # - Moderate image size (640px)
    # - Mixed precision training enabled
    
    # Start training
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        pose=12.0,
        kobj=1.0,
        nbs=64,  # Nominal batch size
        val=True,
        save=True,
        save_period=10,
        cache='ram',
        rect=False,
        amp=True,  # Enable mixed precision training
        fraction=1.0,
        profile=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        cos_lr=True,
        close_mosaic=10,
        resume=args.resume,
        # Data augmentation parameters
        augment=True,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        # Project settings
        project=args.project,
        name=args.name,
        exist_ok=True,
        deterministic=True,
        # VRAM optimization
        plots=True,
        half=True,  # Use half precision (FP16) for memory savings
        patience=30,  # Early stopping patience
    )
    
    # Print training results
    print(f"Training complete. Results: {results}")
    print(f"Best model saved at: {Path(args.project) / args.name / 'weights' / 'best.pt'}")
    print(f"Last model saved at: {Path(args.project) / args.name / 'weights' / 'last.pt'}")
    
if __name__ == "__main__":
    main() 