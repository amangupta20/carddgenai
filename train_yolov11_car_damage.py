from ultralytics import YOLO
import os
import torch
import argparse
from pathlib import Path
import psutil  # For CPU core detection
import re

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
# Calculate optimal number of workers (typically 4-8 per CPU core)
CPU_COUNT = os.cpu_count() or psutil.cpu_count(logical=True) or 8
WORKERS = min(CPU_COUNT, 16)    # Use at most 16 workers to avoid diminishing returns
RESUME = False                  # Whether to resume training from last checkpoint
PIN_MEMORY = True               # Pin CPU memory for faster data transfer to GPU

# Output configuration
PROJECT = 'runs/car_damage'     # Project directory
NAME_BASE = 'train'             # Base experiment name (will be auto-incremented)
# =====================================

def get_next_run_name(project_dir, base_name):
    """
    Automatically increments the run name to avoid overwriting previous runs.
    For example, if 'train' exists, it will use 'train2'. If 'train2' exists, it will use 'train3', etc.
    
    Args:
        project_dir: Project directory
        base_name: Base name for the run
        
    Returns:
        Incremented run name
    """
    # Make sure the project directory exists
    os.makedirs(project_dir, exist_ok=True)
    
    # Get all directories in the project directory
    existing_runs = [d for d in os.listdir(project_dir) if os.path.isdir(os.path.join(project_dir, d))]
    
    # If no existing run with base_name, return base_name
    if base_name not in existing_runs:
        return base_name
    
    # Find all runs with the pattern base_name + number
    pattern = re.compile(f"^{re.escape(base_name)}(\\d*)$")
    numbered_runs = [run for run in existing_runs if pattern.match(run)]
    
    if not numbered_runs:
        return f"{base_name}2"  # Start with 2 if only the base name exists
    
    # Extract numbers and find the highest
    max_number = 1  # Default to 1, will become 2 after increment
    for run in numbered_runs:
        match = pattern.match(run)
        if match and match.group(1):  # If there's a number suffix
            try:
                number = int(match.group(1))
                max_number = max(max_number, number)
            except ValueError:
                continue
    
    return f"{base_name}{max_number + 1}"

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
    parser.add_argument('--name', type=str, default=None, help='Experiment name (will be auto-incremented if not specified)')
    parser.add_argument('--resume', action='store_true', default=RESUME, help='Resume training from last checkpoint')
    parser.add_argument('--pretrained', type=str, default=PRETRAINED, help='Path to pretrained model')
    parser.add_argument('--pin-memory', action='store_true', default=PIN_MEMORY, help='Pin memory for faster data transfer')
    parser.add_argument('--force-overwrite', action='store_true', help='Force overwrite existing run with same name')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Auto-increment run name if not explicitly provided via command line
    if args.name is None:
        args.name = get_next_run_name(args.project, NAME_BASE)
    elif not args.force_overwrite:
        # If name is explicitly provided but we don't want to overwrite
        args.name = get_next_run_name(args.project, args.name)
    
    print(f"Using run name: {args.name}")
    
    # Print system information
    print(f"CPU cores: {CPU_COUNT}, using {args.workers} workers for data loading")
    
    # Print GPU memory info if available
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Available CUDA devices: {device_count}")
        for i in range(device_count):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"Memory allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
            print(f"Memory reserved: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
            print(f"Memory reserved (cached): {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
            
            # Get GPU total memory for info
            gpu_properties = torch.cuda.get_device_properties(i)
            print(f"Total GPU memory: {gpu_properties.total_memory / 1e9:.2f} GB")
    else:
        print("CUDA is not available. Using CPU for training (not recommended).")
    
    # Check if data.yaml exists
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file not found: {args.data}")
    
    # Load model
    print(f"Loading {'pretrained ' + args.pretrained if args.pretrained else 'base ' + args.model} model...")
    model = YOLO('yolo11s.pt')

    # Optimization info
    print("\nTraining with CPU+GPU optimization:")
    print(f"- Using {args.workers} CPU workers for data loading")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Image size: {args.img_size}")
    print(f"- Mixed precision: enabled")
    print(f"- Half precision: enabled")
    print(f"- Pin memory: {args.pin_memory}")
    print(f"- Cache: ram")
    print(f"- Output directory: {os.path.join(args.project, args.name)}")

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
        # CPU+GPU optimization
        pin_memory=args.pin_memory,  # Pin memory for faster CPU->GPU transfer
    )
    
    # Print training results
    print(f"Training complete. Results: {results}")
    print(f"Best model saved at: {Path(args.project) / args.name / 'weights' / 'best.pt'}")
    print(f"Last model saved at: {Path(args.project) / args.name / 'weights' / 'last.pt'}")
    
if __name__ == "__main__":
    main() 