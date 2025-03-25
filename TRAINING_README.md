# YOLOv11 Car Damage Detection Training

This repository contains code to train a YOLOv11 model on a dataset of car damage images, optimized for systems with 6GB VRAM.

## Dataset

The dataset contains around 6800 images of car damages, including:
- Broken Windshield
- Front left damaged
- Front right damaged
- Not Damaged
- NumberPlate Broken
- RegNumber
- Smashed-Damaged
- Rear side damaged

## Requirements

```bash
# Install required packages
pip install ultralytics>=11.0.0 torch>=2.3.0 opencv-python
```

## Training Script

The `train_yolov11_car_damage.py` script is configured to train a YOLOv11 model on a 6GB VRAM GPU.

### Key Features:

- **Memory Optimization**: Uses smaller batch size, moderate image size, and half precision for 6GB VRAM
- **Gradient Accumulation**: Simulates larger batch sizes while using less memory
- **Data Augmentation**: Applies various augmentation techniques to improve model generalization
- **Early Stopping**: Uses patience parameter to avoid overfitting
- **Mixed Precision**: Automatically selects the best precision format for your GPU

### Usage:

#### Option 1: Edit Variables in the Script (Recommended)

Open `train_yolov11_car_damage.py` and modify the configuration variables at the top of the file:

```python
# =====================================
# CONFIGURATION - EDIT THESE VARIABLES
# =====================================
# Dataset configuration
DATA_PATH = 'data/data.yaml'    # Path to data.yaml file

# Model configuration 
MODEL_PATH = 'yolov11s.pt'      # Base model or pretrained model path
PRETRAINED = None               # Set to a path to use a pretrained model instead of MODEL_PATH

# Training parameters
EPOCHS = 100                    # Number of training epochs
BATCH_SIZE = 4                  # Batch size for training
IMG_SIZE = 640                  # Image size for training
DEVICE = '0'                    # Device to use (cpu or GPU number)
WORKERS = 8                     # Number of workers for dataloader
RESUME = False                  # Whether to resume training from last checkpoint

# Output configuration
PROJECT = 'runs/car_damage'     # Project directory
NAME = 'train'                  # Experiment name
# =====================================
```

Then run the script:

```bash
# Basic usage
python train_yolov11_car_damage.py
```

#### Option 2: Command Line Arguments (Still Available)

Command line arguments are still available and will override the variables defined in the script:

```bash
# With custom parameters
python train_yolov11_car_damage.py --epochs 150 --batch-size 4 --img-size 640
```

## Tuning for Different VRAM Capacities

### For Less VRAM (4GB):
Modify these variables in the script:
```python
BATCH_SIZE = 2
IMG_SIZE = 480
WORKERS = 4
```

### For More VRAM (8GB+):
Modify these variables in the script:
```python
BATCH_SIZE = 6
IMG_SIZE = 736
```

## Model Evaluation

After training, the model will save checkpoints in the specified project directory. To evaluate the best model:

```bash
# Evaluate the best model
yolo val model=runs/car_damage/train/weights/best.pt data=data/data.yaml
```

## Inference

To run detection on new images:

```bash
# Run detection on a single image
yolo predict model=runs/car_damage/train/weights/best.pt source=path/to/image.jpg

# Run detection on a video
yolo predict model=runs/car_damage/train/weights/best.pt source=path/to/video.mp4
``` 