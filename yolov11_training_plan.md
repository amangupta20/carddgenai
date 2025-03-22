# YOLOv11 Model Training Plan

## 1. Goal

Fine-tune the YOLOv11n model for vehicle damage detection using the car damage dataset located in the `data/` folder. Aim for high mAP@0.5 and acceptable FPS, as indicated in `yolo_v11_documentation.md`. Use the 8 classes defined in `data/data.yaml`.

## 2. Dataset Preparation

- **Dataset Location:** The dataset is located in the `data/raw` directory.
- **Verify Dataset Structure:** Ensure the dataset structure matches the YOLOv8/YOLOv11 format, with `train/`, `val/`, `data.yaml`, etc., inside `data/raw`.
- **Data Augmentation (Automated):** Utilize automated data augmentation techniques during training to improve model generalization and robustness. This will be primarily achieved through:
  - **Ultralytics YOLO Built-in Augmentations:** YOLOv11 training process in Ultralytics framework automatically applies a range of augmentations by default. These include mosaic augmentation, mixup augmentation, random affine transformations (rotation, scale, shear, translation), HSV color-space augmentations, and horizontal flipping. These augmentations are configured within the YOLO training process and do not require manual pre-processing of the dataset.
  - **Roboflow Augmentations (Optional and Configurable):** If desired, and for more control over specific augmentations, Roboflow's platform or SDK could be used to pre-process the dataset with specific augmentations before training. For this initial plan, we will rely on Ultralytics built-in augmentations, but Roboflow augmentations can be explored for further refinement. We will initially focus on the Roboflow-native transformations (45° rotation, ±20% brightness) as mentioned in `vehicle_inspection_plan.md` as examples of augmentations that _could_ be configured if we were to customize augmentations beyond the default Ultralytics set.
  - **Consider Synthetic Damage Generation using GANs (future):** Explore synthetic damage generation using GANs for more advanced augmentation in future iterations.
  - **Consider weather/lighting variations (fog, night scenes) (future):** Explore simulating weather and lighting variations for increased robustness in future iterations.
- **Data Splitting:** The dataset is already split into train/val sets (80/20 split). Verify split ratios if needed.
- **`data.yaml` Configuration:** Use the existing `data/data.yaml` located in the `data/raw` directory. This file defines the dataset paths and class information.

## 3. Environment Setup

- **Python Environment:** Set up a Python environment with required libraries: `ultralytics`, `torch` (PyTorch 2.3+), `opencv-python`, `roboflow`.
- **Ultralytics Version:** Update `ultralytics` package to version 11.0.0 or later: `pip install ultralytics>=11.0.0`.
- **PyTorch Version:** Ensure PyTorch version is 2.3+ (check with `python -c "import torch; print(torch.__version__)"`). Upgrade if necessary.
- **GPU Availability:** GPU is highly recommended for training. Verify GPU availability and CUDA setup if using NVIDIA GPU.

## 4. Model Selection and Configuration

- **Base Model:** Use `yolov11n.pt` as the pre-trained base model.
- **Configuration File:** Use `data/data.yaml` for data configuration.
- **Training Command:** Use the recommended training command from `yolo_v11_documentation.md`, with adjustments as needed:
  ```bash
  yolo train model=yolov11n.pt \
  data=data/data.yaml \
  epochs=100 \
  imgsz=1280 \
  batch=16 \
  cache=ram \
  optimizer=AdamW \
  lr0=0.001
  ```
- **Hyperparameter Tuning (Initial):** Start with the recommended hyperparameters. Note that Ultralytics YOLO automatically applies augmentations during training. For initial tuning, focus on adjusting `epochs`, `imgsz`, `batch`, `lr0`, and potentially explore different optimizers. Further augmentation tuning (beyond the defaults) can be considered in later iterations if needed, potentially by exploring custom augmentation settings within the Ultralytics `data.yaml` configuration or by pre-processing the dataset with Roboflow augmentations.

## 5. Training Execution

- **Run Training Command:** Execute the `yolo train ...` command in the terminal from the project root directory.
- **Monitor Training Progress:** Monitor training progress in the terminal output and using Ultralytics training logs. Track metrics like mAP@0.5, precision, recall, training loss, validation loss.
- **Checkpoint Saving:** Ultralytics YOLO automatically saves model checkpoints during training.

## 6. Evaluation and Validation

- **Validation Set Performance:** Evaluate the trained model's performance on the validation set using metrics like mAP@0.5, precision, recall.
- **Qualitative Evaluation:** Visually inspect predictions on validation images to assess the model's qualitative performance.
- **Compare to Benchmarks:** Compare the achieved mAP@0.5 and FPS to the benchmarks provided in `yolo_v11_documentation.md`.

## 7. Iteration and Refinement (if needed)

- **Hyperparameter Tuning (Advanced):** If initial results are not satisfactory, perform more extensive hyperparameter tuning, including exploring augmentation-related hyperparameters if needed.
- **Data Augmentation Refinement:** Adjust data augmentation strategies, potentially exploring custom augmentations or Roboflow pre-processing.
- **Model Architecture (Advanced):** Consider experimenting with larger YOLOv11 models if performance is still insufficient.
- **Error Analysis:** Analyze failure cases to identify areas for improvement.

## 8. Deployment Preparation

- **Select Best Checkpoint:** Choose the best model checkpoint based on validation performance.
- **Export Model (Optional):** Export the trained model to formats like ONNX or TensorRT for optimized inference if needed for deployment.
