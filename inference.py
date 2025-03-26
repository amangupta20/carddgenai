import cv2
import numpy as np
import torch
from PIL import Image
import time

# Set your paths here
YOLO_MODEL_PATH = "runs/car_damage/main/weights/best.pt"  # Replace with your model path
INPUT_IMAGE_PATH = "image.png"  # Replace with your image path
OUTPUT_IMAGE_PATH = "output.jpg"  # Where to save the result
CONFIDENCE_THRESHOLD = 0.25  # Minimum detection confidence
IMG_SIZE = 640  # Input size for the model

# Load the YOLO model
print(f"Loading YOLO model from {YOLO_MODEL_PATH}...")
try:
    # Try loading with Ultralytics YOLO (works for YOLOv8+)
    from ultralytics import YOLO
    model = YOLO(YOLO_MODEL_PATH)
    model_type = "ultralytics"
    print("Model loaded with Ultralytics YOLO")
except:
    try:
        # Fallback to PyTorch Hub (YOLOv5)
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_MODEL_PATH)
        model_type = "pytorch"
        print("Model loaded with PyTorch Hub (YOLOv5)")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

# Load and preprocess the image
print(f"Processing image: {INPUT_IMAGE_PATH}")
img = cv2.imread(INPUT_IMAGE_PATH)
if img is None:
    print(f"Error: Could not read image at {INPUT_IMAGE_PATH}")
    exit(1)

# Store original image for later
orig_img = img.copy()
height, width = img.shape[:2]
print(f"Image dimensions: {width}x{height}")

# Convert BGR to RGB (YOLO models expect RGB)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Run inference
print("Running inference...")
start_time = time.time()

# Perform detection
if model_type == "ultralytics":
    # YOLOv8 inference
    results = model(img_rgb, conf=CONFIDENCE_THRESHOLD)
    
    # Process results
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            
            try:
                class_name = result.names[class_id]
            except:
                class_name = f"class_{class_id}"
            
            detections.append({
                'box': [x1, y1, x2, y2],
                'confidence': confidence, 
                'class_id': class_id,
                'class_name': class_name
            })
            
else:  # pytorch (YOLOv5)
    # YOLOv5 inference
    results = model(img_rgb)
    predictions = results.xyxy[0].cpu().numpy()
    
    detections = []
    for x1, y1, x2, y2, conf, cls_id in predictions:
        try:
            class_name = model.names[int(cls_id)]
        except:
            class_name = f"class_{int(cls_id)}"
            
        detections.append({
            'box': [x1, y1, x2, y2],
            'confidence': float(conf),
            'class_id': int(cls_id),
            'class_name': class_name
        })

# Calculate inference time
inference_time = time.time() - start_time
print(f"Inference completed in {inference_time:.2f} seconds")
print(f"Detected {len(detections)} objects")

# Draw results on the original image
for det in detections:
    # Get detection info
    box = det['box']
    x1, y1, x2, y2 = [int(coord) for coord in box]
    class_id = det['class_id']
    class_name = det['class_name']
    confidence = det['confidence']
    
    # Generate a color based on class id
    color = (
        hash(str(class_id)) % 256,
        (hash(str(class_id)) * 2) % 256,
        (hash(str(class_id)) * 3) % 256
    )
    
    # Draw bounding box
    cv2.rectangle(orig_img, (x1, y1), (x2, y2), color, 2)
    
    # Prepare label text
    label = f"{class_name}: {confidence:.2f}"
    
    # Draw label background
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(orig_img, (x1, y1 - 20), (x1 + text_size[0], y1), color, -1)
    
    # Draw label text
    cv2.putText(orig_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Log detection to console
    print(f"Detected {class_name} with confidence {confidence:.4f} at coordinates: [{x1}, {y1}, {x2}, {y2}]")

# Save the output image
cv2.imwrite(OUTPUT_IMAGE_PATH, orig_img)
print(f"Output image saved to {OUTPUT_IMAGE_PATH}")

# Print summary
print("\n--- DETECTION SUMMARY ---")
print(f"Total objects detected: {len(detections)}")
print(f"Inference time: {inference_time:.2f} seconds")

# Display detected classes
class_counts = {}
for det in detections:
    class_name = det['class_name']
    if class_name in class_counts:
        class_counts[class_name] += 1
    else:
        class_counts[class_name] = 1

print("\nDetected classes:")
for class_name, count in class_counts.items():
    print(f"- {class_name}: {count}")