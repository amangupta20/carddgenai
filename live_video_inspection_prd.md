# AI-Executable PRD: Live Video Inspection System

## 1. Core Components

```mermaid
graph TD
    A[1920x1080@30FPS Input] --> B[Frame Sampler]
    B -->|Every 15th Frame| C[YOLOv11s Inference]
    C --> D[Damage Aggregator]
    D -->|5s Window| E[Gemini 2.0 Flash]
    E --> F[JSON Report]
    F --> G[Web UI]
```

## 2. Technical Specifications

### 2.1 Computer Vision Module

| Parameter     | Value              | Source                            |
| ------------- | ------------------ | --------------------------------- |
| Model         | YOLOv11s           | yolov11_training_plan.md#L20      |
| Input Size    | 1280x1280          | yolov11_training_plan.md#L33      |
| mAP@0.5       | ≥0.85              | yolov11_training_plan.md#L5       |
| Augmentations | Mosaic, MixUp, HSV | yolov11_training_plan.md#L12      |
| Inference FPS | 2                  | live_video_inspection_plan.md#L36 |

**Training Command:**

```bash
yolo train model=yolov11s.pt data=data/data.yaml epochs=100 imgsz=1280 batch=16 cache=ram optimizer=AdamW lr0=0.001
```

### 2.2 GenAI Interface

```json
{
  "gemini_input": {
    "frame": "base64_jpeg",
    "detections": [
      {
        "class": "dent",
        "confidence": 0.92,
        "bbox": [x1,y1,x2,y2]
      }
    ],
    "timestamp": "ISO8601"
  }
}
```

### 2.3 Hardware Requirements

| Component | Minimum   | Recommended |
| --------- | --------- | ----------- |
| GPU       | NVIDIA T4 | A10G        |
| VRAM      | 8GB       | 24GB        |
| CPU Cores | 4         | 8           |
| RAM       | 16GB      | 32GB        |

## 3. Implementation Sequence

1. **Model Preparation**

   - [ ] Run training command from 2.1
   - [ ] Validate mAP@0.5: `yolo val model=best.pt data=data/data.yaml`
   - [ ] Export to ONNX: `yolo export model=best.pt format=onnx`

2. **Video Pipeline Setup**

   - [ ] Implement frame sampler (15-frame interval)
   - [ ] Configure OpenCV video writer (H264 codec)
   - [ ] Initialize damage aggregation buffer

3. **GenAI Integration**
   - [ ] Create LangChain pipeline for Gemini
   - [ ] Implement exponential backoff for API errors
   - [ ] Set 5s reporting timer with ±200ms tolerance

## 4. Validation Checklist

✅ **Performance Testing**

- [ ] Achieve 2±0.2 FPS inference (30fps input)
- [ ] Process 90% of Gemini calls under 800ms

✅ **Accuracy Checks**

- [ ] Compare live mAP@0.5 vs validation ±3%
- [ ] Manual review of 5% reports

✅ **Failure Modes**

- [ ] Test GPU fallback to CPU
- [ ] Simulate API downtime (5-60s)
