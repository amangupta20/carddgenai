# Car Damage Dataset Documentation

## Key Metadata

```mermaid
graph TD
A[Total Images] --> B[653]
A --> C[Train: 522 (80%)]
A --> D[Val: 131 (20%)]
A --> E[Test: 0]
```

## Class Distribution

| Class          | Instance Count | Examples                                                 |
| -------------- | -------------- | -------------------------------------------------------- |
| Scratch        | 1,842          | ![Example](https://transform.roboflow.com/.../thumb.jpg) |
| Dent           | 1,305          |                                                          |
| Broken Glass   | 893            |                                                          |
| Deformed Parts | 647            |                                                          |

## Annotation Details

- **Format**: YOLOv11-compatible TXT
- **Resolution**: 640x640 (stretched)
- **Augmentations**:
  ```python
  augmentations = {
      'rotation': [-15°, +15°],
      'brightness': ±20%,
      'split_ratio': [80:20:0]
  }
  ```

## Manual Download Instructions

1. Visit [Roboflow Dataset Page](https://universe.roboflow.com/shashidhar-patil/car-damage-dataset/dataset/15)
2. Click "Download" and select "YOLOv11" format
3. Unzip dataset to `project-root/data/raw`
4. Verify folder structure:

```
data/raw/
├── train/
├── val/
├── data.yaml
└── README.dataset.txt
```

```

```
