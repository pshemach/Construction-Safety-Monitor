# Construction Safety Monitor

_Computer Vision System for Construction Site Safety Monitoring_

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-v9c-brightgreen)](https://docs.ultralytics.com/)
[![Roboflow](https://img.shields.io/badge/Roboflow-Dataset-blue)](https://universe.roboflow.com/test-levac/construction-site-safety-jejzr/dataset/1)

## Project Overview

This project implements a **computer vision system** that answers the core question:  
**“Is this situation safe or unsafe?”**

The system:

- Detects workers in a scene
- Recognizes key Personal Protective Equipment (PPE)
- Performs a per-scene compliance check against defined safety rules
- Flags violations clearly with colored bounding boxes and a **SAFE / UNSAFE** verdict

---

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/pshemach/Construction-Safety-Monitor.git
cd Construction-Safety-Monitor
```

Create a virtual environment and install dependencies using **uv**:

```
uv venv
uv sync
```

Activate the environment

Windows:

```
.venv\Scripts\activate
```

Linux / macOS:

```
source .venv/bin/activate
```

Launch app:

```
python app.py
```

---

## Safety Rules Defined

The system enforces the following **clear, enforceable safety rules**:

| Rule                      | Violation Class  | Description                                          | Example Violation              |
| ------------------------- | ---------------- | ---------------------------------------------------- | ------------------------------ |
| Hard Hat Rule             | `NO-Hardhat`     | Every worker must wear a hard hat                    | Worker detected without helmet |
| High-Visibility Vest Rule | `NO-Safety Vest` | Every worker must wear a safety vest in active zones | Worker without vest            |
| Mask Rule (site-specific) | `NO-Mask`        | Mask required in dusty or designated areas           | Worker without mask            |

**Verdict Logic**:

- **SAFE** → No violation classes detected
- **UNSAFE** → Any `NO-*` class is present in the frame

---

## Dataset

**Source**: Roboflow project `construction-site-safety-jejzr` (version 1)  
**Custom Preparation**:

- Downloaded via Roboflow API into Google Drive
- Used **YOLOv9 format** (`data.yaml`)

**Class Distribution** (8 classes):

- `Person`, `Hardhat`, `Safety Vest`, `Gloves`, `Mask`
- Violation classes: `NO-Hardhat`, `NO-Mask`, `NO-Safety Vest`

**Split**:

- Train: 391 images, 3,118 annotations
- Valid: 78 images, 568 annotations
- Test: 52 images, 433 annotations

---

## Model & Training

- **Architecture**: YOLOv9c
- **Environment**: Google Colab (Tesla T4 GPU)
- **Training Notebook**: `Construction_Safety_Monitor_Training.ipynb` (included)

---

## Evaluation

### Validation results (best.pt — val split, 78 images)

| Metric       | Value     |
| ------------ | --------- |
| mAP@0.5      | **0.677** |
| mAP@0.5:0.95 | **0.433** |
| Precision    | **0.857** |
| Recall       | **0.603** |

| Class          | Precision | Recall | mAP@0.5 |
| -------------- | --------- | ------ | ------- |
| Hardhat        | 0.916     | 0.694  | 0.788   |
| Mask           | 0.891     | 0.800  | 0.826   |
| Safety Vest    | 0.887     | 0.641  | 0.720   |
| Person         | 0.902     | 0.665  | 0.751   |
| NO-Hardhat     | 0.801     | 0.576  | 0.664   |
| NO-Safety Vest | 0.864     | 0.593  | 0.690   |
| NO-Mask        | 0.721     | 0.545  | 0.613   |
| Gloves         | 0.872     | 0.311  | 0.360   |

### Test results (best.pt — test split, 52 images)

| Metric       | Value     |
| ------------ | --------- |
| mAP@0.5      | **0.687** |
| mAP@0.5:0.95 | **0.467** |
| Precision    | **0.804** |
| Recall       | **0.540** |

| Class          | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| -------------- | --------- | ------ | ------- | ------------ |
| Hardhat        | 0.960     | 0.623  | 0.801   | 0.557        |
| Mask           | 1.000     | 0.773  | 0.886   | 0.657        |
| Safety Vest    | 0.806     | 0.625  | 0.720   | 0.502        |
| Person         | 0.865     | 0.602  | 0.762   | 0.599        |
| NO-Hardhat     | 0.818     | 0.581  | 0.680   | 0.425        |
| NO-Safety Vest | 0.738     | 0.477  | 0.642   | 0.444        |
| NO-Mask        | 0.741     | 0.426  | 0.603   | 0.274        |
| Gloves         | 0.500     | 0.217  | 0.399   | 0.278        |

---
