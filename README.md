# Construction Safety Monitor

_Computer Vision System for Real-Time Construction Site Safety Monitoring_

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-v11-brightgreen)](https://docs.ultralytics.com/)
[![Roboflow](https://img.shields.io/badge/Roboflow-Dataset-blue)](https://roboflow.com)

## 📋 Project Overview

This project implements a **real-time computer vision system** that answers the core question:  
**“Is this situation safe or unsafe?”**

The system:

- Detects workers in a scene (any distance/pose)
- Recognizes key Personal Protective Equipment (PPE)
- Performs a per-scene compliance check against defined safety rules
- Flags violations clearly with colored bounding boxes and a **SAFE / UNSAFE** verdict

It was built from scratch using a **custom-curated dataset** (public Roboflow dataset + local preparation in Google Colab) and **YOLOv11** for fast, accurate object detection.

## 🛡️ Safety Rules Defined

The system enforces the following **clear, enforceable safety rules**:

| Rule                      | Violation Class  | Description                                          | Example Violation              |
| ------------------------- | ---------------- | ---------------------------------------------------- | ------------------------------ |
| Hard Hat Rule             | `NO-Hardhat`     | Every worker must wear a hard hat                    | Worker detected without helmet |
| High-Visibility Vest Rule | `NO-Safety Vest` | Every worker must wear a safety vest in active zones | Worker without hi-vis vest     |
| Mask Rule (site-specific) | `NO-Mask`        | Mask required in dusty or designated areas           | Worker without mask            |

**Verdict Logic**:

- **SAFE** → No violation classes detected
- **UNSAFE** → Any `NO-*` class is present in the frame

These rules go beyond the minimum (hard hat + vest) by including mask detection, as supported by the dataset.

## 📊 Dataset

**Source**: Roboflow project `construction-site-safety-jejzr` (version 1)  
**Custom Preparation**:

- Downloaded via Roboflow API into Google Drive
- Used **YOLOv11 format** (`data.yaml`)
- **No additional images were collected** (public dataset was used directly as allowed by challenge guidelines)

**Class Distribution** (8 classes):

- `Person`, `Hardhat`, `Safety Vest`, `Gloves`, `Mask`
- Violation classes: `NO-Hardhat`, `NO-Mask`, `NO-Safety Vest`

**Split**:

- Train: 391 images, 3,118 annotations
- Valid: 78 images, 568 annotations
- Test: 52 images, 433 annotations

**Diversity Included**:

- Indoor/outdoor sites, scaffolding, open lots
- Varying lighting (daylight, shadow, artificial)
- Balanced safe vs. unsafe scenes

## 🧠 Model & Training

- **Architecture**: YOLOv11 (medium variant via Ultralytics)
- **Framework**: Ultralytics YOLOv11
- **Environment**: Google Colab (Tesla T4 GPU)
- **Training Notebook**: `Construction_Safety_Monitor_Training.ipynb` (included)
- **Key Choices**:
  - Transfer learning from COCO-pretrained YOLOv11
  - Confidence threshold = 0.40 for inference (balanced precision/recall)
  - Violation flagging based solely on `NO-*` class detection

**Model Weights Location** (after training):
