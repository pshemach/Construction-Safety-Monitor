# Construction Safety Monitor

_Computer Vision System for Real-Time Construction Site Safety Monitoring_

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-v9c-brightgreen)](https://docs.ultralytics.com/)
[![Roboflow](https://img.shields.io/badge/Roboflow-Dataset-blue)](https://universe.roboflow.com/test-levac/construction-site-safety-jejzr/dataset/1)

## 📋 Project Overview

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

## 🛡️ Safety Rules Defined

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

## 📊 Dataset

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

## 🧠 Model & Training

- **Architecture**: YOLOv9c
- **Environment**: Google Colab (Tesla T4 GPU)
- **Training Notebook**: `Construction_Safety_Monitor_Training.ipynb` (included)

---
