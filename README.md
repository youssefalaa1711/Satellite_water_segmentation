# 🛰️ Satellite Water Segmentation using U-Net

This project implements a deep learning pipeline to perform **binary segmentation of water bodies from multi-channel satellite images** using a **U-Net architecture**. The goal is to detect and isolate water regions at the pixel level using satellite data with multiple spectral bands.

---

## 📁 Project Overview

We train a U-Net model to classify each pixel in a satellite image as either **water or non-water**. The input data includes **12 spectral bands**, offering rich information beyond standard RGB, which improves water segmentation performance.

---

## ✅ Key Features

- ✅ U-Net model with ResNet34 encoder (ImageNet pretrained)
- ✅ Supports 12-channel satellite input images
- ✅ Outputs binary water masks
- ✅ Metrics: IoU (Intersection over Union) and F1 score
- ✅ Augmentation with Albumentations
- ✅ Training with early stopping and learning rate scheduling

---

## 🧠 Model Architecture

- **Encoder**: ResNet34 pretrained on ImageNet, modified for 12-channel input
- **Decoder**: U-Net-style upsampling path
- **Output**: 1-channel sigmoid-activated prediction map (water / not water)

---

## 📂 Dataset Preparation

1. **Filter Images**: 
   - Keep only image/mask pairs where the mask contains water pixels (non-zero).
   - Improves training by focusing on relevant samples.

2. **Normalize Input**:
   - Each pixel is scaled between 0 and 1 per image.

3. **Binary Masking**:
   - Original mask is converted to binary (1 for water, 0 for background).

4. **Augmentations**:
   - Random flips and rotations for data diversity during training.

---

## ⚙️ Training Setup

- **Loss**: `BCEWithLogitsLoss` (binary cross entropy with logits)
- **Optimizer**: Adam with learning rate `1e-4`
- **Scheduler**: `ReduceLROnPlateau` to reduce LR on validation loss plateau
- **Metrics**: 
  - **IoU**: Area of overlap / union between prediction and GT
  - **F1-score**: Harmonic mean of precision and recall
- **Early Stopping**: Stops if no improvement for a few epochs

---

## 📈 Results (Example)

After training:
- **Validation IoU**: ~0.63
- **Validation F1**: ~0.71

These scores indicate solid pixel-level agreement between the predicted and ground truth water masks.

---

