# UDA4HeartSeg

This repository provides the official implementation of our paper:

**"Unsupervised Domain Adaptation for Multi-View Echocardiographic Segmentation via Gaussian Contrast and Masked Consistency Learning"**

## 📌 Overview

Multi-view echocardiographic segmentation faces challenges from cross-view structural variations, limited annotations, and inconsistent acquisition methods. We propose an unsupervised domain adaptation framework based on a teacher-student architecture with two key innovations:

- **Gaussian Projection Contrastive Learning (GCL)** – Enhances cross-domain category discrimination by modeling intrinsic cardiac feature relationships across different viewing angles.
- **Masked Uncertainty-aware Consistency Learning (MUCL)** – Enforces consistency between target-view cardiac appearance and pseudo-labels via masked region reasoning.

## 🧪 Dataset

Evaluated on echocardiographic datasets comprising **688 patients** from both internal and external sources. Experiments include:

- Inter-view transfer (e.g., apical ↔ parasternal views)
- Cross-dataset transfer (generalization across diverse adaptation scenarios)

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/XuuuuJH-OvO/UDA4HeartSeg.git
cd UDA4HeartSeg
pip install -r requirements.txt
```

### Pre-trained Model

Create the `pretrained` folder and download the `mit_b5.pth` pre-trained weight file:

```bash
mkdir pretrained
wget -O pretrained/mit_b5.pth https://download.pytorch.org/models/mit_b5.pth
```
