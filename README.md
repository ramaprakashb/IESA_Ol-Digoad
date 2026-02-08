# AI-Based SEM Wafer Defect Classification (IESA DeepTech Hackathon)

## Overview
This project presents an **AI-driven multi-class SEM wafer defect classification system** designed to automate defect identification and reduce dependency on manual SEM inspection in semiconductor manufacturing.

The system leverages **deep learning models trained using NXP eIQ AI software** to classify various wafer defects from grayscale SEM images, providing predicted defect classes along with confidence scores.

---

## Problem Statement
Manual SEM wafer inspection is **time-consuming, error-prone, and not scalable** for advanced semiconductor process nodes. Human-based inspection often leads to inconsistency, delayed feedback, and yield loss.

This project aims to address these challenges by developing a **fast, consistent, and scalable AI-based defect classification pipeline**.

---

## Defect Classes
The model is trained to classify the following **12 classes**:

- Bridge  
- Clean  
- CMP Defects  
- Crack  
- LER (Line Edge Roughness)  
- Open Defects  
- Oxide Thickness Variation  
- Particle Contamination  
- Pattern Collapse  
- Residue  
- Strain  
- Via   

---

## Dataset Description
- Total images: ~1200 SEM images  
- Image type: **Grayscale SEM images**
- Dataset sources:
  - SEM images collected from **published research papers**
  - **Gemini AI–generated SEM images** to improve defect diversity and class balance
- Dataset split:
  - **Training:** 70%
  - **Validation:** 15%
  - **Testing:** 15%

> Note: The dataset is not fully uploaded due to size constraints. The repository includes scripts and structure for dataset preparation.

---

## Methodology
1. Dataset preparation and class-wise organization  
2. Image preprocessing (resize, normalization, grayscale handling)  
3. Model training and validation using **NXP eIQ Portal (Image Classification workflow)**  
4. Model architecture: **MobileNetV2**  
5. Export trained model as **ONNX (float32)**  
6. Perform inference on unseen SEM images using **ONNX Runtime**

---

## Tools & Technologies
- **NXP eIQ Portal** – model training, validation, and export  
- **MobileNetV2** – lightweight CNN architecture  
- **ONNX** – model export and deployment format  
- **Python** – preprocessing, inference, evaluation  
- **ONNX Runtime** – model inference  
- **Gemini AI** – synthetic SEM image generation  

---

## Model Performance
- Validation Accuracy: **~84.8%**
- Number of classes: **13**
- Epochs trained: **~63**

> Accuracy may vary depending on dataset composition and class distribution.

---

## Repository Structure
