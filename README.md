# AKUDENTAL 🦷  
**Version 1.0**

This repository contains information, tools, and resources related to the **AKUDENTAL** dental panoramic X-ray dataset.

AKUDENTAL is an expert-annotated dataset for **instance segmentation and detection** tasks in dental panoramic radiography, designed to support deep learning research in dental imaging.

---

## 📦 Dataset Overview

- **Images**: 333 high-resolution panoramic dental X-rays  
- **Annotations**: Per-instance polygons and bounding boxes for:
  - Teeth (1–32, FDI convention)
  - Restorative procedures (implants, crowns, fillings, bridges)  
- **Format**: COCO-style JSON annotations  
- **Split**: 5-fold cross-validation partitions included

---

## 🚀 Key Features

- Instance-level polygon annotations for teeth and restorations
- Ground truth for both **bounding box** and **segmentation** tasks
- Designed for **multi-task learning** and **cross-dataset evaluation**
- Includes annotation for both present and missing teeth (gaps)

---

## 🔐 Data Access

The dataset will be made publicly available under a **non-commercial CC BY-NC-SA 4.0** license **following formal publication**.

- ⏳ **Images and full annotation files will be uploaded to this repository after the paper is published.**
- 📄 In the meantime, labels, metadata structure, sample code, and evaluation scripts are provided in this repository.

---

## 📁 Repository Structure

