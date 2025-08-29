# Cyber Threat Detection with WGAN Augmentation

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red)

---

## 🔹 Project Overview

This project leverages **Transformer-based deep learning models** and **Wasserstein GANs (WGANs)** to analyze network traffic and generate high-quality synthetic traffic data. The workflow covers:

- **Preprocessing real network traffic datasets**
- **Generating synthetic benign network traffic using WGAN**
- **Training Transformer classifiers for intrusion detection**
- **Predicting network traffic labels**
- **Evaluating model performance**

This tool is designed for network security research, creating synthetic datasets for experimentation, and improving anomaly detection in large-scale traffic.

---

## ⚡ Features

1. **Data Preprocessing**
   - Clean network traffic CSVs
   - Filter invalid or incomplete rows
   - Scale numeric features to [-1, 1] for GANs
   - Encode binary labels for intrusion detection

2. **Synthetic Data Generation (WGAN)**
   - Train **Wasserstein GAN with Gradient Penalty (WGAN-GP)**
   - Generate realistic benign network traffic
   - Inverse scale synthetic data to match original feature ranges

3. **Transformer-based Classification**
   - Deep learning classifier for intrusion detection
   - Binary classification: **Benign** vs **Malicious**
   - Train with both real and synthetic data for improved performance

4. **Evaluation & Metrics**
   - Accuracy, Precision, Recall, F1-score
   - Confusion Matrix for detailed performance analysis
   - Easy integration with new datasets

---
## 📦 Dependencies

Install the following Python packages:

- `torch`  
- `torchvision`  
- `pandas`  
- `numpy`  
- `scikit-learn`  
- `tqdm`  
- `joblib`  


git clone https://github.com/Harisudan-RT/-Cyber-threat-detection-with-WGAN-augmentation.git
cd Cyber-threat-detection-with-WGAN-augmentation
