# 🐱🐭 Tom & Jerry Image Classification using CNN

## 📌 Project Overview
This project focuses on building a **Convolutional Neural Network (CNN)** to classify images from the popular cartoon **Tom and Jerry**.  
The model identifies the presence of characters in an image and classifies it into **four distinct categories**.

The primary goal of this project is to demonstrate **practical deep learning skills** using **TensorFlow and Keras**, covering:
- Data preprocessing
- CNN architecture design
- Model training and validation
- Overfitting handling
- Model evaluation and inference

This project is well-suited for showcasing **computer vision fundamentals** and **multi-class image classification**.

---

## 🎯 Problem Statement
Given an input image extracted from **Tom & Jerry cartoon episodes**, the model predicts **which characters appear in the frame**.

This is a **multi-class image classification problem** with the following classes:

- **Tom** – Image contains only Tom  
- **Jerry** – Image contains only Jerry  
- **Tom & Jerry** – Image contains both characters  
- **Neither** – Image contains neither Tom nor Jerry  

---

## 📊 Dataset Description

- **Total images**: 5,478  
- **Source**: Kaggle  
- **Frame extraction**: 1 frame per second (1 FPS) from video clips  
- **Labeling**: Manually labeled (100% ground-truth accuracy)

### 📁 Dataset Structure

tom_and_jerry/
│
├── tom/ # Images containing only Tom<br>
├── jerry/ # Images containing only Jerry<br>
├── tom_jerry_1/ # Images containing both Tom & Jerry<br>
└── tom_jerry_0/ # Images containing neither character



### 🔗 Dataset Link
Kaggle Dataset:  
https://www.kaggle.com/datasets/balabaskar/tom-and-jerry-image-classification

---

## 🧠 Model Architecture

Two CNN architectures were experimented with during this project.

### 🔹 Model 1 (Baseline CNN)
- Multiple convolutional layers with ReLU activation
- MaxPooling layers for spatial reduction
- Fully connected dense layers
- Softmax output for 4-class classification

### 🔹 Model 2 (Improved CNN – Final Model)
- Optimized number of convolution filters
- Dropout layer added to reduce overfitting
- Better generalization performance on validation data

✔ **Model 2 achieved better validation accuracy and reduced overfitting**, and was selected as the final model.

---

## ⚙️ Training Details

- **Framework**: TensorFlow & Keras  
- **Input image size**: 224 × 224  
- **Loss function**: Categorical Crossentropy  
- **Optimizer**: Adam  
- **Evaluation metrics**: Accuracy  
- **Data augmentation**:
  - Rotation
  - Width & height shift
  - Zoom
  - Horizontal flip  

- **Random seed fixed** for reproducibility

---

## 📈 Results

- Training accuracy increased consistently
- Validation accuracy stabilized with reduced overfitting
- Best model checkpoint selected based on **validation loss**
- Model performs well on unseen images, with reasonable confidence scores

> Occasional misclassifications are expected due to challenging frames, occlusions, and distortions — reflecting real-world conditions.

---

## 🔍 Model Inference

The trained model can:
- Predict the class of a given image
- Display the predicted label along with confidence score
- Visualize predictions on unseen test images

---

## 🧪 Key Learnings

- Importance of **train-validation split**
- Handling **multi-class classification**
- Preventing overfitting using **Dropout**
- Effect of **data augmentation**
- Understanding validation metrics over raw training accuracy

---

## 🚀 Future Improvements

- Apply **transfer learning** (e.g., MobileNetV2, ResNet)
- Perform detailed **error analysis**
- Implement **confusion matrix & class-wise accuracy**
- Extend to **object detection** instead of classification

---

## 🛠 Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Google Colab

---

## 📌 Repository Purpose
This repository is intended for:
- Demonstrating practical CNN implementation
- Showcasing deep learning and computer vision skills
- Portfolio and learning reference

---

## 🙌 Acknowledgements
- Kaggle dataset contributors
- TensorFlow & Keras documentation
- Open-source deep learning community

---

