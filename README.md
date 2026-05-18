# Tom & Jerry Image Classification using CNN

## Project Overview
This project focuses on building a Convolutional Neural Network (CNN) to classify images from the popular cartoon Tom and Jerry.  
The model identifies the presence of characters in an image and classifies it into four distinct categories.

---

## Problem Statement
Given an input image extracted from **Tom & Jerry cartoon episodes**, the model predicts **which characters appear in the frame**.

This is a **multi-class image classification problem** with the following classes:

1. Tom – Image contains only Tom  
2. Jerry – Image contains only Jerry  
3. Tom & Jerry – Image contains both characters  
4. Neither – Image contains neither Tom nor Jerry  

---

## Dataset Description

- **Total images**: 5,478  
- **Source**: Kaggle  
- **Frame extraction**: 1 frame per second (1 FPS) from video clips  
- **Labeling**: Manually labeled (100% ground-truth accuracy)

### Dataset Structure

tom_and_jerry/<br>
│<br>
├── tom/ # Images containing only Tom<br>
├── jerry/ # Images containing only Jerry<br>
├── tom_jerry_1/ # Images containing both Tom & Jerry<br>
└── tom_jerry_0/ # Images containing neither character



###  Dataset Link
Kaggle Dataset:  
https://www.kaggle.com/datasets/balabaskar/tom-and-jerry-image-classification

---

## Model Architecture

Two CNN architectures were experimented with during this project.

### 01. Model 1 (Baseline CNN)
- Multiple convolutional layers with ReLU activation
- MaxPooling layers for spatial reduction
- Fully connected dense layers
- Softmax output for 4-class classification

### 02. Model 2 (Improved CNN – Final Model)
- Optimized number of convolution filters
- Dropout layer added to reduce overfitting
- Improved generalization on validation data

Model 2 achieved better validation performance and was selected as the final model.

---

## Training Details

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


---

### Data Augmentation
- Rotation  
- Width & height shift  
- Zoom  
- Horizontal flip  

### Early Stopping & Best Model Selection
To prevent overfitting and ensure optimal generalization, **EarlyStopping** was used during training:

- **Monitored metric**: `val_loss`  
- **Patience**: 10 epochs  
- **Best weights restored automatically**

The training process automatically stopped when validation loss stopped improving, and the model weights were restored to the epoch with the lowest validation loss. The final saved model represents the best-performing version not necessarily the last training epoch.

---

## Results

- Training accuracy increased consistently
- Validation accuracy stabilized with reduced overfitting
- Best model selected based on lowest validation loss
- Model performs well on unseen images with reliable confidence scores

> Occasional misclassifications are expected due to challenging frames, occlusions, and distortions — reflecting real-world conditions.

---

## Model Inference

The trained model can:
- Predict the class of a given image
- Display the predicted label along with confidence score
- Visualize predictions on unseen test images

---

## Future Improvements

- Apply **transfer learning** (e.g., MobileNetV2, ResNet)
- Perform detailed **error analysis**
- Implement **confusion matrix & class-wise accuracy**
- Extend to **object detection** instead of classification

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Google Colab


---

## Acknowledgements
- Kaggle dataset contributors
- TensorFlow & Keras documentation
- Open-source deep learning community

---

