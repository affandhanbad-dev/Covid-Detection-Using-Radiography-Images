
# 🩺 COVID-19 Chest X-Ray Detection using CNN & Flask

This project is a deep learning web application that detects **COVID-19**, **Viral Pneumonia**, and **Normal** chest X-rays using a **Convolutional Neural Network (CNN)** trained on the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database).

---

## 🚀 Project Overview

- **Model:** Custom CNN built with TensorFlow/Keras  
- **Accuracy:** ~96% on test data  
- **Dataset:** COVID-19 Radiography Database (Kaggle)  
- **Frontend:** HTML + CSS (Flask Templates)  
- **Backend:** Flask Web Framework (Python)  
- **Deployment:** Local Flask server  

---

## 🧠 Model Training (Jupyter/Colab)

The CNN model was trained using the Kaggle dataset:
- Images resized to **120×120** pixels  
- Classes: `COVID`, `Normal`, `Viral Pneumonia`  
- Normalized pixel values (0–1)
- Split: 80% train, 10% validation, 10% test  

### Model Architecture:
- 3 Convolution + MaxPooling layers  
- Dropout for regularization  
- Dense layer with ReLU activation  
- Output layer (3 neurons, Softmax activation)  

The final model achieved **96% accuracy** on the test dataset.

---

## 🧩 Flask Web App

After training, the model (`Covid_Radiography_Detection_model.h5`) and label encoder (`Covid_Radio_label_encoder.pkl`) are loaded into a Flask app for real-time prediction.

### Features:
- Upload a **chest X-ray image**
- CNN model predicts the class
- Displays **prediction** and **confidence score**
- Simple and clean UI

---
![main](assets/Covid_radiographymain.png)
