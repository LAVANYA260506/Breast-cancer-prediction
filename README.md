# 🩺 Breast Cancer Prediction using Machine Learning

This project is a **machine learning–based web application** that predicts whether a breast tumor is **Benign** or **Malignant** using clinical features.  
It aims to assist in **early detection of breast cancer**, which can significantly improve treatment outcomes.

---

## 📌 Project Overview

Breast cancer is one of the most common cancers worldwide. Early diagnosis plays a crucial role in reducing mortality.  
This project uses a trained machine learning model to analyze medical input features and provide predictions through a simple web interface.

---

## 🚀 Features

- Predicts **Benign** or **Malignant** breast cancer
- Pre-trained ML model for fast predictions
- User-friendly web interface
- Scaled input features for better accuracy
- Easy to run locally

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Machine Learning:** Scikit-learn  
- **Web Framework:** Flask / Streamlit *(update based on what you used)*  
- **Frontend:** HTML, CSS  
- **Data Handling:** Pandas, NumPy  
- **Model Serialization:** Pickle  

---

## 📂 Project Structure

Breast_Cancer/
│
├── app/
│ └── main.py # Web application logic
│
├── main/
│ └── main.py # Model loading and prediction logic
│
├── data/
│ └── data.csv # Dataset used for training
│
├── assets/
│ └── style.css # Styling for the web interface
│
├── model.pkl # Trained ML model
├── scaler.pkl # Feature scaler
└── README.md


---

## 📊 Dataset

- The dataset contains **medical diagnostic features** such as:
  - Radius
  - Texture
  - Perimeter
  - Area
  - Smoothness
- Data is preprocessed and scaled before training.

---

## 🧠 Machine Learning Model

- The model is trained on labeled breast cancer data
- Feature scaling is applied using a saved scaler
- Model is serialized using **Pickle** for reuse

---

## ▶️ How to Run the Project

1. **Clone the repository**
   ```bash
   git clone https://github.com/LAVANYA260506/Breast-cancer-prediction.git
Navigate to the project folder

cd Breast_Cancer
Install required dependencies

pip install -r requirements.txt
Run the application

python app/main.py
Open your browser and access the app (URL will be shown in terminal).

✅ Output
The model predicts whether the tumor is:

Benign

Malignant

📈 Future Improvements
Add more ML models for comparison

Improve UI/UX

Deploy the application on cloud (Render / AWS / Hugging Face)

Add model performance metrics

👩‍💻 Author
Lavanya A
Student | Aspiring ML Engineer

📜 License
This project is for educational purposes.


---


