# 🚦 Smart Traffic Volume Prediction System

A full-stack Machine Learning application that predicts traffic volume using historical traffic data and temporal features. The project compares multiple regression models, deploys the best-performing model through an interactive Streamlit dashboard, and supports user authentication, analytics, and batch prediction.

---

## 📌 Overview

Traffic congestion is one of the major challenges faced by modern cities. Accurate traffic volume prediction helps transportation authorities optimize traffic signal timings, reduce congestion, and improve urban mobility.

This project builds a machine learning pipeline that:

- Cleans and preprocesses traffic data
- Performs feature engineering
- Trains multiple regression models
- Selects the best model based on evaluation metrics
- Deploys the model using Streamlit
- Supports secure user authentication
- Allows single and batch predictions

---

## ✨ Features

- 🔐 User Registration & Login
- 🚗 Traffic Volume Prediction
- 📊 Interactive Traffic Analytics
- 📁 Batch Prediction using CSV Upload
- 🌲 Random Forest Based Prediction Model
- 📈 Feature Importance Visualization
- 💾 SQLite Database for Authentication
- 🔒 Password Hashing using bcrypt
- ⚡ Fast and Lightweight Streamlit Interface

---

## 🏗️ System Architecture

```
Traffic Dataset
      │
      ▼
Data Preprocessing
      │
      ▼
Feature Engineering
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Best Model Selection
      │
      ▼
Streamlit Web Application
      │
      ▼
Prediction • Analytics • Batch Processing
```

---

## 🧠 Machine Learning Pipeline

### Data Preprocessing

- Missing value handling
- Duplicate removal
- Datetime conversion
- Data validation

### Feature Engineering

Features extracted from datetime:

- Hour
- Day
- Month
- Weekday
- Rush Hour Indicator

### Models Compared

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

### Evaluation Metrics

- Mean Squared Error (MSE)
- R² Score

The Random Forest model achieved the best performance and was selected for deployment.

---

## 📊 Technology Stack

### Programming Language

- Python 3.x

### Machine Learning

- Scikit-Learn
- Pandas
- NumPy
- Joblib

### Visualization

- Matplotlib

### Web Framework

- Streamlit

### Database

- SQLite

### Security

- bcrypt

---

## 📂 Project Structure

```
Smart-Traffic-Volume-Prediction-System/
│
├── app.py
├── Solution.ipynb
├── model/
├── dataset/
├── images/
├── users.db
├── requirements.txt
├── README.md
└── assets/
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/Sarath-Patti/Smart-Traffic-Volume-Prediction-System.git
```

Move into the project

```bash
cd Smart-Traffic-Volume-Prediction-System
```

Create virtual environment

```bash
python -m venv venv
```

Activate virtual environment

### Windows

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📸 Application Screens

- Login Page
- Registration Page
- Prediction Dashboard
- Analytics Dashboard
- Batch Prediction
- Feature Importance Graphs

*(Add screenshots inside an `images/` folder and update this section.)*

---

## 📈 Model Performance

| Model | Performance |
|--------|------------|
| Linear Regression | Baseline |
| Decision Tree | Improved |
| ⭐ Random Forest | Best Accuracy |

---

## 🔒 Authentication

The application includes:

- User Registration
- Secure Login
- Password Hashing using bcrypt
- SQLite Database Storage

---

## 📁 Batch Prediction

Users can upload a CSV file containing multiple traffic records and download the predicted traffic volumes.

---

## 🎯 Applications

- Smart Cities
- Traffic Management
- Urban Planning
- Highway Monitoring
- Transportation Analytics
- Intelligent Transportation Systems (ITS)

---

## 🚀 Future Improvements

- Real-Time Traffic Prediction
- Weather API Integration
- Google Maps Integration
- Deep Learning Models (LSTM)
- XGBoost / LightGBM
- Cloud Deployment
- Docker Support
- REST API
- Mobile Application

---

## 📚 References

- Scikit-Learn Documentation
- Streamlit Documentation
- Random Forest Research Paper
- Pandas Documentation

---

## 👨‍💻 Author

**Sarath Patti**

M.Tech, Computer Science & Engineering

National Institute of Technology Rourkela

GitHub: https://github.com/Sarath-Patti

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!