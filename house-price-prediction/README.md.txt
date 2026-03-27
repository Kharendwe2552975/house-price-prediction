# 🏠 House Price Prediction (Machine Learning Project)

## 📌 Overview
This project focuses on building a machine learning model to predict house prices using structured housing data. It demonstrates the full data science workflow, including data preprocessing, exploratory data analysis (EDA), model building, and evaluation.

The goal is to transform raw data into meaningful insights and accurate predictions that can support real-world decision-making in property valuation.

---

## 🎯 Objectives
- Clean and preprocess raw housing data  
- Perform exploratory data analysis to understand trends  
- Build and compare regression models  
- Evaluate model performance using standard metrics  
- Identify key features that influence house prices  

---

## 🛠️ Technologies Used
- **Python**
- **Pandas** – Data manipulation and cleaning  
- **NumPy** – Numerical computations  
- **Matplotlib** – Data visualization  
- **Scikit-learn** – Machine learning models and evaluation  

---

## 📊 Dataset
The dataset contains structured information about houses, such as:
- Number of rooms  
- Location  
- Size (square footage)  
- Additional property features  

> 📌 Note: The dataset can be replaced with any housing dataset (e.g., Kaggle datasets).

---

## 🔍 Project Workflow

### 1. Data Preprocessing
- Removed missing or inconsistent data  
- Encoded categorical variables into numerical format  
- Prepared dataset for machine learning models  

### 2. Exploratory Data Analysis (EDA)
- Analysed relationships between features and house prices  
- Identified key variables influencing predictions  
- Visualized data distributions and correlations  

### 3. Model Building
Two regression models were implemented:
- **Linear Regression** (baseline model)  
- **Random Forest Regressor** (advanced model)  

### 4. Model Evaluation
Models were evaluated using:
- Mean Absolute Error (MAE)  
- Mean Squared Error (MSE)  
- Root Mean Squared Error (RMSE)  
- R² Score  

---

## 📈 Results
- The **Random Forest model** performed better than Linear Regression  
- It achieved lower error rates and improved prediction accuracy  
- Key features such as size and location had the strongest impact on house prices  

---

## ▶️ How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
2. Install dependencies
pip install -r requirements.txt
3. Run the project
python house_price_prediction.py

## 🌐 Web Application (Streamlit)

This project includes an interactive web application built with Streamlit.

### ▶️ Run the app:
```bash
streamlit run app.py