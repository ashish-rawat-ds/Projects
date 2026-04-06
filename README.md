# 💳 Fraud Detection System (Machine Learning Project)

## 📌 Project Overview

This project focuses on detecting fraudulent financial transactions using Machine Learning techniques.
The goal is to build a predictive model that can identify whether a transaction is **fraudulent (1)** or **legitimate (0)**.

The project includes:

* Data Analysis (EDA)
* Feature Engineering
* Handling Imbalanced Data
* Machine Learning Model Training
* Streamlit Web Application for Predictions

---

## 📂 Project Structure

```
Fraud_Detection_Capstone_Project
│
├── Fraud_Detection.ipynb        # Jupyter Notebook (EDA + Model Training)
├── Fraud_Detection_script.py    # Python script for model building
├── fraud_detection.app.py       # Streamlit web application
├── Save_model.plk               # Trained ML model
├── Fraud_Detection_PPTX.pptx    # Project presentation
└── README.md                    # Project documentation
```

---

## 📊 Dataset Description

The dataset contains transaction details such as:

* Step (Time step of transaction)
* Transaction Type
* Amount
* Old Balance (Origin)
* New Balance (Origin)
* Old Balance (Destination)
* New Balance (Destination)

Target variable:

```
isFraud
0 → Not Fraud  
1 → Fraud
```

---

## ⚙️ Feature Engineering

Two important features were created:

```
origDiff = oldbalanceOrg - newbalanceOrig
destDiff = oldbalanceDest - newbalanceDest
```

These features help identify suspicious balance changes.

---

## 🔧 Data Preprocessing

The following preprocessing steps were applied:

* Handling missing values
* Outlier handling using **RobustScaler**
* Label Encoding for categorical variables
* Handling imbalanced dataset using **SMOTE**

---

## 🤖 Machine Learning Models Used

The following algorithms were trained and compared:

* Logistic Regression
* Random Forest
* Decision Tree
* Support Vector Machine (SVM)

Random Forest performed best and was selected as the final model.

---

## 📈 Model Evaluation

Evaluation metrics used:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* ROC-AUC Score

---

## 🌐 Streamlit Web Application

A simple web interface was created using **Streamlit** where users can input transaction details and get fraud predictions.

Example prediction logic used in the app:


The app predicts whether a transaction is **Fraud** or **Not Fraud** along with the probability.

---

## 🚀 How to Run the Project

### 1️⃣ Install dependencies

```
pip install pandas numpy scikit-learn matplotlib seaborn streamlit imbalanced-learn joblib
```

### 2️⃣ Run Streamlit App

```
streamlit run fraud_detection.app.py
```

---

## 📊 Exploratory Data Analysis

EDA includes:

* Fraud vs Non-Fraud distribution
* Transaction type distribution
* Transaction amount distribution
* Balance correlation analysis
* Outlier detection

Example EDA and model pipeline implementation can be found in the training script.


---

## 💻 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit
* SMOTE (Imbalanced-learn)

---

## 👨‍💻 Author

**Ashish Rawat**

Machine Learning Project – Fraud Detection System
