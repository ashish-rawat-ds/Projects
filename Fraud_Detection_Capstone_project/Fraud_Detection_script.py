# Import library

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix,precision_score,recall_score,f1_score,roc_auc_score,roc_curve
import warnings
from joblib import dump


warnings.filterwarnings("ignore")
sns.set(style='whitegrid')

# Load Dataset
df=pd.read_csv("Fraud_Analysis_Dataset.csv")

# Dataset Basic Information
df.info()
df.describe()
df.describe(include="object")

# EDA
# define categorical columns
cat_col=df.select_dtypes(include="object").columns
# define numerical columns
num_col=df.select_dtypes(exclude="object").columns

# 1. Fraud vs Non-Fraud Count

plt.figure(figsize=(7,5))
df['isFraud'].value_counts().plot(kind='bar', color='orange')
plt.title("Fraud vs Non-Fraud Count")
plt.xlabel("Fraud (1 = Yes, 0 = No)")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()
 

# 2. Transaction Type Distribution

plt.figure(figsize=(8,5))
df['type'].value_counts().plot(kind='bar', color='orange')
plt.title("Transaction Type Distribution")
plt.xlabel("Type")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()


# 3. Distribution of Transaction Amounts

plt.figure(figsize=(8,5))
plt.hist(df['amount'], bins=50, color='orange')
plt.title("Distribution of Transaction Amounts")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()


# 4. Old vs New Balance (Origin)

plt.figure(figsize=(8,5))
plt.scatter(df['oldbalanceOrg'], df['newbalanceOrig'], s=5, color='orange')
plt.title("Old Balance vs New Balance (Origin)")
plt.xlabel("Old Balance (Origin)")
plt.ylabel("New Balance (Origin)")
plt.grid(True, linestyle='--', alpha=0.4)
plt.show()


# 5. Old vs New Balance (Destination)

plt.figure(figsize=(8,5))
plt.scatter(df['oldbalanceDest'], df['newbalanceDest'], s=5, color='orange')
plt.title("Old Balance vs New Balance (Destination)")
plt.xlabel("Old Balance (Destination)")
plt.ylabel("New Balance (Destination)")
plt.grid(True, linestyle='--', alpha=0.4)
plt.show()


# 7. Step vs Amount Scatter Plot

plt.figure(figsize=(8,5))
plt.scatter(df['step'], df['amount'], s=5, color='orange')
plt.title("Transaction Amount Across Time Steps")
plt.xlabel("Step")
plt.ylabel("Amount")
plt.grid(True, linestyle='--', alpha=0.4)
plt.show()


# boxplot for outlier detection

plt.figure(figsize=(12,8))

for i, col in enumerate(num_col, 1):
    plt.subplot(2,4, i)      
    sns.boxplot(y=df[col])
    plt.title(col)

plt.tight_layout()
plt.savefig("Outlier.png")
plt.show()


# corr
plt.figure(figsize = (15,7))
plt.title('Correlation of Attributes', y=1.05, size=19)
sns.heatmap(df.corr(numeric_only=True), cmap='plasma',annot=True, fmt='.2f', cbar=False)


# Feature Engineering

# 1. Difference in balances (before - after) for origin and destination
df['origDiff'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['destDiff'] = df['oldbalanceDest'] - df['newbalanceDest']

df.fillna(0, inplace=True)


# Data Preprocessing

# Correlation Of Attributes After Feature Engineering

plt.figure(figsize = (15,7))
plt.title('Correlation of Attributes', y=1.05, size=19)
sns.heatmap(df.corr(numeric_only=True), cmap='plasma',annot=True, fmt='.2f', cbar=False)
plt.savefig("Correlation after feture engineering.png")


# Drop Irrelivent Columns

df.drop(["nameOrig","nameDest","newbalanceDest","oldbalanceDest"],axis=1,inplace=True)


# Handle Missing Values

df.isnull().sum()

# new numerical columns after feature engineering

new_num_col=df.select_dtypes(exclude="object").columns

#  Handle Outlier

plt.figure(figsize=(16,10))

for i, col in enumerate(new_num_col, 1):
    plt.subplot(3,4,i)      
    sns.boxplot(y=df[col])
    plt.title(col)

plt.tight_layout()
plt.show()

# count the outliers from numerical columns
Q1 = df[new_num_col].quantile(0.25)
Q3 = df[new_num_col].quantile(0.75)
IQR = Q3 - Q1
outliers_count = ((df[new_num_col] < (Q1 - 1.5 * IQR)) | (df[new_num_col] > (Q3 + 1.5 * IQR))).sum()

outliers_count

# here we use RobustScaler() to handle outlier.
df[new_num_col]=RobustScaler().fit_transform(df[new_num_col])

# encode the categorical columns
new_cat_col=df.select_dtypes(include="object").columns
for col in new_cat_col:
    df[col]=LabelEncoder().fit_transform(df[col])

# Select X(Independent Feature) and Y(Dependent Feature)

X=df.drop("isFraud",axis=1)
y=df["isFraud"]

# Split Trainning and Testing Data

X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

# As targer is imbalance so here we use oversampling technique which is SMOTE()

smote=SMOTE(k_neighbors=5,sampling_strategy=0.3,random_state=42)
X_train_res,y_train_res=smote.fit_resample(X_train,y_train)

# # Logistic Regression 

# Train Model

logistic_model=LogisticRegression()
logistic_model.fit(X_train_res,y_train_res)

# * Model Evaluation

print(accuracy_score(y_test,logistic_model.predict(X_test)))
print("Training Accuracy:", accuracy_score(y_train_res, logistic_model.predict(X_train_res)))
print("Testing Accuracy:", accuracy_score(y_test, logistic_model.predict(X_test)))
print(classification_report(y_test,logistic_model.predict(X_test)))

# # Random Forest

# Train Model

ran_for_model=RandomForestClassifier()
ran_for_model.fit(X_train_res,y_train_res)

# Model Evaluation

print(accuracy_score(y_test,ran_for_model.predict(X_test)))
print("Training Accuracy:", accuracy_score(y_train_res, ran_for_model.predict(X_train_res)))
print("Testing Accuracy:", accuracy_score(y_test, ran_for_model.predict(X_test)))
print(classification_report(y_test,ran_for_model.predict(X_test)))

# Confusion metrix plot

cm = confusion_matrix(y_test, ran_for_model.predict(X_test))
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest - Confusion Matrix")
plt.show()

# feature importance plot
importance=ran_for_model.feature_importances_
importance
plt.figure(figsize=(10,6))
sns.barplot(x=importance, y=X_train_res.columns)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# # Decision Tree

# Train Model

Dec_tree_model=DecisionTreeClassifier()
Dec_tree_model.fit(X_train_res,y_train_res)

# Model Evaluation

print(accuracy_score(y_test,Dec_tree_model.predict(X_test)))
print("Training Accuracy:", accuracy_score(y_train_res, Dec_tree_model.predict(X_train_res)))
print("Testing Accuracy:", accuracy_score(y_test, Dec_tree_model.predict(X_test)))
print(classification_report(y_test,Dec_tree_model.predict(X_test)))


# # SVM

# Train Model

svc_model=SVC()
svc_model.fit(X_train_res,y_train_res)

# Model Evaluation

print(accuracy_score(y_test,svc_model.predict(X_test)))
print("Training Accuracy:", accuracy_score(y_train_res, svc_model.predict(X_train_res)))
print("Testing Accuracy:", accuracy_score(y_test, svc_model.predict(X_test)))
print(classification_report(y_test,svc_model.predict(X_test)))


# Save model
dump((ran_for_model,RobustScaler()),"Save_model.plk")




