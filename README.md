# My-portfolio-
My first project in python
# Credit Risk Prediction for Prospera Bank
### 📌Introduction
predicting credit risk; assessing the likelihood a customer will default on loans, is the most earliest and impactful application of ML(Machine learning). Given the of imbalance in the default and non-default debtors, this project seek address the class imbalance problem using classification algorithms. Accurate credit risk prediction system can highly improve the profit of financial institutions like Prospera Bank by identifying risky debtors and optimising credit approval strategies.
### 📊Data Source
**Source:** (https://Kaggle.com)
**Provider:** Dataleum
**Entries:** 32,581 records
**Variables:** 12 features (numerical + categorical data)
**Target Variable:** loan_status (1 = Default, 0 = Repayment)
### ⚙️Data Preprocessing
**Data Cleaning:** Took care of the missing values and relevant variales selected
**Data Transformation:** Label enconding and feature scaling
**Data Partitioning:** 80% training set, 20% test set
**Handling Class Imbalance:** Addressed using appropriate model evaluation metric (Precession, Recall, F1, AUC_score)
### 🛠️Libraries Used
```Python
# Data Manipulation & Visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

# Preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Evaluation
from sklearn.metrics import classification_report, roc_auc_score
```

###🧠Algorithms Used
1. Logistic Regression
2. Random Forest
3. XGBoost
Each model was teained to classify loan_status into either default or repayment

### 📈Summary of Results
Metric	Logistic Regression|	Random Forest|	XGBoost
Accuracy	82%	                  83%	           82%
AUC Score	0.77	                0.80           0.74
Precision (Class 1)	0.69	      0.74	         0.65
Recall (Class 1)	0.32	        0.45	         0.30
F1-Score (Class 1)	0.43	      0.56	         0.41

📝 Class 1 = Default, Class 0 = Repayment
### Observations:
1. Random Forest outperformed other models across most metrics.
2. Logistic Regression and XGBoost had comparable accuracy, but struggled more with the minority class (defaults).
3. Class imbalance significantly impacted recall and F1-scores for the default class.

### 📌Recommendations
1. Set loan limits for high-risk borrowers to reduce exposure.
2. Define stricter loan approval criteria using model outputs.
3. Track customer repayment behaviors to detect early warning signs.
4. Re-train models regularly using updated customer data to improve prediction accuracy.

### 🙌Acknowledgements
Special thanks to Dataleum for curating and sharing the dataset via Kaggle.

This project is a part of a machine learning initiative aimed at enhancing risk management for a financial institution.



