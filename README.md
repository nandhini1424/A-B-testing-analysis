# 📊 A/B Test Retention Prediction with Random Forest

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/status-Completed-brightgreen)


> A machine learning pipeline to predict **7-day user retention** from A/B test data.  
> Helps identify which app version performs better in retaining users.

---

## ✨ Features

- 📦 Preprocessing pipeline using `Pipeline` and `ColumnTransformer`:
  - One-hot encoding for categorical features
  - Standard scaling & median imputation for numeric features
- 🌳 Trains a `RandomForestClassifier` to predict `retention_7`
- 📊 Outputs accuracy & classification report
- 📈 Calculates predicted retention rate for each version (`gate_30` vs. `gate_40`)
- 💾 Saves the trained model & preprocessor with `joblib`

---

## 📦 Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- joblib

## Install dependencies:
```bash
pip install pandas numpy scikit-learn joblib
```
## ▶️ How to Run
1. Place your dataset as ab_test_data.csv in the same folder.

2. Run the script:
   ```bash
   python main.py
## 🧠 How It Works
- Loads user data with columns:
  userid, version, sum_gamerounds, retention_1, retention_7

- Drops missing data (dropna)

- ## Preprocesses features:

     - Numeric: sum_gamerounds

     - Categorical: version, retention_1

     - Splits into train & test sets (80/20)

     - Trains a Random Forest model

     - Evaluates accuracy and prints a detailed classification report

     - Predicts retention and compares average retention by version

     - Identifies the better version based on predicted retention
   
## ✅ Example Output
   ```bash
   Accuracy: 0.74
              precision    recall  f1-score   support

           0       0.72      0.95      0.82      2286
           1       0.65      0.18      0.28       929

    accuracy                           0.71      3215
   macro avg       0.69      0.56      0.55      3215
weighted avg       0.70      0.71      0.65      3215

version
gate_30    0.18
gate_40    0.21
Name: predicted_retention, dtype: float64
The better version is: gate_40
```

