# 🏦 Credit Scoring Model

### Machine Learning-Based Credit Default Prediction

A complete Machine Learning pipeline that predicts whether a customer is likely to **default on their next month's credit payment** using financial, demographic, billing, and payment-history data.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-purple?style=for-the-badge&logo=pandas" alt="Pandas">
  <img src="https://img.shields.io/badge/SMOTE-Class%20Balancing-green?style=for-the-badge" alt="SMOTE">
</p>

---

## 📌 Project Highlights

* 📊 **25,247 customer records**
* 🔧 **35 features** after feature engineering
* ⚖️ SMOTE for class imbalance
* 🤖 3 ML models trained and compared
* 🌲 Random Forest achieved the best overall performance
* 🎯 **80.67% Accuracy**
* 📈 **77.18% ROC-AUC**
* 🔮 Customer-level credit risk prediction

---

## 🧠 Machine Learning Pipeline

```text
Dataset
   ↓
Data Exploration
   ↓
Missing Value Handling
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
SMOTE Balancing
   ↓
Model Training
   ↓
Hyperparameter Tuning
   ↓
Model Evaluation
   ↓
Risk Prediction
```

---

## 🤖 Models & Results

| Model               |   Accuracy |  Precision |     Recall |   F1-Score |    ROC-AUC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression |     76.10% |     41.44% | **61.64%** | **49.56%** |     76.77% |
| Decision Tree       |     74.53% |     36.29% |     44.59% |     40.02% |     67.73% |
| **Random Forest**   | **80.67%** | **49.23%** |     46.57% |     47.86% | **77.18%** |

### 🏆 Best Model — Random Forest

```text
Accuracy   → 80.67%
Precision  → 49.23%
Recall     → 46.57%
F1-Score   → 47.86%
ROC-AUC    → 77.18%
```

---

## 📊 Model Performance

### ROC Curves

<p align="center">
  <img src="roc_curves.png" alt="ROC Curves" width="750">
</p>

### Model Comparison

<p align="center">
  <img src="model_comparison.png" alt="Model Comparison" width="750">
</p>

### Confusion Matrices

<p align="center">
  <img src="confusion_matrices.png" alt="Confusion Matrices" width="750">
</p>

### Feature Importance

<p align="center">
  <img src="feature_importance.png" alt="Feature Importance" width="750">
</p>

---

## 🔧 Feature Engineering

The project creates additional features to capture customer financial behavior:

* `avg_bill_amt`
* `avg_pay_amt`
* `avg_pay_ratio`
* `pay_status_sum`
* `max_delay`
* `num_delays`
* `debt_ratio`
* `pay_std`
* `bill_std`

---

## 🔮 Sample Prediction

The Random Forest model predicts the probability of default and assigns a risk level.

| Prediction  | Default Probability | Risk Level     |
| ----------- | ------------------: | -------------- |
| Non-Default |              29.40% | 🟢 Low Risk    |
| Non-Default |              17.61% | 🟢 Low Risk    |
| Non-Default |              25.41% | 🟢 Low Risk    |
| Non-Default |               9.47% | 🟢 Low Risk    |
| Non-Default |              47.34% | 🟡 Medium Risk |

---

## 📁 Project Structure

```text
CodeAlpha_CreditScoringModel/
│
├── 📄 credit_scoring.py
├── 📄 train_dataset_final1.csv
├── 📄 requirements.txt
├── 📄 README.md
│
├── 📊 roc_curves.png
├── 📊 model_comparison.png
├── 📊 confusion_matrices.png
└── 📊 feature_importance.png
```

---

## 🛠️ Technologies

```text
Python
Pandas
NumPy
Scikit-learn
Imbalanced-learn
SMOTE
Matplotlib
Seaborn
```

---

## 🚀 How to Run

### Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
cd CodeAlpha_CreditScoringModel
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Model

```bash
python credit_scoring.py
```

---

## 🎯 Key Learning Outcomes

**Data Preprocessing → Feature Engineering → SMOTE → Classification → Hyperparameter Tuning → Model Evaluation → Credit Risk Prediction**

---

## 👨‍💻 Author

### Ikram Ul Haq

**Machine Learning Developer | Python | AI Agents | LLMs**

---

⭐ **If you find this project useful, consider giving it a star!**
