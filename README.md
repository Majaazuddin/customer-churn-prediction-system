# 🚀 Customer Churn Prediction System (End-to-End MLOps Project)

An end-to-end machine learning system to predict customer churn, deployed using FastAPI and Docker, with monitoring, drift detection, and automated retraining.

---

## 🧠 Business Problem

Customer churn occurs when customers stop using a company’s service.  
For telecom and subscription-based companies, churn leads to:

- Revenue loss
- Higher customer acquisition costs
- Reduced customer lifetime value

### 🎯 Objective
Build a production-ready machine learning system that:
- Predicts customer churn
- Returns churn probability
- Works via a real-time API
- Monitors data drift
- Supports automated retraining

---

## 🏗️ System Architecture

# 🚀 Customer Churn Prediction System (End-to-End MLOps Project)

An end-to-end machine learning system to predict customer churn, deployed using FastAPI and Docker, with monitoring, drift detection, and automated retraining.

---

## 🧠 Business Problem

Customer churn occurs when customers stop using a company’s service.  
For telecom and subscription-based companies, churn leads to:

- Revenue loss
- Higher customer acquisition costs
- Reduced customer lifetime value

### 🎯 Objective
Build a production-ready machine learning system that:
- Predicts customer churn
- Returns churn probability
- Works via a real-time API
- Monitors data drift
- Supports automated retraining

---

## 🏗️ System Architecture

Customer Data
↓
Data Cleaning & Feature Engineering
↓
Machine Learning Models
↓
Best Model (XGBoost)
↓
FastAPI (/predict endpoint)
↓
Docker Container
↓
Monitoring & Drift Detection
↓
Auto-Retraining Pipeline

---

## 🧹 Data Cleaning

Steps performed on the raw dataset:

- Loaded Telco Customer Churn dataset (~7000 records)
- Handled missing values:
  - Numerical features → median
  - Categorical features → most frequent value
- Detected and handled outliers using IQR method
- Converted categorical values to numerical format
- Saved clean dataset for modeling

---

## 🛠️ Feature Engineering

- Binary encoding for Yes/No columns
- One-hot encoding for categorical features
- Ensured feature alignment between training and inference
- Prepared final encoded dataset for ML models

---

## 🤖 Models Used

The following models were trained and compared:

1. **Logistic Regression**
   - Baseline model
   - Easy to interpret

2. **Random Forest**
   - Captures non-linear relationships
   - Improved performance over baseline

3. **XGBoost (Final Model)**
   - Best overall performance
   - High ROC-AUC and Recall
   - Robust to noise and complex patterns

---

## 📈 Model Evaluation

Evaluation metrics used:
- Confusion Matrix
- Precision
- Recall
- ROC-AUC

| Model | ROC-AUC | Precision | Recall |
|------|--------|-----------|--------|
| Logistic Regression | Baseline | Lower | Lower |
| Random Forest | Improved | Better | Better |
| XGBoost | Best | Highest | Highest |

**ROC-AUC** was preferred over accuracy due to class imbalance in churn data.

---

## 🔍 Explainability (SHAP)

- Used SHAP to explain XGBoost predictions
- Identified global feature importance
- Explained individual customer churn predictions
- Improved transparency and business trust

---

## 🌐 Deployment

- Deployed model using **FastAPI**
- Created `/predict` endpoint returning:
  - Churn probability
  - Churn decision (YES / NO)
- Tested API using Swagger UI
- Containerized application using **Docker**

---

## 📊 Monitoring & Drift Detection

After deployment:
- Logged input feature statistics
- Tracked prediction distributions
- Compared training data vs live inference data
- Detected data drift using statistical comparisons

This ensures model reliability over time.

---

## 🔄 Auto-Retraining (Bonus)

- Drift detection triggers retraining logic
- Retraining uses only clean, labeled historical data
- New model is saved and can be redeployed
- Prevents performance degradation in production

---

## 🚀 Future Improvements

- Deploy on cloud (AWS / Azure / GCP)
- Add alerting for drift detection
- Improve retraining using delayed labels
- Build monitoring dashboard
- Add CI/CD pipeline for model updates

---

## 🧑‍💻 Key Learnings

- End-to-end ML systems go beyond model training
- Deployment, monitoring, and retraining are critical
- Data drift can silently degrade performance
- Explainability builds trust in ML systems
- Production ML requires careful engineering decisions

---

## ✅ Project Status

✔ End-to-end ML pipeline completed  
✔ API deployment and Dockerization  
✔ Monitoring, drift detection, and auto-retraining implemented  

---

## 📌 How to Run the Project

```bash
uvicorn api.main:app --reload

http://127.0.0.1:8000/docs

---

## ✅ WHAT TO DO NEXT (IMPORTANT)

1️⃣ **Paste this into `README.md`**  
2️⃣ Save the file  
3️⃣ Run:

```powershell
git add README.md
git commit -m "Update final project README"
git push -u origin main
