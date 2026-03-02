# Diabetes Prediction with Machine Learning
> Leveraging the CDC BRFSS dataset and parallel computing to build scalable, generalizable diabetes prediction models.

---

## Overview

This project investigates the use of machine learning to predict diabetes risk using the **CDC Behavioral Risk Factor Surveillance System (BRFSS)** dataset, sourced from Kaggle. Unlike commonly used datasets such as the PIMA Indians Diabetes Database (768 rows), the BRFSS dataset contains hundreds of thousands of real-world records, enabling more generalizable models and making distributed computing techniques viable.

A key focus of this project is the integration of **Hadoop MapReduce** to parallelize preprocessing and feature engineering stages of the pipeline.

---

## Dataset

- **Source**: [CDC BRFSS Dataset on Kaggle](https://www.kaggle.com/)
- **Size**: ~250,000+ records
- **Features**: Demographic, lifestyle, and health survey responses
- **Target**: Binary diabetes diagnosis (diabetic / non-diabetic)

---

## Methodology

### 1. Data Preprocessing
- Feature selection
- Normalization and encoding
- Class imbalance handling

### 2. Parallel Computing
- Hadoop MapReduce for distributed preprocessing and feature engineering

### 3. Models to Evaluate
| Model | Type |
|---|---|
| Logistic Regression | Baseline |
| K-Nearest Neighbors | Distance-Based |
| Random Forest | Ensemble |
| XGBoost | Ensemble |
| Neural Network | Deep Learning |

### 4. Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC

---


---

## Getting Started

```bash
git clone https://github.com/your-repo/diabetes-prediction.git
cd diabetes-prediction
pip install -r requirements.txt
```

---

## Literature Review Summary

Over 9 papers were reviewed. Key findings:
- Most studies rely on the PIMA Indians Diabetes Database
- No reviewed paper incorporated parallel computing in its pipeline
- This project addresses both limitations by using BRFSS and Hadoop MapReduce

---

## Team

| Member | Role |
|---|---|
| Wyatt | Literature Review |
| Owen | Data Collection & Preprocessing |
| Charan | ML Model Implementation |
| Luke | Parallel Computing Implementation |
| Cameron | Results, Visualizations & Discussion |

---

## License

This project is for academic purposes.
