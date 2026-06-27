# Telecom Customer Churn Prediction System

An end-to-end machine learning system that predicts customer churn for a telecom company, segments customers by behavior, estimates customer lifetime value, and explains model predictions — built and benchmarked across multiple algorithms with and without class-imbalance correction.

## Overview

Customer churn (customers leaving the service) is costly for telecom providers. This project builds a complete pipeline — from raw data to an interpretable, deployable model — to predict which customers are at risk of churning, understand *why*, and quantify their business value.

## How It Works

### 1. Data Cleaning & Preprocessing
- Source: telecom customer dataset (`telecom_churn.csv`) with demographic, account, and service-usage features.
- `TotalCharges` was stored as text and silently contained blank strings for new customers (not detected by `isnull()`); it was coerced to numeric and missing values were filled with `0`, since these represent customers who have not yet been billed.
- The non-predictive `customerID` column was dropped.

### 2. Feature Encoding
- **Binary categorical features** (`gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `Churn`) were label-encoded.
- **Multi-class categorical features** (`MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaymentMethod`) were one-hot encoded.

### 3. Train/Test Split & Class Imbalance
- Data was split 80/20 (stratified on the target) into train/test sets.
- The target class (`Churn`) was imbalanced; **SMOTE** (Synthetic Minority Over-sampling) was applied to the training set only, to avoid leaking synthetic samples into evaluation.

### 4. Model Training & Benchmarking
Three classifiers were trained and evaluated, each with and without SMOTE, to isolate the actual impact of the resampling step:

| Model | Without SMOTE | With SMOTE |
|---|---|---|
| Logistic Regression | ✅ | ✅ |
| Random Forest (200 trees) | ✅ | ✅ |
| **XGBoost** (300 trees, depth 5) | ✅ | ✅ **(best)** |

Each run was evaluated with Accuracy, Precision, Recall, F1-Score, ROC-AUC, a full classification report, and ROC curve plots.

### 5. Best Model — XGBoost + SMOTE

| Metric | Score |
|---|---|
| F1-Score | 0.89 |
| ROC-AUC | 0.95 |

### 6. Model Explainability (SHAP)
- **SHAP** (TreeExplainer) was used on the trained Random Forest model to identify which features most influence churn predictions, both globally (summary plot) and for individual customers (force plot).

### 7. Customer Segmentation (K-Means)
- Customers were clustered into **4 segments** using K-Means on standardized `tenure`, `MonthlyCharges`, and `TotalCharges`.
- The optimal number of clusters was selected using the **Elbow Method**.
- Churn rate was computed per cluster to identify which customer segments are highest-risk.

### 8. Customer Lifetime Value (CLTV) & Risk Tiering
- Each customer's churn probability was predicted using the trained XGBoost model.
- **Expected tenure** was estimated as `tenure / (1 − churn_probability)`.
- **CLTV** was calculated as `MonthlyCharges × Expected Tenure`.
- Customers were split into **Low / Medium / High risk** tiers based on CLTV quantiles, allowing the business to prioritize retention efforts by value-at-risk.

### 9. Deployment
- The final XGBoost model, the fitted scaler, and the feature schema were serialized with `joblib` for reuse.
- An interactive **Streamlit** application was built on top of the trained model to allow non-technical stakeholders to input customer data and view churn risk in real time.

## Tech Stack

- **Python**, **Pandas**, **NumPy**
- **scikit-learn** — preprocessing, Logistic Regression, Random Forest, K-Means, evaluation metrics
- **imbalanced-learn (SMOTE)** — class imbalance handling
- **XGBoost** — gradient-boosted classifier (best-performing model)
- **SHAP** — model explainability
- **Matplotlib** — ROC curves and visualizations
- **Streamlit** — interactive web application
- **Joblib** — model serialization

## Project Structure

```
.
├── notebook.ipynb           # Full pipeline: EDA → preprocessing → modeling → clustering → CLTV → export
├── telecom_churn.csv        # Raw dataset
├── churn_model_xgb.pkl      # Serialized best model (XGBoost + SMOTE)
├── scaler.pkl               # Fitted StandardScaler (for clustering features)
├── feature_names.pkl        # Feature schema used at inference time
└── app.py                   # Streamlit application (loads the serialized model)
```

## Running the Project

1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn xgboost shap matplotlib streamlit joblib
   ```
2. Run the notebook top-to-bottom to reproduce preprocessing, training, evaluation, and model export.
3. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Key Insights

- SMOTE measurably improved recall on the minority (churned) class across all three models, at a manageable precision trade-off.
- XGBoost with SMOTE delivered the strongest overall balance of precision and recall (F1: 0.89, ROC-AUC: 0.95).
- Combining churn probability with CLTV — rather than churn probability alone — gives a more business-relevant prioritization: a high-risk, low-value customer matters less than a medium-risk, high-value one.

## Future Improvements

- Hyperparameter tuning (e.g. grid/Bayesian search) for XGBoost rather than fixed parameters.
- Cross-validation instead of a single train/test split for more robust metric estimates.
- Incorporate SHAP-derived insights directly into the Streamlit app for per-customer explanations.
