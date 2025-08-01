# 🏠 House Price Prediction — Advanced Regression & Model Optimization

This project uses Kaggle’s **House Prices – Advanced Regression Techniques** dataset to predict residential property prices.  
It implements a complete **machine learning pipeline** with preprocessing, feature engineering, model comparison, and hyperparameter tuning using **Optuna**.

---

## 📌 Problem Statement

Accurately predicting house prices is a common challenge in the real estate industry.  
By analyzing historical data on property characteristics (location, size, age, amenities), we aim to **build a robust regression model** that can estimate sale prices with minimal error.

---

## 🎯 Objectives

1. **Preprocess & clean the data** (handle missing values, encode categorical features, scale numerical features).
2. **Engineer new features** to capture more meaningful patterns.
3. **Test and compare multiple regression algorithms** (Linear, Ridge, Lasso, Random Forest, GBM, XGBoost, LightGBM, CatBoost).
4. **Select top features** based on importance.
5. **Tune the best model’s hyperparameters** using **Optuna** for optimal performance.
6. **Explain model predictions** using **SHAP values**.

---

## 📂 Dataset

- **Source:** [Kaggle – House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- **Rows:** 1460 (training set) + 1459 (test set)
- **Features:** 79 explanatory variables
- **Target:** `SalePrice` (continuous)

---

## 🔄 Workflow

1. **Data Loading & Preprocessing**  
   - Handle missing categorical & numerical values.  
   - Encode categorical variables.  
   - Scale numerical features.

2. **Feature Engineering**  
   - Created `TotalSF`, `TotalBathrooms`, `Age`, and `RemodAge`.

3. **Skewness Correction**  
   - Applied log transformations to skewed numerical features.

4. **Feature Selection**  
   - Used Random Forest importance to select top 45 impactful features.

5. **Model Training & Comparison**  
   - Linear Regression  
   - Ridge & Lasso Regression  
   - Random Forest  
   - Gradient Boosting  
   - XGBoost  
   - LightGBM  
   - CatBoost

6. **Hyperparameter Tuning with Optuna**  
   - Optimized best-performing model (XGBoost).

7. **Explainability with SHAP**  
   - Visualized feature contributions.

---

## 📊 Results

| Model           | R² (Train) | RMSE (Train) | MAE (Train) |
|-----------------|------------|--------------|-------------|
| Linear          | 0.8997     | 0.13         | 0.09        |
| Ridge           | 0.8983     | 0.13         | 0.09        |
| Lasso           | 0.8985     | 0.13         | 0.09        |
| Random Forest   | 0.9832     | 0.05         | 0.03        |
| GBM             | 0.9545     | 0.09         | 0.06        |
| XGBoost         | **0.9871** | **0.05**     | **0.03**    |
| LightGBM        | 0.9777     | 0.06         | 0.04        |
| CatBoost        | 0.9764     | 0.06         | 0.05        |

- **Best Model:** XGBoost (Optuna tuned)  
- **Best Skew Threshold:** 1.0  
- **Selected Features:** 45

---

## 📈 Key Insights

- Feature engineering significantly improved model accuracy.
- Skewness correction reduced bias for certain numeric features.
- Tree-based models like XGBoost handled mixed data types effectively.
- Optuna tuning further reduced RMSE to 0.05.

---

## 📦 Technologies Used

- Python
- Pandas, NumPy, Seaborn, Matplotlib
- scikit-learn
- XGBoost, LightGBM, CatBoost
- Category Encoders
- Optuna (Hyperparameter Tuning)
- SHAP (Model Explainability)
- FPDF (PDF report generation)

---

## 🚀 How to Run

```bash
# Clone repo
git clone https://github.com/<your-username>/house-price-prediction.git
cd house-price-prediction

# Install dependencies
pip install -r requirements.txt

# Run script
python main.py
