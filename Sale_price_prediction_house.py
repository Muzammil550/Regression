import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import shap
import optuna

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import category_encoders as ce

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from fpdf import FPDF

# Create folders
os.makedirs("visualizations", exist_ok=True)

# Load data
train_df = pd.read_csv(r"C:\Users\WAJAHAT TRADERS\Downloads\house-prices-advanced-regression-techniques (2)\train.csv")
test_df = pd.read_csv(r"C:\Users\WAJAHAT TRADERS\Downloads\house-prices-advanced-regression-techniques (2)\train.csv")

y_train = train_df["SalePrice"]
X_train = train_df.drop(["Id", "SalePrice"], axis=1)
X_test = test_df.drop("Id", axis=1)

# Combine for preprocessing
combined = pd.concat([X_train, X_test], axis=0)

# ===== Missing Value Handling =====
none_cols = [
    "Alley", "PoolQC", "Fence", "MiscFeature", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType"
]
for col in none_cols:
    combined[col] = combined[col].fillna("None")

combined["GarageYrBlt"] = combined["GarageYrBlt"].fillna(combined["YearBuilt"])
combined["LotFrontage"] = combined.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median())
)

# Fill categorical with mode
cat_cols = combined.select_dtypes(include="object").columns
for col in cat_cols:
    combined[col] = combined[col].fillna(combined[col].mode()[0])

# Fill numerical with median
num_cols = combined.select_dtypes(exclude="object").columns
for col in num_cols:
    combined[col] = combined[col].fillna(combined[col].median())

# ===== Feature Engineering =====
combined["TotalSF"] = combined["TotalBsmtSF"] + combined["1stFlrSF"] + combined["2ndFlrSF"]
combined["TotalBathrooms"] = (combined["FullBath"] + combined["HalfBath"] * 0.5 +
                               combined["BsmtFullBath"] + combined["BsmtHalfBath"] * 0.5)
combined["Age"] = combined["YrSold"] - combined["YearBuilt"]
combined["RemodAge"] = combined["YrSold"] - combined["YearRemodAdd"]

# ===== Automatic Skew Threshold Testing =====
def evaluate_skew(threshold):
    temp = combined.copy()
    skewness = temp[num_cols].apply(lambda x: x.skew()).sort_values(ascending=False)
    skewed_feats = skewness[abs(skewness) > threshold].index
    temp[skewed_feats] = np.log1p(temp[skewed_feats])
    temp_train = temp.iloc[:len(y_train), :]
    temp_target = np.log1p(y_train)
    temp_train = ce.OneHotEncoder(cols=temp_train.select_dtypes(include="object").columns,
                                  use_cat_names=True).fit_transform(temp_train)
    temp_train = StandardScaler().fit_transform(temp_train)
    scores = cross_val_score(LinearRegression(), temp_train, temp_target,
                             scoring="neg_root_mean_squared_error", cv=5)
    return -scores.mean()

thresholds = [0.5, 0.75, 1.0]
threshold_rmse = {t: evaluate_skew(t) for t in thresholds}
best_threshold = min(threshold_rmse, key=threshold_rmse.get)
print(f"Best skew threshold: {best_threshold} with RMSE {threshold_rmse[best_threshold]:.4f}")

# Apply log1p to skewed features
skewness = combined[num_cols].apply(lambda x: x.skew()).sort_values(ascending=False)
skewed_feats = skewness[abs(skewness) > best_threshold].index
combined[skewed_feats] = np.log1p(combined[skewed_feats])

# Transform target
y_train = np.log1p(y_train)

# ===== Encoding =====
categorical_cols = combined.select_dtypes(include="object").columns.tolist()
encoder = ce.OneHotEncoder(cols=categorical_cols, use_cat_names=True)
combined = encoder.fit_transform(combined)

# Split back
X_train = combined.iloc[:len(y_train), :]
X_test = combined.iloc[len(y_train):, :]

# ===== Scaling =====
scaler = StandardScaler()
X_train[X_train.columns] = scaler.fit_transform(X_train)
X_test[X_test.columns] = scaler.transform(X_test)

# ===== Feature Selection =====
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
selected_features = importances[importances >= 0.001].index.tolist()

plt.figure(figsize=(10, 6))
sns.barplot(x=importances.head(15), y=importances.head(15).index, hue=importances.head(15).index,
            palette="viridis", dodge=False, legend=False)
plt.title("Top 15 Features by Random Forest")
plt.savefig("visualizations/top_features.png")
plt.show()

X_train = X_train[selected_features]
X_test = X_test[selected_features]

# ===== Models & Hyperparameters =====
results = {}

def train_model(name, model, params):
    grid = GridSearchCV(model, params, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    pred_train = best.predict(X_train)
    r2 = r2_score(y_train, pred_train)
    rmse = np.sqrt(mean_squared_error(y_train, pred_train))
    mae = mean_absolute_error(y_train, pred_train)
    results[name] = {"model": best, "r2": r2, "rmse": rmse, "mae": mae}
    print(f"{name} | R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

train_model("Linear", LinearRegression(), {})
train_model("Ridge", Ridge(), {"alpha": [1.0, 10.0, 50.0], "solver": ["auto", "cholesky", "lsqr"]})
train_model("Lasso", Lasso(), {"alpha": [0.001, 0.01, 0.1, 1.0], "max_iter": [1000, 5000]})
train_model("Random Forest", RandomForestRegressor(random_state=42),
            {"n_estimators": [200, 500], "max_depth": [None, 10, 20], "min_samples_split": [2, 5]})
train_model("GBM", GradientBoostingRegressor(random_state=42),
            {"n_estimators": [200, 500], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]})
train_model("XGBoost", XGBRegressor(random_state=42),
            {"n_estimators": [200, 500], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]})
train_model("LightGBM", LGBMRegressor(random_state=42),
            {"n_estimators": [200, 500], "learning_rate": [0.05, 0.1], "max_depth": [-1, 10]})
train_model("CatBoost", CatBoostRegressor(verbose=0, random_state=42),
            {"iterations": [200, 500], "learning_rate": [0.05, 0.1], "depth": [4, 6]})

# ===== Final Results =====
print("\n=== Final Model R², RMSE & MAE ===")
for model, metrics in results.items():
    print(f"{model}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")

# ===== Optuna for Best Model =====
best_model_name = max(results.items(), key=lambda x: x[1]['r2'])[0]
best_model_class = type(results[best_model_name]["model"])
print(f"\nOptuna tuning best model: {best_model_name}")

def objective(trial):
    params = {}
    if best_model_name == "Random Forest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10)
        }
    elif best_model_name == "XGBoost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 10)
        }
    model = best_model_class(**params)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error")
    return -scores.mean()

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=300, show_progress_bar=True)
print("Best Optuna params:", study.best_params)

# ===== SHAP Explainability =====
best_model = best_model_class(**study.best_params)
best_model.fit(X_train, y_train)
explainer = shap.Explainer(best_model, X_train)
shap_values = explainer(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar", show=True)

# ===== Kaggle Submission =====
preds = np.expm1(best_model.predict(X_test))
submission = pd.DataFrame({"Id": test_df["Id"], "SalePrice": preds})
submission.to_csv("submission.csv", index=False)
print("✅ Saved submission.csv")

# ===== PDF Report =====
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", 'B', 16)
pdf.cell(0, 10, "House Price Prediction Report", ln=True)

pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 8, f"Best skew threshold: {best_threshold}\n")
pdf.multi_cell(0, 8, f"Selected Features ({len(selected_features)}): {', '.join(selected_features[:20])}...\n")
pdf.image("visualizations/top_features.png", w=180)

for model_name, res in results.items():
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, f"{model_name} | R²: {res['r2']:.4f} | RMSE: {res['rmse']:.2f} | MAE: {res['mae']:.2f}", ln=True)

pdf.output("House_Price_Report.pdf")
print("📄 Report saved as House_Price_Report.pdf")
