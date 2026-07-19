"""
Linear Regression: House Price Prediction
==========================================
Predicts a house's sale price from three features:
    - Square footage   (GrLivArea  - above-grade living area)
    - Number of bedrooms (BedroomAbvGr)
    - Number of bathrooms (FullBath + 0.5 * HalfBath)

Expects the Ames Housing data in a "data/" folder next to this script
(data/train.csv), matching the structure of the provided data.zip.

Run:
    python house_price_model.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42


def resolve_data_dir():
    """Locate train.csv either in ./data or in the project root."""
    candidate_dirs = [BASE_DIR / "data", BASE_DIR]
    for candidate in candidate_dirs:
        if (candidate / "train.csv").exists():
            return candidate
    expected_paths = "\n".join(f"  - {path}" for path in candidate_dirs)
    raise FileNotFoundError(
        "Could not find train.csv.\nChecked:\n"
        f"{expected_paths}\n\n"
        "Place train.csv in either a 'data' folder next to this script or in the same folder as the script."
    )


DATA_DIR = resolve_data_dir()

sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_DIR / "train.csv")
print(f"Loaded {len(df)} rows, {df.shape[1]} columns from {DATA_DIR / 'train.csv'}")

# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
# "Square footage" -> GrLivArea   (above-grade living area, sq ft)
# "Bedrooms"        -> BedroomAbvGr
# "Bathrooms"       -> FullBath + 0.5 * HalfBath  (standard real-estate convention,
#                       e.g. a "2.5 bath" house)
df["TotalBath"] = df["FullBath"] + 0.5 * df["HalfBath"]

FEATURES = ["GrLivArea", "BedroomAbvGr", "TotalBath"]
TARGET = "SalePrice"

print("\nMissing values in the columns we use:")
print(df[FEATURES + [TARGET]].isnull().sum().to_string())

# ---------------------------------------------------------------------------
# 3. Remove known outliers
# ---------------------------------------------------------------------------
# The dataset's author documents a few very large homes (>4000 sq ft) that
# sold far under market value (likely non-arms-length sales). Left in, they
# pull the regression line badly off course, so we drop them.
before = len(df)
df = df[df["GrLivArea"] <= 4000].reset_index(drop=True)
print(f"\nRemoved {before - len(df)} outlier rows (GrLivArea > 4000 sq ft)")

# ---------------------------------------------------------------------------
# 4. Quick EDA
# ---------------------------------------------------------------------------
print("\nFeature summary:")
print(df[FEATURES + [TARGET]].describe().round(1).to_string())

print("\nCorrelation with SalePrice:")
print(df[FEATURES + [TARGET]].corr()[TARGET].round(3).to_string())

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, feat in zip(axes, FEATURES):
    ax.scatter(df[feat], df[TARGET], alpha=0.35, s=15, color="#4C72B0")
    ax.set_xlabel(feat)
    ax.set_ylabel("SalePrice ($)")
    ax.set_title(f"{feat} vs SalePrice")
fig.suptitle("Feature Relationships with Sale Price", fontsize=13)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "eda_feature_relationships.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. Train / test split
# ---------------------------------------------------------------------------
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

# ---------------------------------------------------------------------------
# 6. Train the model
# ---------------------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 7. Evaluate
# ---------------------------------------------------------------------------
def evaluate(X, y, label):
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    print(f"{label:>6} -> MAE: ${mae:,.0f}   RMSE: ${rmse:,.0f}   R^2: {r2:.3f}")
    return preds

print("\nModel performance:")
_ = evaluate(X_train, y_train, "Train")
test_preds = evaluate(X_test, y_test, "Test")

# ---------------------------------------------------------------------------
# 8. Interpret coefficients
# ---------------------------------------------------------------------------
terms = "  +  ".join(f"{coef:,.2f} * {name}" for coef, name in zip(model.coef_, FEATURES))
print("\nLearned equation:")
print(f"SalePrice = {model.intercept_:,.2f}  +  {terms}")

print("\nInterpretation (holding other features fixed):")
print(f"  - Each extra sq ft of living area:  ${model.coef_[0]:,.0f} change in price")
print(f"  - Each extra bedroom:               ${model.coef_[1]:,.0f} change in price")
print(f"  - Each extra bathroom:              ${model.coef_[2]:,.0f} change in price")

# ---------------------------------------------------------------------------
# 9. Diagnostic plots
# ---------------------------------------------------------------------------
residuals = y_test - test_preds

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(y_test, test_preds, alpha=0.4, s=20, color="#4C72B0")
lims = [min(y_test.min(), test_preds.min()), max(y_test.max(), test_preds.max())]
axes[0].plot(lims, lims, "--", color="red", linewidth=1.5, label="Perfect prediction")
axes[0].set_xlabel("Actual Price ($)")
axes[0].set_ylabel("Predicted Price ($)")
axes[0].set_title("Actual vs Predicted (Test Set)")
axes[0].legend()

axes[1].scatter(test_preds, residuals, alpha=0.4, s=20, color="#DD8452")
axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5)
axes[1].set_xlabel("Predicted Price ($)")
axes[1].set_ylabel("Residual (Actual - Predicted)")
axes[1].set_title("Residual Plot (Test Set)")

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "model_evaluation.png", dpi=150)
plt.close(fig)

print(f"\nPlots saved to {OUTPUT_DIR}/")

# ---------------------------------------------------------------------------
# 10. Predict on a new house
# ---------------------------------------------------------------------------
def predict_price(sqft, bedrooms, bathrooms):
    """Predict the sale price of a single house."""
    x = pd.DataFrame([[sqft, bedrooms, bathrooms]], columns=FEATURES)
    return model.predict(x)[0]

example = predict_price(sqft=1800, bedrooms=3, bathrooms=2)
print(f"\nExample prediction -> 1800 sq ft, 3 bed, 2 bath: ${example:,.0f}")
