import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
    USE_XGBOOST = True
except Exception as e:
    print(f"Warning: XGBoost could not be loaded: {type(e).__name__}")
    print("Falling back to RandomForestRegressor...")
    print("Note: To use XGBoost, install OpenMP: brew install libomp")
    from sklearn.ensemble import RandomForestRegressor
    USE_XGBOOST = False

df = pd.read_csv("Health_Risk_Dataset.csv")

risk_map = {"Low": 0, "Medium": 1, "High": 2}
df["Risk_Label"] = df["Risk_Level"].map(risk_map)

missing_risk = df["Risk_Label"].isna().sum()
if missing_risk > 0:
    print(f"Warning: {missing_risk} rows have unmapped Risk_Level values")
    print(f"Unique Risk_Level values: {df['Risk_Level'].unique()}")
    df = df.dropna(subset=["Risk_Label"])
    print(f"Removed {missing_risk} rows with missing risk labels")
    print(f"Remaining rows: {len(df)}")

X = df.drop(columns=["Patient_ID", "Risk_Level", "Risk_Label"])
y = df["Risk_Label"]

categorical_cols = ["Consciousness", "On_Oxygen", "O2_Scale"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

if USE_XGBOOST:
    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42
    )
else:
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )

pipeline = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model)
])
