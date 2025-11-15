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

X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.2, random_state=42
)


pipeline.fit(X_train, y_train)


preds = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)


print("\n" + "="*50)
print("MODEL EVALUATION METRICS")
print("="*50)
print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"MAE (Mean Absolute Error): {mae:.4f}")
print(f"R² Score: {r2:.4f}")


try:
   n_splits = min(5, len(X_train) // 2) if len(X_train) > 10 else 3
   if n_splits >= 2:
       cv_scores = cross_val_score(pipeline, X_train, y_train, cv=n_splits, scoring='neg_mean_squared_error')
       cv_rmse = np.sqrt(-cv_scores)
       print(f"\nCross-Validation RMSE ({n_splits}-fold): {cv_rmse.mean():.4f} (+/- {cv_rmse.std() * 2:.4f})")
   else:
       print("\nDataset too small for cross-validation")
except Exception as e:
   print(f"\nCross-validation skipped: {e}")
fitted_preprocessor = pipeline.named_steps['preprocess']
try:
   feature_names = fitted_preprocessor.get_feature_names_out()
except AttributeError:
   numeric_feature_names = numeric_cols
   cat_encoder = fitted_preprocessor.named_transformers_['cat']
   categorical_feature_names = list(cat_encoder.get_feature_names_out(categorical_cols))
   feature_names = numeric_feature_names + categorical_feature_names


importances = pipeline.named_steps['model'].feature_importances_
feature_importance_df = pd.DataFrame({
   'Feature': feature_names,
   'Importance': importances
}).sort_values('Importance', ascending=False)


print("\n" + "="*50)
print("TOP 10 MOST IMPORTANT FEATURES")
print("="*50)
print(feature_importance_df.head(10).to_string(index=False))


print("\n" + "="*50)
print("GENERATING RISK SCORES")
print("="*50)


raw_predictions = pipeline.predict(X)


scaler = MinMaxScaler(feature_range=(0, 100))
df["Risk_Score"] = scaler.fit_transform(raw_predictions.reshape(-1, 1)).flatten()


df["Risk_Score"] = df["Risk_Score"].round(2)


print(f"\nRisk Score Statistics:")
print(f"  Min: {df['Risk_Score'].min():.2f}")
print(f"  Max: {df['Risk_Score'].max():.2f}")
print(f"  Mean: {df['Risk_Score'].mean():.2f}")
print(f"  Median: {df['Risk_Score'].median():.2f}")
print(f"  Std Dev: {df['Risk_Score'].std():.2f}")


print(f"\nRisk Score by Original Risk Level:")
print(df.groupby('Risk_Level')['Risk_Score'].agg(['mean', 'min', 'max', 'std']).round(2))


output_file = "Health_Risk_Dataset_With_Scores.csv"
df.to_csv(output_file, index=False)


print(f"\n{'='*50}")
print(f"Results saved to: {output_file}")
print(f"{'='*50}")
print(f"\nThe dataset now includes a continuous Risk_Score (0-100)")
print(f"that provides nuanced risk assessment for resource allocation.")