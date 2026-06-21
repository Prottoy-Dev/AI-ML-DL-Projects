import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

sensor_cols = [f"sensor_{i}" for i in range(1, 22)]

df = pd.read_csv("cleaned_train_FD001.csv")

# ---------------------------------------------------------------------------
# PART A: Drop sensors that never change — they carry no useful signal
# ---------------------------------------------------------------------------
variances = df[sensor_cols].var()
constant_sensors = variances[variances < 1e-6].index.tolist()
active_sensors = [c for c in sensor_cols if c not in constant_sensors]
print("Dropping constant sensors (no information):", constant_sensors)
print(f"Keeping {len(active_sensors)} active sensors\n")

# ---------------------------------------------------------------------------
# PART B: Feature engineering — rolling stats + lag features, per engine
# ---------------------------------------------------------------------------
df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)

for col in active_sensors:
    g = df.groupby("unit_id")[col]
    df[f"{col}_rollmean5"] = g.transform(lambda x: x.rolling(5, min_periods=1).mean())
    df[f"{col}_rollstd5"] = g.transform(lambda x: x.rolling(5, min_periods=1).std().fillna(0))
    df[f"{col}_lag1"] = g.shift(1)
    df[f"{col}_lag5"] = g.shift(5)

# Early cycles in each engine don't have a "5 cycles ago" yet — backfill within engine
lag_cols = [c for c in df.columns if "_lag" in c]
df[lag_cols] = df.groupby("unit_id")[lag_cols].bfill()

# ---------------------------------------------------------------------------
# PART C: Cap RUL at 125 — beyond that, the exact number is just noise
# ---------------------------------------------------------------------------
df["RUL_clipped"] = df["RUL"].clip(upper=125)

# ---------------------------------------------------------------------------
# PART D: Split by ENGINE, not by row, so we never leak one engine's future
# into its own "validation" — this is the overfitting guard the brief asks about
# ---------------------------------------------------------------------------
all_units = df["unit_id"].unique()
train_units, val_units = train_test_split(all_units, test_size=0.2, random_state=42)

feature_cols = [c for c in df.columns if c not in ["unit_id", "cycle", "RUL", "RUL_clipped"]]

train_df = df[df.unit_id.isin(train_units)]
val_df = df[df.unit_id.isin(val_units)]

X_train, y_train = train_df[feature_cols], train_df["RUL_clipped"]
X_val, y_val = val_df[feature_cols], val_df["RUL_clipped"]

# ---------------------------------------------------------------------------
# PART E: Train — depth/leaf limits are a second overfitting guard
# ---------------------------------------------------------------------------
model = RandomForestRegressor(
    n_estimators=150, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

preds = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, preds))
mae = mean_absolute_error(y_val, preds)

print("=== Validation performance (engines the model never trained on) ===")
print(f"RMSE: {rmse:.2f} cycles")
print(f"MAE:  {mae:.2f} cycles")

baseline_pred = np.full_like(y_val, y_train.mean(), dtype=float)
baseline_rmse = np.sqrt(mean_squared_error(y_val, baseline_pred))
print(f"\nNaive baseline RMSE (always predicting the average RUL): {baseline_rmse:.2f} cycles")

importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 10 most important features:")
print(importances.head(10))

joblib.dump(model, "rul_model.joblib")
df.to_csv("features_train_FD001.csv", index=False)
print("\nSaved rul_model.joblib and features_train_FD001.csv")
