import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

col_names = (
    ["unit_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
sensor_cols = [f"sensor_{i}" for i in range(1, 22)]

test = pd.read_csv("test_FD001.txt", sep=r"\s+", header=None, names=col_names)

true_rul_at_cutoff = pd.read_csv("RUL_FD001.txt", header=None, names=["RUL_at_cutoff"])
true_rul_at_cutoff["unit_id"] = true_rul_at_cutoff.index + 1  # row order = engine order, 1-indexed

model = joblib.load("rul_model.joblib")

# ---------------------------------------------------------------------------
# PART A: Same feature engineering as training — must match exactly
# ---------------------------------------------------------------------------
constant_sensors = ["sensor_1", "sensor_5", "sensor_6", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]
active_sensors = [c for c in sensor_cols if c not in constant_sensors]

test = test.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
for col in active_sensors:
    g = test.groupby("unit_id")[col]
    test[f"{col}_rollmean5"] = g.transform(lambda x: x.rolling(5, min_periods=1).mean())
    test[f"{col}_rollstd5"] = g.transform(lambda x: x.rolling(5, min_periods=1).std().fillna(0))
    test[f"{col}_lag1"] = g.shift(1)
    test[f"{col}_lag5"] = g.shift(5)

lag_cols = [c for c in test.columns if "_lag" in c]
test[lag_cols] = test.groupby("unit_id")[lag_cols].bfill()

# ---------------------------------------------------------------------------
# PART B: Reconstruct the TRUE RUL at every cycle, not just the final one.
# NASA only tells us the RUL at the cutoff point — but RUL drops by exactly 1
# every cycle, so we can work backward to get the true value at every earlier
# cycle too. That's what makes the "ground truth vs predicted trend" plot possible.
# ---------------------------------------------------------------------------
last_cycle = test.groupby("unit_id")["cycle"].max().rename("last_cycle")
test = test.merge(last_cycle, on="unit_id")
test = test.merge(true_rul_at_cutoff, on="unit_id")
test["true_RUL"] = test["RUL_at_cutoff"] + (test["last_cycle"] - test["cycle"])
test["true_RUL_clipped"] = test["true_RUL"].clip(upper=125)

# ---------------------------------------------------------------------------
# PART C: Predict RUL at every cycle for every test engine
# ---------------------------------------------------------------------------
X_test = test[model.feature_names_in_]
test["predicted_RUL"] = model.predict(X_test)

# ---------------------------------------------------------------------------
# PART D: Official evaluation — score at the cutoff cycle only (the standard
# way this benchmark is evaluated, since that's the only point NASA verified)
# ---------------------------------------------------------------------------
final_rows = test.loc[test.groupby("unit_id")["cycle"].idxmax()]
rmse = np.sqrt(mean_squared_error(final_rows["true_RUL_clipped"], final_rows["predicted_RUL"]))
mae = mean_absolute_error(final_rows["true_RUL_clipped"], final_rows["predicted_RUL"])

print("=== Official test-set evaluation (100 engines, never seen in training) ===")
print(f"RMSE: {rmse:.2f} cycles")
print(f"MAE:  {mae:.2f} cycles")

# ---------------------------------------------------------------------------
# PART E: Visualization — ground truth vs predicted trend over time
# ---------------------------------------------------------------------------
sample_engines = [3, 24, 47]
fig, axes = plt.subplots(1, len(sample_engines), figsize=(15, 4), sharey=True)

for ax, unit in zip(axes, sample_engines):
    eng = test[test.unit_id == unit].sort_values("cycle")
    ax.plot(eng["cycle"], eng["true_RUL_clipped"], label="Actual RUL", color="#2563eb", linewidth=2)
    ax.plot(eng["cycle"], eng["predicted_RUL"], label="Predicted RUL", color="#f97316", linestyle="--")
    ax.set_title(f"Engine {unit}")
    ax.set_xlabel("Cycle")

axes[0].set_ylabel("Remaining Useful Life (cycles)")
axes[0].legend()
plt.tight_layout()
plt.savefig("rul_prediction_trends.png", dpi=150)
print("\nSaved rul_prediction_trends.png")
