import pandas as pd
import numpy as np

np.random.seed(42)

col_names = (
    ["unit_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
setting_cols = [f"setting_{i}" for i in range(1, 4)]

train = pd.read_csv("train_FD001.txt", sep=r"\s+", header=None, names=col_names)

# ---------------------------------------------------------------------------
# PART A: Create the label we want to predict (RUL = Remaining Useful Life)
# ---------------------------------------------------------------------------
# For each engine, RUL at a given cycle = (the cycle it finally failed at) - (this cycle)
max_cycle = train.groupby("unit_id")["cycle"].max().rename("max_cycle")
train = train.merge(max_cycle, on="unit_id")
train["RUL"] = train["max_cycle"] - train["cycle"]
train = train.drop(columns="max_cycle")

print("=== PART A: RUL label added ===")
print(train[train.unit_id == 1][["unit_id", "cycle", "RUL"]].tail(3))
print()


# ---------------------------------------------------------------------------
# PART B: Inject realistic IoT faults into a COPY of the clean data
# ---------------------------------------------------------------------------
def inject_iot_faults(df, dropout_frac=0.02, missing_row_frac=0.015, spike_frac=0.01, seed=42):
    rng = np.random.default_rng(seed)
    noisy = df.copy()

    # 1) Intermittent dropouts: individual sensor readings randomly go missing
    dropout_mask = rng.random(noisy[sensor_cols].shape) < dropout_frac
    noisy[sensor_cols] = noisy[sensor_cols].mask(dropout_mask)

    # 2) Missing timestamps: whole rows occasionally vanish (lost transmission)
    drop_idx = noisy.sample(frac=missing_row_frac, random_state=seed).index
    noisy = noisy.drop(index=drop_idx)

    # 3) Out-of-bounds spikes: a sensor briefly reports an unrealistic value
    spike_idx = noisy.sample(frac=spike_frac, random_state=seed + 1).index
    spike_cols = rng.choice(sensor_cols, size=len(spike_idx))
    for idx, col in zip(spike_idx, spike_cols):
        noisy.loc[idx, col] *= rng.uniform(3, 8)

    return noisy.sort_values(["unit_id", "cycle"]).reset_index(drop=True)


noisy_train = inject_iot_faults(train)

print("=== PART B: Faults injected ===")
print(f"Rows before: {len(train)}  |  rows after (some dropped): {len(noisy_train)}")
print(f"Missing values now present: {noisy_train[sensor_cols].isna().sum().sum()}")
print()


# ---------------------------------------------------------------------------
# PART C: Clean it back up
# ---------------------------------------------------------------------------
def clean_iot_data(df):
    cleaned_parts = []
    for unit_id, group in df.groupby("unit_id"):
        group = group.set_index("cycle")

        # Fill in any missing cycle numbers caused by dropped rows
        full_range = pd.RangeIndex(group.index.min(), group.index.max() + 1)
        group = group.reindex(full_range)
        group["unit_id"] = unit_id

        # Detect spikes: a reading far from its local rolling median is suspicious
        for col in sensor_cols:
            rolling_med = group[col].rolling(window=5, center=True, min_periods=1).median()
            spike_mask = (group[col] - rolling_med).abs() > 3 * group[col].std()
            group.loc[spike_mask, col] = np.nan

        # Now fill every gap (dropouts + missing rows + removed spikes) by interpolation
        group[sensor_cols + setting_cols] = group[sensor_cols + setting_cols].interpolate(
            method="linear", limit_direction="both"
        )
        cleaned_parts.append(group.reset_index().rename(columns={"index": "cycle"}))

    cleaned = pd.concat(cleaned_parts, ignore_index=True)

    # RUL only depends on (unit_id, cycle), never on sensor values, so we can
    # always look up the correct label rather than trying to "clean" it.
    rul_lookup = train[["unit_id", "cycle", "RUL"]]
    cleaned = cleaned.drop(columns="RUL", errors="ignore").merge(rul_lookup, on=["unit_id", "cycle"], how="left")
    return cleaned


cleaned_train = clean_iot_data(noisy_train)

print("=== PART C: Cleaned ===")
print(f"Remaining missing values: {cleaned_train[sensor_cols].isna().sum().sum()}")
print()

# ---------------------------------------------------------------------------
# PART D: How good was the cleaning? Compare against the original clean data.
# ---------------------------------------------------------------------------
comparison = cleaned_train.merge(train, on=["unit_id", "cycle"], suffixes=("_cleaned", "_true"))
rmses = [
    np.sqrt(((comparison[f"{c}_cleaned"] - comparison[f"{c}_true"]) ** 2).mean())
    for c in sensor_cols
]
print("=== PART D: Cleaning quality ===")
print(f"Average sensor reconstruction RMSE after cleaning: {np.mean(rmses):.4f}")

cleaned_train.to_csv("cleaned_train_FD001.csv", index=False)
print("\nSaved cleaned_train_FD001.csv")
