import pandas as pd

# The NASA files have no headers, just numbers separated by spaces.
# Each row = one engine, at one point in time (one "flight cycle").
# Columns: engine id, cycle number, 3 operating settings, 21 sensor readings.
col_names = (
    ["unit_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

train = pd.read_csv("train_FD001.txt", sep=r"\s+", header=None, names=col_names)

print("Shape (rows, columns):", train.shape)
print()
print("How many unique engines?", train["unit_id"].nunique())
print()
print("First 5 rows:")
print(train.head())
print()
print("How long did each engine run before failing (in cycles)?")
print(train.groupby("unit_id")["cycle"].max().describe())
