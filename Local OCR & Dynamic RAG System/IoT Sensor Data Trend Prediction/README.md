# IoT Sensor Data Trend Prediction — NASA Turbofan Engine RUL

## 1. Industrial Context and Target Variable

This project predicts **Remaining Useful Life (RUL)** — the number of operating
cycles left before a machine fails — using NASA's C-MAPSS Turbofan Engine
Degradation Simulation dataset (FD001 subset). This sits squarely in the
**manufacturing machinery / predictive maintenance** domain named in the brief.

The dataset tracks 100 jet engines, each run from a healthy state to failure,
with 21 sensors (temperatures, pressures, fan speeds, etc.) and 3 operational
settings recorded at every cycle. A separate set of 100 test engines is
provided, each truncated at an arbitrary point before failure, with the true
remaining cycle count held out for scoring — this is the standard, widely-used
benchmark protocol for this dataset.

**Target variable:** RUL (cycles remaining), framed as a regression problem.

## 2. Repository Structure

| File | Purpose |
|---|---|
| `CMAPSSData/` | Raw NASA data (`train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`) |
| `step1_explore.py` | Initial data loading and sanity checks |
| `step2_label_and_clean.py` | Builds the RUL label; injects and cleans realistic IoT faults |
| `step3_features_and_model.py` | Feature engineering and model training |
| `step4_test_eval_and_plot.py` | Official held-out test evaluation and trend visualization |
| `rul_prediction_trends.png` | Actual vs. predicted RUL trend for sample engines |
| `cleaned_train_FD001.csv`, `features_train_FD001.csv`, `rul_model.joblib` | Generated intermediate artifacts |

Run the four scripts in order from the same directory; each one produces the
file the next one depends on.

```
pip install pandas numpy scikit-learn matplotlib joblib
python3 step1_explore.py
python3 step2_label_and_clean.py
python3 step3_features_and_model.py
python3 step4_test_eval_and_plot.py
```

## 3. Data Cleaning: Methodology and Justification

The raw C-MAPSS data is simulated and exceptionally clean — it has no missing
values, dropouts, or sensor faults by design. To demonstrate (and validate)
a real IoT data-cleaning pipeline as required by the brief, we deliberately
performed **fault injection testing**: a copy of the training data was
corrupted with three fault types commonly reported in IoT sensor deployments,
then put through a cleaning pipeline:

- **Intermittent dropouts** (~2% of readings set missing) — simulating
  transient sensor read failures.
- **Missing timestamps** (~1.5% of rows removed entirely) — simulating lost
  transmissions.
- **Out-of-bounds spikes** (~1% of readings multiplied 3–8×) — simulating
  hardware glitches.

The cleaning pipeline, applied per engine:
1. **Reindex each engine's cycle sequence** to its full min–max range, which
   surfaces any missing-timestamp gaps as missing values.
2. **Spike detection** via a rolling-median filter (a Hampel-filter-style
   approach): any reading deviating from its local 5-cycle rolling median by
   more than 3× the sensor's standard deviation is flagged and removed.
3. **Linear interpolation** within each engine fills every remaining gap —
   dropouts, missing-timestamp gaps, and removed spikes alike, all in one step.

Because the original clean values are known, cleaning quality could be
**directly measured**: average reconstruction RMSE across all 21 sensors was
**0.20**, i.e. the pipeline recovered the corrupted data almost exactly.

RUL labels themselves are never "cleaned" or interpolated — they're looked up
by `(unit_id, cycle)` from the uncorrupted source, since RUL is a deterministic
function of which cycle this is and how long that engine ultimately ran, not
of any (possibly noisy) sensor reading.

## 4. Feature Engineering

- **Dropped 7 constant sensors** (`sensor_1, 5, 6, 10, 16, 18, 19`) — under
  FD001's single operating condition these never vary, so they carry no
  predictive signal and only add noise/dimensionality.
- **Rolling mean and rolling std (window = 5 cycles)** per sensor, per engine —
  a single reading is noisy and uninformative about degradation; a short
  rolling window smooths sensor noise and exposes the underlying trend.
- **Lag-1 and lag-5 features** per sensor, per engine — expose the recent rate
  of change, which is what actually signals degradation, rather than just
  absolute sensor level.
- Early cycles in each engine (before 5 cycles of history exist) are
  back-filled within that engine only, never across engines.
- **RUL is clipped at 125 cycles.** Sensors look essentially identical whether
  an engine has 200 or 300 cycles left — there's no real degradation signal
  yet, so asking the model to distinguish between large RUL values just adds
  label noise. Capping (a standard convention in this benchmark's literature)
  focuses learning on the part of an engine's life where degradation is
  actually observable.

## 5. Model Architecture and Overfitting Safeguards

**Model:** `RandomForestRegressor` (150 trees, max depth 12, min 5 samples per
leaf). A tree ensemble was chosen over a sequence model (e.g. LSTM) to keep
iteration fast and the result easy to validate within the project timeline,
while still capturing nonlinear interactions between sensors well — sequence
models are noted as a natural next step (see Limitations).

**Overfitting guards:**
1. **Train/validation split by engine ID, not by row.** A random row-level
   split would put different cycles from the *same* engine's lifecycle into
   both training and validation, leaking information about that engine's
   degradation curve and producing an optimistic, misleading score. Splitting
   by engine ensures validation engines are never seen during training.
2. **Constrained tree depth and minimum leaf size**, limiting how finely the
   model can memorize individual training examples.
3. **RUL clipping** reduces label noise in the high-RUL region, where the true
   number is effectively unknowable from sensor data alone.

## 6. Results

| Evaluation | RMSE (cycles) | MAE (cycles) |
|---|---|---|
| Validation (held-out engines from training data) | 16.34 | 11.56 |
| **Official test set** (100 engines never used in training) | **17.66** | **12.42** |
| Naive baseline (always predict mean RUL) | 41.70 | — |

Validation and official test performance are close (16.34 vs. 17.66 RMSE),
indicating the model generalizes to genuinely unseen engines rather than
overfitting to the training set. Both comfortably beat the naive baseline.

The most important feature was the rolling mean of `sensor_4` — independently
flagged in published research on this benchmark as one of the most
degradation-sensitive sensors, which is a useful sanity check that the model
is learning a physically meaningful signal rather than spurious noise.

See `rul_prediction_trends.png` for the actual-vs-predicted RUL trend on three
sample test engines.

## 7. Limitations and Future Work

- Only the FD001 subset (single operating condition) is used; FD002/FD004
  involve six operating conditions and would need condition-normalization
  before this pipeline applies directly.
- The cleaning pipeline's 5-cycle rolling window assumes faults are short;
  longer outages would need a wider window or a different recovery strategy.
- A sequence model (LSTM/GRU/Transformer) could likely improve on the Random
  Forest's RMSE further, at the cost of more tuning time — a natural next step
  beyond this project's scope.
