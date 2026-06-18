"""
End-to-end solar production forecasting model for Spain (Seville region).

Pipeline:
    1. Load the OPSD hourly file (actual solar generation, Spain)
    2. Load the Open-Meteo hourly weather file (Seville)
    3. Merge on timestamp
    4. Feature engineering: cyclical time encoding, days_since_start,
       capacity factor (normalizes for Spain's growing installed capacity)
    5. Chronological train/validation/test split
    6. Train XGBoost with early stopping
    7. Evaluate (RMSE, MAE, daytime-only MAPE)
    8. Feature importance + actual-vs-predicted plot

Requirements:
    pip install pandas numpy xgboost scikit-learn matplotlib --break-system-packages
    (xgboost >= 2.0 recommended, for the constructor-based early_stopping_rounds API)
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 0. EDIT THESE if your files have different names / locations
# ------------------------------------------------------------------
OPSD_PATH = "ES_solar_generaition_1_.csv"
WEATHER_PATH = "open-meteo-37_36N5_98W21m__1_.csv"

# Approximate cumulative installed *PV* capacity in Spain, end of year, in MW.
# Source: Red Electrica de Espana figures reported via pv-magazine, and
# Wikipedia "Solar power in Spain". Does not include Spain's ~2300 MW of CSP
# (concentrated solar power), which stayed roughly constant across this period.
CAPACITY_LOOKUP_MW = {
    2015: 4270,
    2016: 4330,
    2017: 4460,
    2018: 4730,
    2019: 8700,
    2020: 9500,
}

# ------------------------------------------------------------------
# 1. Load OPSD solar generation (Spain)
# ------------------------------------------------------------------
print("Loading OPSD data...")
opsd = pd.read_csv(
    OPSD_PATH,
    usecols=["utc_timestamp", "ES_solar_generation_actual"],
    parse_dates=["utc_timestamp"],
)
opsd = opsd.rename(columns={"ES_solar_generation_actual": "solar_mw"})
opsd = opsd.dropna(subset=["solar_mw"])
opsd["utc_timestamp"] = opsd["utc_timestamp"].dt.tz_localize(None)  # drop tz info to match weather file
print(f"  {len(opsd)} rows, {opsd['utc_timestamp'].min()} -> {opsd['utc_timestamp'].max()}")

# ------------------------------------------------------------------
# 2. Load weather data (Open-Meteo CSV export)
#    Note: Open-Meteo CSVs start with a coordinates/metadata block (2 lines)
#    and a blank line before the real header row, so we skip 3 lines. The
#    column names also include units (e.g. "temperature_2m (°C)"), which
#    we clean up to match the bare names used elsewhere in this script.
# ------------------------------------------------------------------
print("Loading weather data...")
weather = pd.read_csv(WEATHER_PATH, skiprows=3)
weather = weather.rename(columns={
    "temperature_2m (°C)": "temperature_2m",
    "cloud_cover (%)": "cloud_cover",
    "shortwave_radiation (W/m²)": "shortwave_radiation",
    "wind_speed_10m (km/h)": "wind_speed_10m",
})
weather["time"] = pd.to_datetime(weather["time"])
print(f"  {len(weather)} rows, {weather['time'].min()} -> {weather['time'].max()}")

# ------------------------------------------------------------------
# 3. Merge the two sources on timestamp
# ------------------------------------------------------------------
df = pd.merge(opsd, weather, left_on="utc_timestamp", right_on="time", how="inner")
df = df.drop(columns=["time"]).sort_values("utc_timestamp").reset_index(drop=True)
print(f"Merged: {len(df)} rows")

# ------------------------------------------------------------------
# 4. Feature engineering
# ------------------------------------------------------------------
df["hour"] = df["utc_timestamp"].dt.hour
df["doy"] = df["utc_timestamp"].dt.dayofyear

# Cyclical encoding so the model understands hour 23 is close to hour 0,
# and day 365 is close to day 1
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365)
df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365)

# Lets the model learn long-term trend (e.g. growing capacity over time)
df["days_since_start"] = (df["utc_timestamp"] - df["utc_timestamp"].min()).dt.days

# Capacity factor: normalizes raw MW by how much capacity existed that year,
# so the target reflects weather conditions rather than fleet growth
df["year"] = df["utc_timestamp"].dt.year
df["capacity_mw"] = df["year"].map(CAPACITY_LOOKUP_MW)
df["capacity_factor"] = df["solar_mw"] / df["capacity_mw"]

FEATURES = [
    "temperature_2m", "cloud_cover", "shortwave_radiation", "wind_speed_10m",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "days_since_start",
]
TARGET = "capacity_factor"

df = df.dropna(subset=FEATURES + [TARGET])
print(f"After dropping rows with missing features/target: {len(df)} rows")

# ------------------------------------------------------------------
# 5. Chronological split: 70% train / 15% validation / 15% test
#    (never shuffle randomly for time series - that leaks the future
#    into training and gives an unrealistically good score)
# ------------------------------------------------------------------
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = df.iloc[:train_end]
val = df.iloc[train_end:val_end]
test = df.iloc[val_end:]

print(f"Train: {len(train)} | Validation: {len(val)} | Test: {len(test)}")

X_train, y_train = train[FEATURES], train[TARGET]
X_val, y_val = val[FEATURES], val[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

# ------------------------------------------------------------------
# 6. Train XGBoost (early stopping picks the right n_estimators automatically)
# ------------------------------------------------------------------
model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=30,
    eval_metric="rmse",
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print(f"\nBest iteration (rounds actually used): {model.best_iteration}")

# ------------------------------------------------------------------
# 7. Evaluate on the held-out test set
# ------------------------------------------------------------------
pred = model.predict(X_test)
pred = np.clip(pred, 0, 1)  # capacity factor is physically bounded to [0, 1]

rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)

# MAPE only over daytime hours - otherwise division by ~0 at night blows up the metric
daytime_mask = y_test.values > 0.01
mape = np.mean(np.abs((y_test.values[daytime_mask] - pred[daytime_mask]) / y_test.values[daytime_mask])) * 100

print(f"\n--- Test set results ---")
print(f"RMSE (capacity factor): {rmse:.4f}")
print(f"MAE  (capacity factor): {mae:.4f}")
print(f"MAPE (daytime hours only): {mape:.1f}%")
print(f"MAE in approx. MW terms: {mae * test['capacity_mw'].mean():.1f} MW")

# ------------------------------------------------------------------
# 8. Feature importance + actual-vs-predicted plot
# ------------------------------------------------------------------
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nFeature importances:")
print(importances.to_string())

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].barh(importances.index, importances.values)
axes[0].set_title("Feature importance")
axes[0].invert_yaxis()

n_hours_to_plot = 24 * 7  # one week
sample = test.iloc[:n_hours_to_plot]
sample_pred = pred[:n_hours_to_plot]
axes[1].plot(sample["utc_timestamp"], sample[TARGET], label="Actual", linewidth=1.5)
axes[1].plot(sample["utc_timestamp"], sample_pred, label="Predicted", linewidth=1.5, alpha=0.8)
axes[1].set_title("Actual vs predicted capacity factor (first week of test set)")
axes[1].set_ylabel("Capacity factor")
axes[1].legend()
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=150)
print("\nSaved plot: model_evaluation.png")

# ------------------------------------------------------------------
# 9. Save the model and the final merged/feature-engineered dataset
# ------------------------------------------------------------------
model.save_model("xgb_solar_model.json")
df.to_csv("merged_dataset_full.csv", index=False)
print("Saved: xgb_solar_model.json, merged_dataset_full.csv")
