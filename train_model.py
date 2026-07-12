"""
Trains one XGBoost model on the combined dataset from build_dataset.py.
Same script whether you give it 1 country or 10 - it doesn't know or care.

Usage:
    python train_model.py
(edit COUNTRY_FILES below to add/remove countries)
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

from build_dataset import build_dataset, FEATURES, TARGET

# ------------------------------------------------------------------
# EDIT THIS to control which countries go into the model
# ------------------------------------------------------------------
COUNTRY_FILES = {
    "ES": "data/ES_solar_generaition_1_.csv",
    "DE": "data/DE_opsd_extract.csv",
    "GR": "data/GR_IT_opsd_extract.csv",
    "IT": "data/GR_IT_opsd_extract.csv",
}
WEATHER_DIR = "data/"


def chronological_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    """
    Splits EACH country chronologically on its own, then recombines.
    (Splitting the whole multi-country frame chronologically as one block
    would be wrong - it would put, say, all of Spain in train and all of
    Germany in test just because of row order. Each country needs its own
    70/15/15 split along its own timeline.)
    """
    train_parts, val_parts, test_parts = [], [], []
    for country, group in df.groupby("country"):
        group = group.sort_values("timestamp")
        n = len(group)
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))
        train_parts.append(group.iloc[:train_end])
        val_parts.append(group.iloc[train_end:val_end])
        test_parts.append(group.iloc[val_end:])
    return (pd.concat(train_parts).reset_index(drop=True),
            pd.concat(val_parts).reset_index(drop=True),
            pd.concat(test_parts).reset_index(drop=True))


def train_xgb(train, val):
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=30,
        eval_metric="rmse",
    )
    model.fit(train[FEATURES], train[TARGET],
              eval_set=[(val[FEATURES], val[TARGET])], verbose=False)
    return model


def evaluate(model, test: pd.DataFrame, label: str = "Test set"):
    pred = model.predict(test[FEATURES])
    pred = np.clip(pred, 0, 1)

    rmse = np.sqrt(mean_squared_error(test[TARGET], pred))
    mae = mean_absolute_error(test[TARGET], pred)
    daytime_mask = test[TARGET].values > 0.01
    mape = np.mean(np.abs(
        (test[TARGET].values[daytime_mask] - pred[daytime_mask]) / test[TARGET].values[daytime_mask]
    )) * 100

    print(f"\n--- {label} ---")
    print(f"  Rows: {len(test)}")
    print(f"  RMSE (capacity factor): {rmse:.4f}")
    print(f"  MAE  (capacity factor): {mae:.4f}")
    print(f"  MAPE (daytime hours only): {mape:.1f}%")
    return {"rmse": rmse, "mae": mae, "mape": mape, "pred": pred}


if __name__ == "__main__":
    df = build_dataset(COUNTRY_FILES, weather_dir=WEATHER_DIR)
    df = df.dropna(subset=FEATURES + [TARGET])

    train, val, test = chronological_split(df)
    print(f"\nTrain: {len(train)} | Validation: {len(val)} | Test: {len(test)}")

    model = train_xgb(train, val)
    print(f"Best iteration: {model.best_iteration}")

    results = evaluate(model, test)

    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nFeature importances:")
    print(importances.to_string())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(importances.index, importances.values)
    ax.set_title("Feature importance")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("model_evaluation.png", dpi=150)
    print("\nSaved: model_evaluation.png")

    model.save_model("xgb_solar_model.json")
    print("Saved: xgb_solar_model.json")
