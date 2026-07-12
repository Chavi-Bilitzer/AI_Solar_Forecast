"""
Leave-One-Country-Out (LOCO) validation.

This is the test of whether the model actually learned a GENERAL
relationship between weather + geography -> solar output, versus just
memorizing patterns specific to the countries it trained on.

For each country in the dataset:
    1. Train on every OTHER country
    2. Test on the held-out country (which the model never saw at all -
       not even a few rows of it, unlike the normal train/val/test split)
    3. Compare that score to a normal same-country test score

If the "held-out country" score is close to the "same-country" score,
that's real evidence the latitude/GHI features let the model transfer
to places it's never seen. If it's much worse, the model needs either
more countries, better geographic features, or both.

Usage:
    python leave_one_country_out.py
(needs at least 2 countries configured in COUNTRY_FILES below)
"""

import pandas as pd

from build_dataset import build_dataset, FEATURES, TARGET
from train_model import chronological_split, train_xgb, evaluate

COUNTRY_FILES = {
    "ES": "data/ES_solar_generaition_1_.csv",
    "DE": "data/DE_opsd_extract.csv",
    "GR": "data/GR_IT_opsd_extract.csv",
    "IT": "data/GR_IT_opsd_extract.csv",
    "PT": "data/PT_opsd_extract.csv",
}
WEATHER_DIR = "data/"


def run_loco(df: pd.DataFrame):
    countries = sorted(df["country"].unique())
    if len(countries) < 2:
        print(f"Only {len(countries)} country in the dataset ({countries}). "
              f"LOCO needs at least 2 - add another country to COUNTRY_FILES first.")
        return

    # Baseline: normal same-country test score, for comparison
    print("=" * 60)
    print("BASELINE: normal chronological split (country seen in training)")
    print("=" * 60)
    train, val, test = chronological_split(df)
    baseline_model = train_xgb(train, val)
    baseline_results = evaluate(baseline_model, test, label="Baseline (all countries in train)")

    loco_results = {}
    for held_out in countries:
        print("\n" + "=" * 60)
        print(f"LOCO: holding out {held_out}")
        print("=" * 60)

        train_countries = df[df["country"] != held_out]
        test_country = df[df["country"] == held_out]

        # train/val split still needs to be chronological, just within
        # the remaining countries (no test split needed here - the whole
        # held-out country IS the test set)
        train, val, _ = chronological_split(train_countries)

        model = train_xgb(train, val)
        results = evaluate(model, test_country, label=f"{held_out} (never seen in training)")
        loco_results[held_out] = results

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Country':<10}{'MAE (unseen)':<16}{'MAPE (unseen)':<16}")
    for country, r in loco_results.items():
        print(f"{country:<10}{r['mae']:<16.4f}{r['mape']:<16.1f}")
    print(f"\n{'Baseline':<10}{baseline_results['mae']:<16.4f}{baseline_results['mape']:<16.1f}"
          f"  (for comparison - same-country test set)")

    return loco_results, baseline_results


if __name__ == "__main__":
    df = build_dataset(COUNTRY_FILES, weather_dir=WEATHER_DIR)
    df = df.dropna(subset=FEATURES + [TARGET])
    run_loco(df)
