"""
Builds one unified, feature-engineered dataset out of any list of
countries. This replaces the Spain-only merge/feature-engineering block
that used to live directly inside train_solar_model.py.

Usage:
    from build_dataset import build_dataset
    df = build_dataset({
        "ES": "data/ES_solar_generaition_1_.csv",
        "DE": "data/DE_opsd_extract.csv",
    }, weather_dir="data/")

Adding a country to the model is then just adding one more entry to that
dict (plus a country_config.py registry entry, plus its weather CSV).
"""

import numpy as np
import pandas as pd

from country_config import get_country, capacity_for_year
from data_loading import load_generation, load_weather_openmeteo


def _build_single_country(country_code: str, generation_filepath: str, weather_dir: str) -> pd.DataFrame:
    country = get_country(country_code)
    print(f"  Loading {country['name']} ({country_code})...")

    generation = load_generation(generation_filepath, country_code)
    weather = load_weather_openmeteo(f"{weather_dir.rstrip('/')}/{country['weather_file']}")

    df = pd.merge(generation, weather, on="timestamp", how="inner")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"    {len(df)} matched rows, {df['timestamp'].min()} -> {df['timestamp'].max()}")

    # --- time features (same for every country - physical, not learned) ---
    df["hour"] = df["timestamp"].dt.hour
    df["doy"] = df["timestamp"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365)

    # --- capacity factor target (normalizes for this country's fleet size) ---
    df["year"] = df["timestamp"].dt.year
    df["capacity_mw"] = df["year"].apply(lambda y: capacity_for_year(country_code, y))
    df["capacity_factor"] = df["generation_mw"] / df["capacity_mw"]

    # --- country-context features (THIS is what lets one model generalize
    #     across countries instead of confusing them - see conversation
    #     history for why this matters) ---
    df["latitude"] = country["latitude"]
    df["avg_ghi"] = country["avg_ghi_kwh_m2_day"]

    df["country"] = country_code
    return df


def build_dataset(country_files: dict, weather_dir: str = "data/") -> pd.DataFrame:
    """
    country_files: {"ES": "path/to/es_generation.csv", "DE": "path/to/de_generation.csv", ...}
    weather_dir: folder containing each country's weather CSV (filename per
                 country is looked up from country_config.py)

    Returns one combined, feature-engineered DataFrame - ready to feed to
    train_model.py or leave_one_country_out.py.
    """
    print(f"Building dataset for: {list(country_files.keys())}")
    parts = [
        _build_single_country(code, path, weather_dir)
        for code, path in country_files.items()
    ]
    combined = pd.concat(parts, ignore_index=True).sort_values(["country", "timestamp"]).reset_index(drop=True)
    print(f"\nCombined dataset: {len(combined)} rows across {combined['country'].nunique()} countries")
    return combined


FEATURES = [
    "temperature_2m", "cloud_cover", "shortwave_radiation", "wind_speed_10m",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "latitude", "avg_ghi",   # the country-context features
]
TARGET = "capacity_factor"


if __name__ == "__main__":
    # Quick sanity check when run directly
    df = build_dataset({"ES": "data/ES_solar_generaition_1_.csv"}, weather_dir="data/")
    df = df.dropna(subset=FEATURES + [TARGET])
    print(f"\nAfter dropping missing values: {len(df)} rows")
    print(df[FEATURES + [TARGET, "country"]].head())
    df.to_csv("data/combined_dataset.csv", index=False)
    print("Saved data/combined_dataset.csv")
