"""
Data loading adapters. Each adapter's job is to take a raw file (in
whatever format that source uses) and return a clean, standardized
DataFrame - so the rest of the pipeline never has to know or care
whether the data came from OPSD, Terna, or anywhere else.

STANDARD FORMAT (every adapter must return exactly this):
    Generation:  columns ["timestamp", "generation_mw"]
    Weather:     columns ["timestamp", "temperature_2m", "cloud_cover",
                           "shortwave_radiation", "wind_speed_10m"]

To add a new generation source (e.g. Terna for Italy):
    1. Write `load_generation_terna(filepath, country_code) -> DataFrame`
       following the exact same input/output contract as
       load_generation_opsd below.
    2. Add it to the `_GENERATION_LOADERS` dict at the bottom of this file.
    3. In country_config.py, set that country's "generation_source" to
       "terna" (or whatever key you registered it under).
No other file needs to change.
"""

import pandas as pd
from country_config import get_country


# ------------------------------------------------------------------
# Generation adapters
# ------------------------------------------------------------------

def load_generation_opsd(filepath: str, country_code: str) -> pd.DataFrame:
    """
    Loads generation data from an OPSD-format file.

    Works with EITHER:
      - the full multi-country OPSD wide file (time_series_60min_singleindex.csv),
        which has one column per country like "DE_solar_generation_actual",
        "ES_solar_generation_actual", etc. - the right column is picked
        automatically based on country_code
      - a single-country extract that already contains only that column
        (like Chavi's current ES_solar_generaition_1_.csv)
    Either way you get the same clean two-column output, so you never need
    to manually re-extract a column for a new country - just point this
    function at the full OPSD file and give it the country code.
    """
    country = get_country(country_code)
    col = country["opsd_column"]

    df = pd.read_csv(filepath, usecols=["utc_timestamp", col])
    df = df.rename(columns={col: "generation_mw", "utc_timestamp": "timestamp"})
    df = df.dropna(subset=["generation_mw"])
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    return df[["timestamp", "generation_mw"]]


_GENERATION_LOADERS = {
    "opsd_wide": load_generation_opsd,
    # "terna": load_generation_terna,          # add when Italy is implemented
    # "cbs_netherlands": load_generation_cbs,  # add when Netherlands is implemented
}


def load_generation(filepath: str, country_code: str) -> pd.DataFrame:
    """Dispatches to the right adapter based on this country's
    generation_source, as registered in country_config.py."""
    country = get_country(country_code)
    source = country["generation_source"]
    if source not in _GENERATION_LOADERS:
        raise NotImplementedError(
            f"No loader registered for generation_source='{source}' "
            f"(needed for {country_code}). Add one to _GENERATION_LOADERS "
            f"in data_loading.py."
        )
    return _GENERATION_LOADERS[source](filepath, country_code)


# ------------------------------------------------------------------
# Weather adapter (Open-Meteo - same format regardless of country,
# since it's the same API/export for every location)
# ------------------------------------------------------------------

def load_weather_openmeteo(filepath: str) -> pd.DataFrame:
    """
    Loads an Open-Meteo hourly CSV export.

    Handles the known Open-Meteo export quirks: a 3-line metadata/blank
    header block before the real header row, and column names that
    include units (e.g. "temperature_2m (°C)").
    """
    df = pd.read_csv(filepath, skiprows=3)
    df = df.rename(columns={
        "temperature_2m (°C)": "temperature_2m",
        "cloud_cover (%)": "cloud_cover",
        "shortwave_radiation (W/m²)": "shortwave_radiation",
        "wind_speed_10m (km/h)": "wind_speed_10m",
        "time": "timestamp",
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df[["timestamp", "temperature_2m", "cloud_cover",
               "shortwave_radiation", "wind_speed_10m"]]
