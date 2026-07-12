"""
Central registry of all countries the model can be trained on.

To add a new country: add one entry to COUNTRIES below. That's it - the
rest of the pipeline (data_loading.py, build_dataset.py, train_model.py,
leave_one_country_out.py) reads this dict and needs no other changes,
AS LONG AS the country's generation data comes from a source we already
have an adapter for (currently: "opsd_wide" - see data_loading.py).

If a country's generation data comes from a source we don't have yet
(e.g. Terna for Italy, CBS for Netherlands), you need to:
  1. Write a new loader function in data_loading.py (see load_generation_opsd
     for the pattern to follow - it must return a DataFrame with exactly
     two columns: "timestamp" and "generation_mw")
  2. Set this country's "generation_source" to a new string (e.g. "terna")
  3. Add a matching "elif source == 'terna':" branch in
     data_loading.load_generation()

WHERE THE NUMBERS COME FROM:
  - capacity_mw_by_year: end-of-year cumulative installed PV capacity (MW).
    Spain: Red Electrica de Espana figures via pv-magazine / Wikipedia.
    Germany: Fraunhofer ISE / Federal Network Agency (Bundesnetzagentur)
    cumulative figures - APPROXIMATE, verify against Our World in Data
    (https://ourworldindata.org/grapher/installed-solar-pv-capacity) before
    using for anything beyond a first prototype.
  - avg_ghi_kwh_m2_day: rough average daily solar irradiance (Global
    Horizontal Irradiance), used as a static "how sunny is this country
    on average" feature. Approximate values from Global Solar Atlas
    (https://globalsolaratlas.info) - re-check per country if precision
    matters.
  - latitude/longitude: representative point for the country's weather
    data (should match the coordinates used in the Open-Meteo CSV
    filename/download, not just "the capital").
"""

COUNTRIES = {
    "ES": {
        "name": "Spain",
        "generation_source": "opsd_wide",
        "opsd_column": "ES_solar_generation_actual",
        "weather_file": "open-meteo-37_36N5_98W21m__1_.csv",
        "latitude": 37.36,
        "longitude": -5.98,
        "avg_ghi_kwh_m2_day": 4.6,   # Seville area - high irradiance
        "capacity_mw_by_year": {
            2015: 4270, 2016: 4330, 2017: 4460,
            2018: 4730, 2019: 8700, 2020: 9500,
        },
    },
    "DE": {
        "name": "Germany",
        "generation_source": "opsd_wide",
        "opsd_column": "DE_solar_generation_actual",
        "weather_file": "open-meteo-berlin.csv",
        "latitude": 52.52,
        "longitude": 13.41,
        "avg_ghi_kwh_m2_day": 2.9,   # Berlin area - much lower irradiance than Spain
        "capacity_mw_by_year": {
            # APPROXIMATE cumulative PV capacity (Fraunhofer ISE) - verify before
            # relying on this for anything beyond a first prototype.
            2015: 39700, 2016: 41300, 2017: 42900,
            2018: 45900, 2019: 49500, 2020: 54000,
        },
    },
    # Add Greece, and later Italy/Netherlands/Poland (different generation_source), here.
}


def get_country(code: str) -> dict:
    """Look up a country's config by its 2-letter code, with a clear error
    if it hasn't been added to the registry yet."""
    code = code.upper()
    if code not in COUNTRIES:
        raise KeyError(
            f"Country '{code}' is not in COUNTRIES yet. "
            f"Available: {list(COUNTRIES.keys())}. "
            f"Add an entry to country_config.py to use it."
        )
    return COUNTRIES[code]


def capacity_for_year(code: str, year: int) -> float:
    """Cumulative installed capacity (MW) for a country in a given year.
    Falls back to the nearest known year if the exact year is missing
    (capacity changes slowly, so this is a reasonable approximation)."""
    cap_by_year = get_country(code)["capacity_mw_by_year"]
    if year in cap_by_year:
        return cap_by_year[year]
    nearest_year = min(cap_by_year, key=lambda y: abs(y - year))
    return cap_by_year[nearest_year]
