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
    "GR": {
        "name": "Greece",
        "generation_source": "opsd_wide",
        "opsd_column": "GR_solar_generation_actual",
        "weather_file": "open-meteo-athens.csv",
        "latitude": 37.98,
        "longitude": 23.73,
        "avg_ghi_kwh_m2_day": 4.5,   # Greece gets ~50% more solar irradiation than Germany (Wikipedia)
        "capacity_mw_by_year": {
            # APPROXIMATE (Wikipedia/Statista) - capacity was fairly flat 2015-2018, growth
            # resumed 2019+. Verify against Our World in Data before relying on this.
            2015: 2600, 2016: 2610, 2017: 2650,
            2018: 2650, 2019: 2830, 2020: 3020,
        },
    },
    "IT": {
        "name": "Italy",
        "generation_source": "opsd_wide",
        "opsd_column": "IT_solar_generation_actual",
        "weather_file": "open-meteo-rome.csv",
        "latitude": 41.90,
        "longitude": 12.49,
        "avg_ghi_kwh_m2_day": 4.0,   # ranges ~3.6 (Po valley, north) to ~4.6 (Sicily, south) - Rome is roughly the middle
        "capacity_mw_by_year": {
            # APPROXIMATE (Wikipedia: ~2% annual growth, 300-400 MW/year 2014-2018) - verify
            # against Our World in Data before relying on this.
            2015: 18900, 2016: 19300, 2017: 19700,
            2018: 20100, 2019: 20900, 2020: 21600,
        },
    },
    "PT": {
        "name": "Portugal",
        "generation_source": "opsd_wide",
        "opsd_column": "PT_solar_generation_actual",
        "weather_file": "open-meteo-faro.csv",
        "latitude": 37.02,
        "longitude": -7.93,
        "avg_ghi_kwh_m2_day": 4.8,   # southern Portugal (Faro/Algarve) - one of Europe's sunniest regions, slightly higher than Seville
        "capacity_mw_by_year": {
            # APPROXIMATE (Wikipedia/Statista/IEA-PVPS) - Portugal grew slowly until 2018 then
            # accelerated (+190MW 2018, +220MW 2019, reaching ~1030MW by end of 2020).
            # Verify against Our World in Data before relying on this.
            2015: 400, 2016: 430, 2017: 480,
            2018: 670, 2019: 890, 2020: 1030,
        },
    },
    "AT": {
        "name": "Austria",
        "generation_source": "opsd_wide",
        "opsd_column": "AT_solar_generation_actual",
        "weather_file": "open-meteo-vienna.csv",
        "latitude": 48.19,
        "longitude": 16.38,
        "avg_ghi_kwh_m2_day": 3.2,   # APPROXIMATE - Alpine/continental, similar order to Germany
        "capacity_mw_by_year": {
            # APPROXIMATE (Wikipedia/SolarPower Europe) - Austria's PV market was fairly small
            # and slow-growing until a recent boom (post-2021, not covered by our data range).
            # Verify against Our World in Data before relying on this.
            2015: 800, 2016: 900, 2017: 1000,
            2018: 1100, 2019: 1200, 2020: 1400,
        },
    },
    "BE": {
        "name": "Belgium",
        "generation_source": "opsd_wide",
        "opsd_column": "BE_solar_generation_actual",
        "weather_file": "open-meteo-brussels.csv",
        "latitude": 50.86,
        "longitude": 4.33,
        "avg_ghi_kwh_m2_day": 2.8,   # APPROXIMATE - similar cloudy/temperate profile to Germany/Netherlands
        "capacity_mw_by_year": {
            # APPROXIMATE (Wikipedia: 2,983 MW end of 2013, 4,254 MW end of 2018, steady growth
            # in between) - verify against Our World in Data before relying on this.
            2015: 3200, 2016: 3400, 2017: 3800,
            2018: 4254, 2019: 4500, 2020: 4700,
        },
    },
    "DK": {
        "name": "Denmark",
        "generation_source": "opsd_wide",
        "opsd_column": "DK_solar_generation_actual",
        "weather_file": "open-meteo-copenhagen.csv",
        "latitude": 55.64,
        "longitude": 12.60,
        "avg_ghi_kwh_m2_day": 2.6,   # APPROXIMATE - Denmark has lower solar insolation than most of
                                     # our other countries (Wikipedia: "lower solar insolation than
                                     # many countries closer to Equator")
        "capacity_mw_by_year": {
            # APPROXIMATE (Wikipedia: 200MW target reached in 2012, growth was fairly slow/flat
            # through the later 2010s until incentive changes) - verify against Our World in Data
            # before relying on this.
            2015: 800, 2016: 850, 2017: 900,
            2018: 950, 2019: 1000, 2020: 1050,
        },
    },
    # Add Netherlands/Poland (different generation_source) here in future.
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
