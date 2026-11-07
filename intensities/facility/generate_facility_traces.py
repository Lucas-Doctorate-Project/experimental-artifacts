"""Facility traces (PUE and WUE) for the 4-week intensity windows.

The campaign fixed PUE = 1 and WUE = 0. This script gives each country a
hypothetical HPC facility and writes its PUE and WUE on the same instants as
the intensity extracts in ../traces/:

- PUE is constant, the 2024 national average reported under the EU data-centre
  reporting scheme.
- WUE follows the "economizer" cooling scenario. The facility cools without
  water most of the year and evaporates water only in the warmest hours. It
  uses WUE_EVAP in the instants whose wet-bulb temperature is in the top
  evap_share of their calendar year, and 0 otherwise. evap_share is chosen so
  that the annual mean equals the reported national WUE.

The two other cooling scenarios need no trace. "Dry" is WUE = 0, which is the
current campaign. "Continuous" is the reported WUE as a constant, written to
facility.csv.

Terms used throughout:

- country: two-letter code of the electricity zone, "DE", "FR" or "PL". It is
  the `zone` column of ../windows.csv and the prefix of the trace files.
- instant: one 15-minute step of a trace.
- full trace: ../traces/<country>.csv, 2018-2025. Window extract:
  ../traces/<country>_<YYYY-MM-DD>.csv, one per window of ../windows.csv.

Usage, from the experimental-artifacts repository root:

    python intensities/facility/generate_facility_traces.py

The hourly ERA5 weather is cached in raw/<country>.csv. A missing cache is
fetched from the Open-Meteo archive API, which needs no key. Delete a cache
file to force a fresh download.
"""

import json
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


FACILITY_DIR = Path(__file__).resolve().parent
INTENSITIES_DIR = FACILITY_DIR.parent
RAW_DIR = FACILITY_DIR / "raw"
TRACES_DIR = FACILITY_DIR / "traces"

ZONE_ID = "AS0"  # zone name used inside the traces
STEP_SEC = 900
STEPS_PER_DAY = 96
SCHEDULING_DAYS = 28  # a window extract is the 28-day scheduling window plus a tail
TRACE_START, TRACE_END = "2018-01-01", "2026-01-01"  # local time, end exclusive, as in ../generate_traces.ipynb
N_INSTANTS = 280512  # instants of the full trace

# Water that rejects all the IT heat by evaporation. Each kWh of IT electricity
# becomes 3.6 MJ of heat, and evaporating 1 kg (1 L) of water absorbs the
# latent heat of vaporization, ~2.45 MJ near 20 °C.
WUE_EVAP = 3.6 / 2.45  # L/kWh, about 1.47

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
# UTC dates, one day of margin on each side of the local-time trace.
WEATHER_START, WEATHER_END = "2017-12-31", "2026-01-01"


@dataclass(frozen=True)
class Facility:
    """The hypothetical HPC facility of one country.

    pue and wue are the 2024 energy-weighted national averages of the EU
    reporting scheme (Delegated Regulation (EU) 2024/1364; First technical
    report, 2025, Tables 25-26). That wue is total water input over IT energy,
    averaged over the centres that reported water > 0.
    """

    site: str  # town of the main national HPC centre; the grid is national, so the site only sets the weather
    latitude: float
    longitude: float
    timezone: str  # the country's local time, which is also the time of the traces
    pue: float
    wue: float  # L/kWh of IT energy

    @property
    def evap_share(self) -> float:
        """Fraction of each year in which the economizer evaporates.

        In the economizer each instant uses either WUE_EVAP or no water, so the
        annual mean WUE is evap_share * WUE_EVAP. Making it equal to the
        reported wue gives evap_share = wue / WUE_EVAP: DE 44%, FR 17% and
        PL 14% of each year.
        """
        return self.wue / WUE_EVAP


FACILITIES = {
    "DE": Facility(site="Jülich", latitude=50.905, longitude=6.413,
                   timezone="Europe/Berlin", pue=1.39, wue=0.65),
    "FR": Facility(site="Bruyères-le-Châtel", latitude=48.597, longitude=2.193,
                   timezone="Europe/Paris", pue=1.55, wue=0.25),
    "PL": Facility(site="Kraków", latitude=50.067, longitude=19.912,
                   timezone="Europe/Warsaw", pue=1.55, wue=0.21),
}


# --- Weather ------------------------------------------------------------------

def fetch_weather(country: str) -> pd.DataFrame:
    """Hourly ERA5 2 m temperature (°C) and relative humidity (%) at the site, indexed in UTC.

    One request per country to the Open-Meteo archive API covers the 8 years.
    models=era5 forces plain ERA5, since the default blends other models in
    recent years. timezone=GMT returns UTC, which has no DST ambiguity.
    Open-Meteo answers with the nearest ERA5 cell (0.25°), printed for the record.
    """
    facility = FACILITIES[country]
    query = urllib.parse.urlencode({
        "latitude": facility.latitude,
        "longitude": facility.longitude,
        "start_date": WEATHER_START,
        "end_date": WEATHER_END,
        "hourly": "temperature_2m,relative_humidity_2m",
        "models": "era5",
        "timezone": "GMT",
    })
    print(f"{country}: fetching ERA5 from {WEATHER_START} to {WEATHER_END}...")
    with urllib.request.urlopen(f"{OPEN_METEO_URL}?{query}", timeout=300) as response:
        payload = json.load(response)
    print(f"{country}: nearest ERA5 cell at {payload['latitude']} N, {payload['longitude']} E, "
          f"elevation {payload['elevation']} m")

    hourly = payload["hourly"]
    weather = pd.DataFrame(
        {"temperature_2m": hourly["temperature_2m"],
         "relative_humidity_2m": hourly["relative_humidity_2m"]},
        index=pd.to_datetime(hourly["time"]).tz_localize("UTC"),
    )
    weather.index.name = "time"
    return weather


def load_weather(country: str) -> pd.DataFrame:
    """Hourly site weather indexed in UTC, read from raw/<country>.csv.

    Same cache pattern as ../generate_traces.ipynb: the API is only called when
    the cache file is missing, and the answer is saved for the next run.
    """
    path = RAW_DIR / f"{country}.csv"
    if not path.exists():
        RAW_DIR.mkdir(exist_ok=True)
        fetch_weather(country).to_csv(path)
    weather = pd.read_csv(path, index_col=0)
    weather.index = pd.to_datetime(weather.index, utc=True)
    return weather


def compute_wet_bulb(temperature: float | pd.Series, relative_humidity: float | pd.Series) -> float | pd.Series:
    """Wet-bulb temperature (°C) from air temperature (°C) and relative humidity (%).

    Stull (2011), J. Appl. Meteor. Climatol. 50, 2267-2269. Written with numpy,
    so it works on a single value or on a whole series. Fitted for RH 5-99% and
    T -20 to 50 °C. The error grows only in very cold hours, which are never
    near the warm end that sets the threshold.
    """
    t, rh = temperature, relative_humidity
    return (t * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
            + np.arctan(t + rh) - np.arctan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
            - 4.686035)


# Worked example in Stull (2011): T = 20 °C and RH = 50% give Tw = 13.7 °C.
# Checked on import, so a mistyped coefficient fails at once.
assert abs(compute_wet_bulb(20.0, 50.0) - 13.7) < 0.05


# --- Economizer WUE on the full trace -----------------------------------------

def build_trace_grid(country: str) -> pd.DatetimeIndex:
    """The instants of the full trace ../traces/<country>.csv, in local time.

    Timestamp 0 is 2018-01-01 00:00 local, and every step is 15 minutes of
    real elapsed time. So a DST day has 92 or 100 instants, exactly like the
    trace. The check against N_INSTANTS confirms both grids match.
    """
    timezone = FACILITIES[country].timezone
    grid = pd.date_range(pd.Timestamp(TRACE_START, tz=timezone), pd.Timestamp(TRACE_END, tz=timezone),
                         freq="15min", inclusive="left")
    assert len(grid) == N_INSTANTS, country
    return grid


def compute_wet_bulb_on_grid(weather: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.Series:
    """Wet-bulb temperature (°C) on every instant of the trace grid.

    Hourly wet-bulb is upsampled to 15 minutes by linear interpolation, the
    same upsampling the hourly ENTSO-E countries get. It is done in UTC to stay
    clear of DST, then placed on the local-time grid. Every hourly UTC instant
    falls on the grid, because CET and CEST are whole-hour offsets. The final
    check fails if the weather does not cover the whole trace.
    """
    wet_bulb = compute_wet_bulb(weather["temperature_2m"], weather["relative_humidity_2m"])
    wet_bulb = wet_bulb.resample("15min").asfreq().interpolate(method="linear")
    wet_bulb = wet_bulb.reindex(grid.tz_convert("UTC"))
    assert not wet_bulb.isna().any(), "weather does not cover the whole trace"
    wet_bulb.index = grid
    return wet_bulb


def compute_yearly_thresholds(wet_bulb: pd.Series, evap_share: float) -> pd.Series:
    """Wet-bulb threshold (°C) of each local calendar year, indexed by year.

    The threshold is the wet-bulb value with evap_share of the year's instants
    at or above it, that is, the quantile 1 - evap_share of that year. It is
    not a setpoint of a real facility, but the line that leaves enough warm
    instants above it for the annual mean WUE to match the reported one. In DE
    the economizer must evaporate 44% of the year, so the threshold falls near
    the annual median wet-bulb of Jülich, around 10 °C. FR and PL only need
    their warmest 14-17% of instants, so their threshold is a summer value,
    around 15-17 °C.

    Why one threshold per year, and not one for all 8 years: each year then
    matches the reported mean, so a hot year does not take water from a cold
    one. Why per calendar year, and not per window: a winter window would
    otherwise have to evaporate as much as a summer window.
    """
    return wet_bulb.groupby(wet_bulb.index.year).quantile(1 - evap_share)


def compute_economizer_wue(wet_bulb: pd.Series, thresholds: pd.Series) -> pd.Series:
    """Economizer WUE (L/kWh) on every instant: WUE_EVAP at or above its year's threshold, 0 below.

    This on/off shape puts the reported annual water in as few instants as
    physics allows (each one at its maximum, WUE_EVAP), so it is the most
    concentrated shape in time that is compatible with the reported mean.
    Water ends up in the afternoon, peaking around 15-18 h local time.
    """
    own_year_threshold = thresholds.loc[wet_bulb.index.year].to_numpy()
    evaporating = wet_bulb >= own_year_threshold
    return evaporating.astype(float) * WUE_EVAP


def build_full_trace_wue(country: str) -> pd.Series:
    """Economizer WUE (L/kWh) on every instant of the country's full trace.

    Chains the steps above: site weather, wet-bulb on the trace grid, yearly
    thresholds, WUE. Then checks that each year's mean WUE matches the
    reported one, which proves the calibration worked. Interpolated wet-bulb
    has few ties, so the match is within 1%.
    """
    facility = FACILITIES[country]
    wet_bulb = compute_wet_bulb_on_grid(load_weather(country), build_trace_grid(country))
    thresholds = compute_yearly_thresholds(wet_bulb, facility.evap_share)
    wue = compute_economizer_wue(wet_bulb, thresholds)

    annual_mean = wue.groupby(wue.index.year).mean()
    assert np.allclose(annual_mean, facility.wue, rtol=0.01), (country, annual_mean)
    print(f"{country}: evaporates in {facility.evap_share:.1%} of each year, "
          f"wet-bulb threshold {thresholds.min():.1f} to {thresholds.max():.1f} °C over "
          f"{thresholds.index.min()}-{thresholds.index.max()}")
    return wue


# --- Outputs ------------------------------------------------------------------

def write_facility_table() -> None:
    """Write facility.csv, one row per country, the handoff to the analysis scripts.

    Its wue column is the constant of the continuous scenario, and pue is the
    constant used in every scenario.
    """
    table = pd.DataFrame([asdict(facility) for facility in FACILITIES.values()],
                         index=pd.Index(list(FACILITIES), name="zone"))
    table["wue_evap"] = round(WUE_EVAP, 4)
    table["evap_share"] = (table["wue"] / WUE_EVAP).round(4)
    table.to_csv(FACILITY_DIR / "facility.csv")


def find_window_start_step(country: str, start_date: str) -> int:
    """Index, in the full trace, of the first instant of a window.

    Same rule as ../window_selection.ipynb. A window starts at its Monday
    00:00 local time, and the full trace counts elapsed seconds from
    2018-01-01 00:00 local time, so the index is the elapsed time over 900 s.
    """
    timezone = FACILITIES[country].timezone
    elapsed = pd.Timestamp(start_date, tz=timezone) - pd.Timestamp(TRACE_START, tz=timezone)
    return int(elapsed.total_seconds()) // STEP_SEC


def count_instants(trace_path: Path) -> int:
    """Number of distinct timestamps in a trace file in Batsim's long format."""
    return pd.read_csv(trace_path)["timestamp"].nunique()


def format_batsim_trace(pue: float, wue: pd.Series) -> pd.DataFrame:
    """Batsim's environmental trace format, with pue and then wue at every instant from timestamp 0.

    Same layout as the intensity extracts (see window_extract in
    ../window_selection.ipynb): timestamp,zone,property,value.
    """
    n = len(wue)
    return pd.DataFrame({
        "timestamp": (np.arange(n) * STEP_SEC).repeat(2),
        "zone": ZONE_ID,
        "property": ["pue", "wue"] * n,
        "value": np.column_stack([np.full(n, pue), wue.round(4)]).ravel(),
    })


def write_window_extracts(windows: pd.DataFrame, wue_by_country: dict[str, pd.Series]) -> None:
    """Write traces/<country>_<YYYY-MM-DD>.csv for every window of ../windows.csv.

    Each extract starts at the window's first instant and has as many instants
    as its intensity extract (4608, 48 days), so both files line up instant by
    instant. Stale extracts are removed first, so the directory always mirrors
    windows.csv. Prints the mean WUE and the share of time evaporating over
    the 28-day scheduling window, without the tail.
    """
    TRACES_DIR.mkdir(exist_ok=True)
    for stale in TRACES_DIR.glob("*.csv"):
        stale.unlink()

    for window in windows.itertuples():
        country = window.zone
        start = find_window_start_step(country, window.start_date)
        n_instants = count_instants(INTENSITIES_DIR / window.file)
        wue = wue_by_country[country].iloc[start : start + n_instants]
        assert len(wue) == n_instants, window.file

        trace = format_batsim_trace(FACILITIES[country].pue, wue)
        trace.to_csv(TRACES_DIR / Path(window.file).name, index=False)

        scheduling = wue.iloc[: SCHEDULING_DAYS * STEPS_PER_DAY]
        print(f"{country} {window.start_date} {window.season:<6}: mean WUE {scheduling.mean():.3f} L/kWh, "
              f"evaporating {(scheduling > 0).mean():.0%} of the time")


def validate_window_extracts(windows: pd.DataFrame) -> None:
    """Re-read every facility extract from disk and check it against its intensity extract.

    Checks: zone always AS0, no missing values, exactly the pue and wue
    properties, and the same timestamps as the intensity extract.
    """
    for window in windows.itertuples():
        facility_trace = pd.read_csv(TRACES_DIR / Path(window.file).name)
        intensity_trace = pd.read_csv(INTENSITIES_DIR / window.file)
        assert (facility_trace["zone"] == ZONE_ID).all(), window.file
        assert not facility_trace.isna().any().any(), window.file
        wide = facility_trace.pivot(index="timestamp", columns="property", values="value")
        assert list(wide.columns) == ["pue", "wue"], window.file
        assert (wide.index == np.sort(intensity_trace["timestamp"].unique())).all(), window.file
    print(f"validated {len(windows)} facility extracts against their intensity extracts")


def main() -> None:
    windows = pd.read_csv(INTENSITIES_DIR / "windows.csv")
    wue_by_country = {country: build_full_trace_wue(country) for country in FACILITIES}
    write_facility_table()
    write_window_extracts(windows, wue_by_country)
    validate_window_extracts(windows)


if __name__ == "__main__":
    main()
