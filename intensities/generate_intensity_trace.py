"""Generate carbon and water intensity traces from ENTSO-E generation data.

Usage: python generate_intensity_trace.py <COUNTRY_CODE>

Fetches actual generation per production type (2018-2025) for the given
country, computes the generation-weighted effective intensity at each
instant, and writes a 15-minute resolution trace to
traces/<COUNTRY_CODE>.csv with columns: timestamp,zone,property,value.
"""

import os
import sys

import pandas as pd
from entsoe import EntsoePandasClient
from entsoe.mappings import lookup_area

# Per-source intensity factors (UNECE 2020 defaults where available, else
# IPCC 2014). Keys must match source names returned by ENTSO-E.
CARBON_INTENSITY = {  # gCO2-eq/kWh
    "Biomass": 230, "Fossil Gas": 280, "Fossil Hard coal": 630, "Fossil Oil": 280,
    "Fossil Brown coal/Lignite": 1000, "Fossil Coal-derived gas": 630,
    "Hydro Pumped Storage": 81, "Hydro Run-of-river and poundage": 81, "Hydro Water Reservoir": 81,
    "Nuclear": 5.1, "Solar": 21, "Waste": 230, "Geothermal": 38,
    "Wind Offshore": 13, "Wind Onshore": 12, "Energy storage": 21,
    "Other": 280, "Other renewable": 230,
}
WATER_INTENSITY = {  # L/kWh
    "Biomass": 1.147, "Fossil Gas": 1.086, "Fossil Hard coal": 1.802, "Fossil Oil": 1.086,
    "Fossil Brown coal/Lignite": 1.802, "Fossil Coal-derived gas": 1.802,
    "Hydro Pumped Storage": 17.0, "Hydro Run-of-river and poundage": 17.0, "Hydro Water Reservoir": 17.0,
    "Nuclear": 1.957, "Solar": 0.004, "Waste": 1.147, "Geothermal": 0.95,
    "Wind Offshore": 0, "Wind Onshore": 0, "Energy storage": 0.004,
    "Other": 1.086, "Other renewable": 1.147,
}

ZONE = "AS0"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) != 2:
    print("Usage: python generate_intensity_trace.py <COUNTRY_CODE>  (e.g. DE, FR, PL)")
    sys.exit(1)
country = sys.argv[1].upper()

tz = lookup_area(country).tz
start = pd.Timestamp("2018-01-01", tz=tz)
end = pd.Timestamp("2026-01-01", tz=tz)

raw_path = os.path.join(SCRIPT_DIR, "raw", f"{country}.csv")
os.makedirs(os.path.dirname(raw_path), exist_ok=True)
if os.path.exists(raw_path):
    print(f"Using cached generation data from {raw_path}")
    gen = pd.read_csv(raw_path, index_col=0)
    # The cache mixes CET/CEST offsets, so parse through UTC first.
    gen.index = pd.to_datetime(gen.index, utc=True).tz_convert(tz)
else:
    api_key = os.environ.get("ENTSOE_API_KEY")
    if not api_key:
        print("ENTSOE_API_KEY environment variable is not set.")
        sys.exit(1)
    print(f"Fetching generation data for {country} from {start.date()} to {end.date()}...")
    client = EntsoePandasClient(api_key=api_key)
    gen = client.query_generation(country, start=start, end=end, nett=True)
    gen.to_csv(raw_path)
    print(f"Saved raw generation data to {raw_path}")

gen = gen.loc[:, gen.sum() > 0]

unmapped = [s for s in gen.columns if s not in CARBON_INTENSITY or s not in WATER_INTENSITY]
if unmapped:
    print("Sources without intensity factors, add them to the dicts and rerun:")
    for source in unmapped:
        print(f"  - {source}")
    sys.exit(1)

# Nett values can go negative (e.g. pumped storage consuming more than it
# generates), which would distort the weighted average.
gen = gen.fillna(0).clip(lower=0)

# Rows with implausibly low totals are reporting artifacts (e.g. FR's
# all-zero duplicated DST hour in 2018, or 2023-07-20 09:00 where every
# source but a 3 MW sliver of wind reads zero). Genuine totals never drop
# below ~35% of the median, so treat anything under 20% as missing and let
# the interpolation bridge it.
total = gen.sum(axis=1)
artifacts = total < 0.2 * total.median()
if artifacts.any():
    print(f"Dropping {int(artifacts.sum())} artifact rows (total < 20% of median):")
    for ts in gen.index[artifacts]:
        print(f"  - {ts}")
gen = gen[~artifacts]

# Standardize to 15-minute resolution: existing samples keep their values,
# points in between are filled by linear interpolation.
gen = gen.resample("15min").asfreq().interpolate(method="linear")

total = gen.sum(axis=1)
gen = gen[total > 0]
total = total[total > 0]

carbon = sum(gen[s] * CARBON_INTENSITY[s] for s in gen.columns) / total
water = sum(gen[s] * WATER_INTENSITY[s] for s in gen.columns) / total

timestamps = ((gen.index - gen.index[0]).total_seconds()).astype(int)
trace = pd.DataFrame({
    "timestamp": timestamps.repeat(2),
    "zone": ZONE,
    "property": ["carbon_intensity", "water_intensity"] * len(gen),
    "value": [v for pair in zip(carbon.round(4), water.round(4)) for v in pair],
})

out_path = os.path.join(SCRIPT_DIR, "traces", f"{country}.csv")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
trace.to_csv(out_path, index=False)
print(f"Wrote {len(trace)} rows to {out_path}")
