# Facility traces (PUE and WUE)

The campaign fixed PUE = 1 and WUE = 0. `generate_facility_traces.py` gives each country a hypothetical HPC facility and writes its PUE and WUE on the same instants as the intensity extracts in `../traces/`. The level of both parameters comes from the EU data-centre reporting scheme. The shape of WUE over time comes from the observed weather of the same dates.

## Usage

Run from the repository root:

```sh
python intensities/facility/generate_facility_traces.py
```

With the weather caches present (`raw/<CC>.csv`, Git LFS) the script runs offline. A missing cache is fetched from the [Open-Meteo archive API](https://open-meteo.com/en/docs/historical-weather-api), which needs no key. Delete a cache file to force a fresh download. The script ends by re-reading every output and checking it against its intensity extract.

Outputs:

- `traces/<CC>_<YYYY-MM-DD>.csv`: one extract per window of `../windows.csv`, same long format and same timestamps as the intensity extract of the same name. Two properties per instant, `pue` (constant) and `wue` (L/kWh of IT energy, economizer scenario). Batsim reads both, so a rerun only needs this file concatenated with the intensity extract. Tracked with Git LFS.
- `facility.csv`: one row per country with the site, its coordinates and timezone, the reported `pue` and `wue`, `wue_evap` and `evap_share`. The `wue` column is the constant of the continuous scenario.
- `raw/<CC>.csv`: hourly ERA5 `temperature_2m` (°C) and `relative_humidity_2m` (%) at the site, in UTC. Tracked with Git LFS.

```
timestamp,zone,property,value
0,AS0,pue,1.39
0,AS0,wue,0.0
900,AS0,pue,1.39
900,AS0,wue,0.0
```

## Method

**Level.** PUE and WUE are the 2024 energy-weighted national averages reported under Delegated Regulation (EU) 2024/1364, from the first technical report of the scheme (2025, Tables 25–26): DE 1.39 and 0.65 L/kWh, FR 1.55 and 0.25, PL 1.55 and 0.21. That WUE is total water *input* over IT energy, averaged over the centres that reported water > 0.

**Site.** One per country, where the main national HPC centre is: Jülich (JSC), Bruyères-le-Châtel (TGCC) and Kraków (Cyfronet). The electricity grid is national, so the site only sets the weather. Open-Meteo returns the nearest ERA5 cell (0.25°).

**Cooling scenarios.** Only the economizer needs a trace.

| Scenario | WUE | Where it lives |
| --- | --- | --- |
| Dry | 0 | the current campaign |
| Continuous evaporative | reported WUE, constant | `facility.csv` |
| Economizer | `wue_evap` in the warmest `evap_share` of each year, 0 otherwise | `traces/` |

In the economizer, the facility cools without water most of the year and evaporates only in the warmest hours. `wue_evap` = 3.6 / 2.45 ≈ 1.47 L/kWh is the water that rejects all the IT heat by evaporation (3.6 MJ per kWh over ~2.45 MJ/kg of latent heat). `evap_share` = reported WUE / `wue_evap`, so the annual mean of the trace equals the reported WUE: DE 44%, FR 17%, PL 14% of each year.

**Steps.**

1. Wet-bulb temperature from hourly ERA5 temperature and relative humidity, by Stull (2011).
2. Linear interpolation to 15 minutes, in UTC, then placed on the instants of the full trace `../traces/<CC>.csv` (2018-01-01 00:00 local, stepping 900 s).
3. For each local calendar year, the wet-bulb threshold is the quantile `1 − evap_share` of that year. Instants at or above it evaporate.
4. Each window is sliced with the rule of `../window_selection.ipynb`: start at its Monday 00:00 local and run as many steps as its intensity extract.

**What the windows look like.** Water concentrates in the afternoon (peak around 15–18 h local, trough around 6 h). The national level, though, sets how much of the day is left dry. In DE, `evap_share` is 44%, so the threshold falls to a wet-bulb of 9–11 °C and the summer windows evaporate 96–100% of the time. They are close to the continuous scenario, while the winter ones are close to dry. In FR and PL the threshold is 15–17 °C, and the summer windows evaporate 22–62% of the time, with a real daily cycle.

## Limitations

- The reported WUE counts water input, not consumption. It bounds on-site consumption from above.
- These are national data-centre averages, dominated by colocation, not HPC values. The WUE average excludes centres that reported no water.
- The economizer is a hypothetical archetype, not a measured facility. Ranking by wet-bulb follows cooling-tower models; adiabatic dry coolers switch on by dry-bulb temperature instead.
- PUE is held constant. Its weather dependence would only scale carbon and off-site water a little more in the warm hours.

## Sources

- Commission Delegated Regulation (EU) 2024/1364, Annex III (EN 50600-4-2 and 4-9).
- DG ENER, AIT, Borderstep and EY (2025). *Assessment of the energy performance and sustainability of data centres in EU – First technical report*. doi:10.2833/3168794.
- Stull, R. (2011). Wet-bulb temperature from relative humidity and air temperature. *J. Appl. Meteor. Climatol.* 50, 2267–2269.
- Hersbach, H., et al. (2020). The ERA5 global reanalysis. *QJRMS* 146, 1999–2049.
- Zippenfenig, P. (2023). Open-Meteo.com Weather API. doi:10.5281/zenodo.7970649. Data under CC BY 4.0.
