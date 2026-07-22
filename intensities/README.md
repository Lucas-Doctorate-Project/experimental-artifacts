# Carbon and water intensity traces

`generate_traces.ipynb` builds the carbon and water intensity traces from the ENTSO-E transparency platform. For each country (PL, FR, DE), it takes the actual generation per production type from 2018 through 2025, computes the generation-weighted intensity of each instant, and writes a 15-minute resolution trace. The notebook also plots the full series, a zoomed sample week showing the diurnal pattern, and the generation mix of each country.

Energy storage and pumped-storage hydro are excluded from the weighted mix because the data do not identify their charging mix, temporal attribution, or round-trip losses. All other production types are treated consistently across countries.

## Usage

Run from the Nix dev shell at the repo root:

```sh
jupyter nbconvert --to notebook --execute --inplace intensities/generate_traces.ipynb
```

One run covers all three countries. With the raw caches present (`raw/<CC>.csv`, checked out via Git LFS) the notebook runs fully offline. A country is only fetched from the ENTSO-E API when its cache file is missing, which requires the `ENTSOE_API_KEY` environment variable. Delete a cache file to force a fresh download.

Outputs, both tracked with Git LFS:

- `traces/<CC>.csv`: the trace. Timestamps are seconds relative to the trace start, stepping by 900. The zone is always `AS0`. Two properties per instant: `carbon_intensity` (gCO2eq/kWh) and `water_intensity` (L/kWh).
- `raw/<CC>.csv`: the raw generation data, written when fetched from the API.

```
timestamp,zone,property,value
0,AS0,carbon_intensity,30.9523
0,AS0,water_intensity,3.6708
900,AS0,carbon_intensity,31.2149
900,AS0,water_intensity,3.6826
```

Behavior notes:

- Countries reporting hourly are upsampled to 15 minutes by linear interpolation.
- Rows whose total generation is below 20% of the country median are reporting artifacts, for example FR's all-zero DST fall-back hour in 2018 and a near-zero row on 2023-07-20. They are treated as missing data, bridged by interpolation, and printed when dropped. Genuine totals never fall below ~35% of the median.
- The notebook aborts if ENTSO-E reports a source that is missing from the intensity dicts. Add the source to both dicts and rerun.

## Window sampling

`window_selection.ipynb` samples contiguous 4-week windows from the full traces for the scheduling experiments. Per zone, it enumerates every window anchored at a Monday 00:00 local time, then draws 3 windows per meteorological season from distinct years, using a fixed RNG seed. Each window carries descriptors for the later analysis: mean carbon and water intensity, and the swing of each signal.

The swing (`swing_carbon`, `swing_water`) measures how much an intensity signal moves up and down within a typical day of the window. For each of the 28 days, take the difference between the day's highest and lowest intensity. Average these daily ranges over the window, then divide by the window's mean intensity. A swing of 0.5 means the signal moves by about half its average level within a day. It matters because the daily ups and downs are what a time-aware scheduler can exploit: a high swing means there are much cleaner hours within reach, while a low swing means the signal is flat and there is little to gain. Reporting it per window lets the analysis correlate the gains of each heuristic with how much room each signal actually offered.

Run it from the Nix dev shell at the repo root:

```sh
jupyter nbconvert --to notebook --execute --inplace intensities/window_selection.ipynb
```

Outputs:

- `traces/<CC>_<YYYY-MM-DD>.csv`: one extract per sampled window, named by zone and start date. Same long format as the full traces, timestamps rebased to start at 0, 3360 instants (35 days at 15-minute resolution). Each extract covers the 28-day scheduling window plus a 7-day tail so the intensity trace outlasts simulations whose jobs spill past the 4-week mark. Window descriptors are computed on the 28-day portion only. Tracked with Git LFS like the full traces.
- `windows.csv`: the manifest the experiment runner iterates over, with columns `zone`, `file`, `start_date`, `season`, `year`, `mean_carbon`, `mean_water`, `swing_carbon`, `swing_water`.

## Intensity factors

Lookup table of intensity factors for water consumption and carbon emissions from energy generation technologies. The water factors are from Macknick et al. 2012[^macknick2012], while carbon factors are from IPCC 2014[^ipcc2014] and UNECE 2020[^unece2020] reports, as compiled by Wikipedia[^wikipedia]. The units are L/kWh (liters per kilowatt-hour) for water and gCO2eq/kWh (grams of CO2 equivalent per kilowatt-hour) for carbon. Water values were converted from gallons/MWh to L/kWh. Entries named "average" are the mean of all listed entries of that fuel type.

### Mapping to ENTSO-E production types

The trace generator uses UNECE 2020 values where available, else IPCC 2014:

| ENTSO-E source | Carbon | Water | Basis |
| --- | --- | --- | --- |
| Biomass | 230 | 1.147 | Biopower default (IPCC 2014), biopower water average |
| Fossil Brown coal/Lignite | 1000 | 1.802 | Coal, pulverized (UNECE 2020), coal water average [^lignite] |
| Fossil Coal-derived gas | 630 | 1.802 | Coal average (UNECE 2020) [^coalgas] |
| Fossil Gas | 280 | 1.086 | Gas average (UNECE 2020) |
| Fossil Hard coal | 630 | 1.802 | Coal average (UNECE 2020) |
| Fossil Oil | 280 | 1.086 | Gas average (UNECE 2020) [^oil] |
| Geothermal | 38 | 0.95 | Geothermal default (IPCC 2014), geothermal water average |
| Hydro Run-of-river and poundage | 81 | 17.0 | Hydropower average (UNECE 2020) |
| Hydro Water Reservoir | 81 | 17.0 | Hydropower average (UNECE 2020) |
| Nuclear | 5.1 | 1.957 | Nuclear default (UNECE 2020), nuclear water average |
| Other | 280 | 1.086 | Gas average (UNECE 2020) [^other] |
| Other renewable | 230 | 1.147 | Biopower values [^otherrenewable] |
| Solar | 21 | 0.004 | Solar photovoltaic average (UNECE 2020) |
| Waste | 230 | 1.147 | Biopower values [^waste] |
| Wind Offshore | 13 | 0 | Wind average (UNECE 2020) |
| Wind Onshore | 12 | 0 | Wind, onshore (UNECE 2020) |

[^lignite]: No lignite entry was collected. Pulverized coal (UNECE 2020), the highest collected coal value, is used because lignite emits more than hard coal.
[^coalgas]: Blast furnace and coke oven gas. Treated as coal rather than natural gas because it is a coal byproduct with near-coal emission factors.
[^oil]: No oil entry was collected. The gas values are reused as the closest fossil proxy.
[^other]: ENTSO-E's unclassified category, mostly fossil industrial generation. The gas average is a middle-of-the-road fossil proxy.
[^otherrenewable]: Commonly biogas and small biomass plants, so the biopower values are used.
[^waste]: Waste incineration plants are comparable to biopower steam plants, so the biopower values are used.

### Collected values

| Technology | IPCC 2014 carbon (gCO2-eq/kWh) | UNECE 2020 carbon (gCO2-eq/kWh) | Water (L/kWh) |
| --- | --- | --- | --- |
| Coal, average | 820 | 630 | 1.802 |
| Coal, generic, tower cooling | | | 2.601 |
| Coal, pulverized | 820 | 1000 | |
| Coal, pulverized, subcritical, tower cooling | | | 1.813 |
| Coal, pulverized, supercritical, tower cooling | | | 1.866 |
| Coal, supercritical | | 950 | |
| Coal, integrated gasification combined cycle | | 850 | |
| Coal, integrated gasification combined cycle, tower cooling | | | 1.438 |
| Coal, pulverized, with carbon capture and storage | | 370 | |
| Coal, pulverized, subcritical, with carbon capture and storage, tower cooling | | | 3.486 |
| Coal, pulverized, supercritical, with carbon capture and storage, tower cooling | | | 3.203 |
| Coal, supercritical, with carbon capture and storage | | 330 | |
| Coal, integrated gasification combined cycle, with carbon capture and storage | | 280 | |
| Coal, integrated gasification combined cycle, with carbon capture and storage, tower cooling | | | 2.078 |
| Coal, generic, once-through cooling | | | 0.946 |
| Coal, pulverized, subcritical, once-through cooling | | | 0.428 |
| Coal, pulverized, supercritical, once-through cooling | | | 0.39 |
| Coal, generic, pond cooling | | | 2.063 |
| Coal, pulverized, subcritical, pond cooling | | | 2.949 |
| Coal, pulverized, supercritical, pond cooling | | | 0.159 |
| Gas, average | 490 | 280 | 1.086 |
| Gas, natural gas combined cycle | 490 | 430 | |
| Gas, natural gas combined cycle, tower cooling | | | 0.776 |
| Gas, steam turbine, tower cooling | | | 3.127 |
| Gas, natural gas combined cycle, with carbon capture and storage | | 130 | |
| Gas, natural gas combined cycle, with carbon capture and storage, tower cooling | | | 1.488 |
| Gas, natural gas combined cycle, once-through cooling | | | 0.379 |
| Gas, steam turbine, once-through cooling | | | 0.909 |
| Gas, natural gas combined cycle, pond cooling | | | 0.909 |
| Gas, natural gas combined cycle, dry cooling | | | 0.008 |
| Nuclear, default | 12 | 5.1 | |
| Nuclear | 12 | 5.1 | |
| Nuclear, average | | | 1.957 |
| Nuclear, tower cooling | | | 2.544 |
| Nuclear, once-through cooling | | | 1.018 |
| Nuclear, pond cooling | | | 2.309 |
| Biopower, default | 230 | | |
| Biomass | 230 | | |
| Biopower, average | | | 1.147 |
| Biopower, steam turbine, tower cooling | | | 2.093 |
| Biopower, biogas, tower cooling | | | 0.89 |
| Biopower, steam turbine, once-through cooling | | | 1.136 |
| Biopower, steam turbine, pond cooling | | | 1.476 |
| Biopower, biogas, dry cooling | | | 0.132 |
| Geothermal, default | 38 | | |
| Geothermal | 38 | | |
| Geothermal, average | | | 0.95 |
| Geothermal, flash, tower cooling | | | 0.057 |
| Geothermal, flash, dry cooling | | | 0.019 |
| Geothermal, binary, dry cooling | | | 1.022 |
| Geothermal, enhanced geothermal system, dry cooling | | | 1.912 |
| Geothermal, binary, hybrid cooling | | | 1.745 |
| Hydropower, average | 24 | 81 | |
| Hydropower | 24 | | 17.0 |
| Hydropower, default | | | 17.0 |
| Hydropower, large | | 150 | |
| Hydropower, medium | | 11 | |
| Solar photovoltaic, average | 45 | 21 | |
| Solar photovoltaic, utility-scale | 48 | | |
| Solar photovoltaic, rooftop | 41 | | |
| Solar photovoltaic, polycrystalline silicon, ground-mounted | | 37 | |
| Solar photovoltaic, polycrystalline silicon, roof-mounted | | 37 | |
| Solar photovoltaic, cadmium telluride, ground-mounted | | 12 | |
| Solar photovoltaic, cadmium telluride, roof-mounted | | 15 | |
| Solar photovoltaic, copper indium gallium selenide, ground-mounted | | 11 | |
| Solar photovoltaic, copper indium gallium selenide, roof-mounted | | 14 | |
| Solar photovoltaic, default | | | 0.004 |
| Solar photovoltaic | | | 0.004 |
| Concentrated solar power, average | 27 | 32 | 1.567 |
| Concentrated solar power | 27 | | |
| Concentrated solar power, power tower | | 22 | |
| Concentrated solar power, parabolic trough | | 42 | |
| Concentrated solar power, parabolic trough, tower cooling | | | 3.43 |
| Concentrated solar power, power tower, tower cooling | | | 2.975 |
| Concentrated solar power, linear Fresnel, tower cooling | | | 3.785 |
| Concentrated solar power, parabolic trough, dry cooling | | | 0.295 |
| Concentrated solar power, power tower, dry cooling | | | 0.098 |
| Concentrated solar power, parabolic trough, hybrid cooling | | | 1.279 |
| Concentrated solar power, power tower, hybrid cooling | | | 0.644 |
| Concentrated solar power, Stirling engine | | | 0.019 |
| Wind, average | 12 | 13 | |
| Wind, onshore | 11 | 12 | |
| Wind, offshore | 12 | | |
| Wind, offshore, concrete foundation | | 14 | |
| Wind, offshore, steel foundation | | 13 | |
| Wind, default | | | 0 |
| Wind | | | 0 |
| Ocean, default | 17 | | |
| Ocean, tidal and wave | 17 | | |

[^macknick2012]: Macknick, J., et al. (2012). "Operational water consumption and withdrawal factors for electricity generating technologies: a review of existing literature." Environmental Research Letters 7(4): 045802.
[^ipcc2014]: IPCC (2014). Climate Change 2014: Mitigation of Climate Change. Working Group III Contribution to the Fifth Assessment Report.
[^unece2020]: UNECE (2020). Life Cycle Assessment of Electricity Generation Options. United Nations Economic Commission for Europe.
[^wikipedia]: Life-cycle greenhouse gas emissions of energy sources. https://en.wikipedia.org/wiki/Life-cycle_greenhouse_gas_emissions_of_energy_sources.
