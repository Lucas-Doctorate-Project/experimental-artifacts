# External carbon-intensity consistency check

This artifact checks whether the ENTSO-E-derived model produces credible national carbon-intensity levels and trends for Germany, France, and Poland. It is an external consistency check, not a claim that either reference series is ground truth.

## The validation checks

### 1. Ember benchmark: are the national levels and annual trends plausible?

**Why:** [Ember Yearly Electricity Data](https://files.ember-energy.org/public-downloads/yearly_full_release_long_format.csv) is the closest public match to the model: both use national, production-based lifecycle emissions.

**Calculation:** For 2018–2024, the artifact compares generation-weighted national means and the seven annual values. It reports the signed level difference and Pearson and Spearman correlations.

**Result:** The model preserves the PL > DE > FR ordering. Its annual values follow Ember closely (Pearson \(r=0.987\)–\(0.999\)), but its period means are lower by 16.9% for Germany, 34.0% for France, and 19.7% for Poland. This is the primary paper result; `out/paper_table.csv` contains its unrounded values.

### 2. EEA check: do the annual trends agree with a second source?

**Why:** [EEA ENER038](https://www.eea.europa.eu/en/analysis/indicators/greenhouse-gas-emission-intensity-of-1) provides an independent trend check, but it is not an equivalent level benchmark. EEA measures combustion-related emissions over gross generation and assigns zero to nuclear and renewables, whereas the model uses lifecycle factors and ENTSO-E generation.

**Calculation:** The artifact compares paired annual values for finalized 2018–2023 data and then repeats the check including EEA's 2024 early estimate. The pinned EEA data do not contain annual generation denominators, so EEA period means and period-level differences are deliberately left blank.

**Result:** The annual correlations remain high (approximately 0.975–1.000). This supports trend consistency under a different data source, but not absolute calibration.

### 3. Non-CCS sensitivity: how much does the factor profile explain?

**Why:** The study's UNECE averages for coal and gas include CCS cases, which lowers the resulting factors. The sensitivity profile removes that implicit CCS averaging without changing the generation data.

**Calculation:** The main changes are 630 to 933.33 gCO2e/kWh for hard coal and coal-derived gas, and 280 to 430 gCO2e/kWh for gas and its fossil proxies. `factors.csv` pins both profiles.

**Result:** Relative to Ember, the period differences change from -16.9% to -4.2% for Germany, from -34.0% to -14.0% for France, and from -19.7% to +2.2% for Poland. Factor selection therefore explains much of the original offset, but not all of France's gap. This is a sensitivity analysis, not a replacement calibration or a rerun of the scheduling experiments.

### 4. 2025 supplement: does one additional year change the conclusion?

**Why:** Extending the period checks whether the result depends on ending the analysis in 2024.

**Calculation:** The artifact repeats the Ember comparison over 2018–2025.

**Result:** The offsets and correlations change little. However, Ember's latest-year estimates may use monthly ENTSO-E data for France and Poland, so 2025 is less independent and is excluded from the paper result.

### 5. Fuel harmonization: is the gap mainly caused by emission factors?

**Why:** A difference in national intensity can come from different fuel factors or from different generation data. This check separates those explanations as far as the available categories allow.

**Calculation:** ENTSO-E sources are grouped into Ember's nine fuel categories. Ember's effective country-year factor for each fuel is then applied to the model's generation mix. The analysis covers 2018–2024; some provisional 2025 fuel rows have zero generation and zero emissions, making their effective factors undefined.

**Result:** After harmonization, the annual residual is about -5.3% to -0.4% for Germany and +2.4% to +5.2% for Poland, but remains -24.5% to -17.0% for France. Factor choice appears to explain most of the German and Polish offsets. The French residual may also involve category mapping, source coverage, within-category technologies, and net-versus-gross accounting; the decomposition cannot identify one unique cause.

### 6. Generation coverage: are the datasets accounting for the same amount of electricity?

**Why:** Even identical emission factors can produce different national results if the two datasets cover different generation totals.

**Calculation:** `generation_coverage_pct` divides included ENTSO-E generation by Ember's reported national generation.

**Result:** Coverage is approximately 85–88% for Germany, 91–92% for France, and 89–92% for Poland. This is an energy-total comparison, not missing calendar time: the modeled series still contains a complete 15-minute calendar. Coverage differences may reflect source scope and net-versus-gross accounting, and their effect on intensity depends on which generation is absent.

## Run the validation

From the `experimental-artifacts` repository root:

```sh
python intensities/validate_carbon.py --output-dir intensities/validation/out
```

The command runs offline from committed inputs and replaces the generated files in `intensities/validation/out/`:

- `country_year.csv`: annual model, Ember, and EEA values by country and factor profile;
- `summary.csv`: primary and supplementary agreement statistics;
- `decomposition.csv`: factor sensitivity, fuel harmonization, and generation coverage for 2018–2024;
- `trace_reproduction.csv`: checks against the tracked four-decimal carbon traces;
- `paper_table.csv`: unrounded values for the compact paper table.

Run the offline regression suite with:

```sh
python -m unittest discover -s intensities/validation/tests -v
```

## Inputs and provenance

The filtered reference snapshots are committed as:

- `reference/ember_2026_de_fr_pl.csv`;
- `reference/eea_ener038_de_fr_pl.csv`.

`reference/sources.json` records the source URLs, retrieval dates, methodology versions, licenses, transformations, qualifications, and SHA-256 checksums. The standard validation requires no network access.

## Aggregation

For country \(c\), year \(y\), and factor profile \(p\), modeled national intensity is total modeled emissions divided by total included generation:

\[
I_{c,y,p} =
\frac{
  \sum_{t \in \mathcal{T}_{c,y}} \sum_f G_{c,t,f} EF_{p,f}\,\Delta t
}{
  \sum_{t \in \mathcal{T}_{c,y}} \sum_f G_{c,t,f}\,\Delta t
}.
\]

Here, \(\mathcal{T}_{c,y}\) contains the complete 15-minute local-calendar timestamps for country \(c\) in year \(y\).

In the equation:

- \(G_{c,t,f}\) is the generation reported for country \(c\), timestamp \(t\), and ENTSO-E production source \(f\), measured as average power in MW;
- \(EF_{p,f}\) is the lifecycle carbon-emission factor assigned to source \(f\) under factor profile \(p\), in gCO2e/kWh;
- \(\Delta t\) is the duration represented by one sample: 0.25 hours for the 15-minute series. Thus, \(G_{c,t,f}\Delta t\) is the generated energy for that interval, with the common MW-to-kWh conversion cancelling between the numerator and denominator.

The 2018–2024 period means use the same total-emissions-over-total-generation construction across years. They are not arithmetic means of the 15-minute intensity samples; that temporal mean is a scheduler-facing signal descriptor, not the national benchmark statistic.

## Interpretation and limits

The comparison can support two limited observations: the model preserves the PL > DE > FR ordering, and its annual variation follows the external series closely. Under the documented study factors, the model retains a downward level offset.

The check does not validate:

- absolute calibration or 15-minute fluctuations;
- marginal or consumption-based emissions;
- imported electricity or avoided real-world emissions;
- water intensity.

Observed gaps may reflect emission factors, fuel-category mapping, incomplete generation coverage, and net-versus-gross accounting. The results should therefore be described as external consistency evidence rather than complete validation of the environmental model.
