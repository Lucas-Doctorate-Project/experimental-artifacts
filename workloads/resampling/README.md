# Mustang user resampling

This directory contains the workload preparation used by the `greenfilling-resampled` campaign. It samples complete user histories from the fixed Mustang slack and stress windows. Context jobs remain unchanged.

| File or directory | Contents |
| --- | --- |
| `mustang_user_resampling.ipynb` | Classification, saved collection inspection, optional generation, and selection |
| `mustang_user_resampling_analysis.py` | Sampling, validation, and plotting functions |
| `artifacts/` | The existing collection of 42 CSV/JSON workload pairs and its provenance |
| `figures/` | The saved classification and source-workload figures |

The selected campaign inputs remain in `../resampled/`:

- Slack: `mustang_slack_heavy_4x_seed20260930`
- Stress: `mustang_stress_heavy_8x_seed20260930`

Open the notebook from this directory or another directory inside the repository. Python needs NumPy, pandas, Matplotlib, IPython, and Plotly for the interactive figures. The source trace is `../sources/mustang.csv`.

The notebook loads the saved collection by default. `GENERATE_WORKLOADS = True` enables generation; use a new `ARTIFACT_DIR` when changing the parameters. `EXPORT_SELECTED_WORKLOADS = True` enables selection export to `../resampled/`. Exporting an existing selection replaces its five files.


The resampled campaign changes both job composition and offered load. Slack and stress identify the source windows. The campaign uses one selected seed per regime, so it does not measure variation across seeds. See the root README for the campaign results and their accounting limits.
