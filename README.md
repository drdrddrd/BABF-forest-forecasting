# BABF Forecasting

This repository contains the code used to forecast changes in Swiss forest
basal area and broadleaf fraction until 2099. The models are trained with data
from the Swiss National Forest Inventory (NFI) and the CH2018 climate scenarios.

The repository includes the preprocessing notebooks, model training and
prediction scripts, plotting notebooks, and a mock data generator. The original
NFI data are confidential and are not included. Generated data, predictions,
tables, and figures are also not tracked by Git.

The associated paper is published open access in Ecological Modelling:
[Machine learning for modeling forest dynamics using National Forest Inventory and climate data](https://doi.org/10.1016/j.ecolmodel.2026.111763).

## Environment

The project uses Conda. Create and activate the environment from the repository
root:

```bash
conda env create -f BABF_env.yml
conda activate BABF_env
```

Run scripts from the repository root unless a notebook path is explicitly
relative to `notebooks/` or `notebooks/sensitivity_analysis/`.

## Data Sources

The full workflow requires external data that is not included in the repository.

1. Swiss National Forest Inventory (NFI)

   The NFI data are confidential and subject to a written data usage agreement.
   They are not publicly downloadable and must not be redistributed. Data access
   is handled through the Swiss Federal Institute for Forest, Snow and Landscape
   Research (WSL). See the [NFI data provision page](https://www.lfi.ch/en/services/data-supply).

2. Swiss Climate Change Scenarios (CH2018)

   The CH2018 daily climate time-series data used in this study are available
   upon request from NCCS/MeteoSwiss. See the [CH2018 datasets page](https://www.nccs.admin.ch/nccs/en/home/climate-change-and-impacts/swiss-climate-change-scenarios/ch2018---climate-scenarios-for-switzerland.html).

   The NCCS page now notes that the newer Climate CH2025 scenarios have been
   published. This repository documents the CH2018-based workflow in order to
   reproduce the analysis used in this study.

3. swisstopo map data

   The plotting notebooks use swisstopo basemap and boundary files. Relevant
   products include [Overview Maps](https://www.swisstopo.admin.ch/de/ubersichtskarten-der-schweiz)
   and [swissBOUNDARIES3D](https://www.swisstopo.admin.ch/en/landscape-model-swissboundaries3d).

## Mock Data

Access to the original NFI data is restricted. To allow the code to be checked
without these files, `scripts/create_mock_data.py` creates a small random
dataset with the filenames and column schemas expected by the workflow.

```bash
conda activate BABF_env
python scripts/create_mock_data.py
```

By default the generator writes files under `data/`, including:

- `data/preprocessed/nfi/nfi_processed_data.csv`
- `data/preprocessed/NFI_CH2018_closest_coords.csv`
- `data/preprocessed/ch2018/CH2018_yearly_metrics.csv`
- `data/preprocessed/ch2018/CH2018_ensemble_mean_pr_RCP26.csv`
- `data/preprocessed/ch2018/CH2018_ensemble_mean_multiple_vars_RCP26.csv`
- `data/preprocessed/ch2018/CH2018_ensemble_mean_multiple_vars_RCP45.csv`
- `data/preprocessed/ch2018/CH2018_ensemble_mean_multiple_vars_RCP85.csv`
- `data/preprocessed/final_nfi_ch2018_merged/NFI_with_Climate_Averages_RCP26.csv`
- `data/preprocessed/final_nfi_ch2018_merged/NFI_with_Climate_Averages_RCP45.csv`
- `data/preprocessed/final_nfi_ch2018_merged/NFI_with_Climate_Averages_RCP85.csv`
- `data/hyperparameters/nfi1/best_params.json`
- `data/hyperparameters/nfi12/best_params.json`
- `data/hyperparameters/nfi123/best_params.json`

The raw CH2018 NetCDF files used by `scripts/euler_CH2018_processing.py` are not
generated. Instead, the mock generator creates the downstream yearly-metrics
CSV used by the later steps.

The generator also creates output directories for predictions, figures, and
tables. You can control the mock data size and random seed with:

```bash
BABF_MOCK_NUM_PLOTS=120 BABF_MOCK_SEED=35 python scripts/create_mock_data.py
```

The mock data can be used with the prediction script, the forward
hyperparameter tuning script, the prediction plotting notebook, and the
sensitivity-analysis notebooks. Map overlays that depend on real swisstopo files
may be skipped or unavailable when only mock data is present.

## Repository Structure

The code expects the following directory structure relative to the repository
root.

```text
.
├── BABF_env.yml
├── README.md
├── configs
│   └── published_model
│       └── best_params.json
├── data
│   ├── hyperparameters
│   ├── predictions
│   │   ├── Predictions_RCP26
│   │   ├── Predictions_RCP45
│   │   └── Predictions_RCP85
│   ├── preprocessed
│   │   ├── ch2018
│   │   ├── final_nfi_ch2018_merged
│   │   └── nfi
│   └── raw
│       ├── ch2018
│       ├── map_data
│       └── nfi
├── figures
│   ├── management_sensitivity
│   ├── model_eval
│   ├── nfi_ch2018
│   ├── predictions
│   └── rolling_window_eval
├── notebooks
│   └── sensitivity_analysis
├── model_weights
├── scripts
└── tables
    └── rolling_window_eval
```

## Workflow

### Workflow With Mock Data

To check the workflow without restricted data, first create the mock files:

```bash
conda activate BABF_env
python scripts/create_mock_data.py
```

The mock generator creates the preprocessed NFI files, CH2018 summary files,
merged NFI-CH2018 files, and placeholder hyperparameter files that are normally
produced in steps 01--05 below. After running it, the workflow can therefore be
started at step 06 for model evaluation, or at step 07 if only the future
predictions are needed. Step 05 can still be run with the mock data if the
hyperparameter tuning code itself should be checked.

Some map outputs in step 04 require the original swisstopo files and are not
fully covered by the mock data.

### Workflow With Original Data

Run the full workflow from the repository root after placing the required raw
data in the expected `data/raw/` locations.

1. `notebooks/NFI_preprocessing_until_2099.ipynb`
   Preprocess raw NFI files and extend the NFI template through future inventory
   years.
2. `scripts/euler_CH2018_processing.py`
   Compute yearly climate metrics from raw CH2018 NetCDF files for the selected
   grid coordinates. This step was designed for the Euler cluster and uses
   `SLURM_CPUS_PER_TASK` when available.
3. `notebooks/CH2018_preprocessing.ipynb`
   Build CH2018 ensemble summaries and merge climate metrics with the NFI data
   for each RCP scenario.
4. `notebooks/NFI_CH2018_plots.ipynb`
   Plot NFI locations and CH2018 climate summaries. The map plot requires real
   swisstopo raster and boundary files.
5. `scripts/hyperparameter_tuning_forward.py`
   Tune forward-validation hyperparameters for `nfi1`, `nfi12`, and `nfi123`;
   write `data/hyperparameters/<config>/best_params.json`.
6. `notebooks/rolling_window_evaluation.ipynb`
   Train and evaluate Lasso, XGBoost, and persistence baselines with
   rolling-window or forward-style evaluation.
7. `scripts/prediction_until_2099.py`
   Train XGBoost models using `configs/published_model/best_params.json` and
   produce iterative predictions through 2099 for RCP26, RCP45, and RCP85.
8. `notebooks/prediction_plots.ipynb`
   Aggregate seed-level prediction CSVs and save final prediction figures.

### Optional Sensitivity Analyses

The sensitivity notebooks provide additional analyses:

- `notebooks/sensitivity_analysis/feature_coverage_analysis.ipynb`
- `notebooks/sensitivity_analysis/leakage_free_evaluation.ipynb`
- `notebooks/sensitivity_analysis/management_sensitivity_analysis.ipynb`
- `notebooks/sensitivity_analysis/rcp_sensitivity_analysis.ipynb`
- `notebooks/sensitivity_analysis/spatial_block_bootstrap_rmse_model_comparison.ipynb`

The spatial block bootstrap notebook saves CSV summaries under
`tables/rolling_window_eval/`. The management sensitivity notebook saves plots
and summaries under `figures/management_sensitivity/`.

## Outputs

The main generated outputs are:

- Prediction CSVs in `data/predictions/Predictions_RCP26`,
  `data/predictions/Predictions_RCP45`, and
  `data/predictions/Predictions_RCP85`
- Rolling-window evaluation figures in `figures/rolling_window_eval/`
- NFI and CH2018 figures in `figures/nfi_ch2018/`
- Prediction figures in `figures/predictions/`
- Management sensitivity outputs in `figures/management_sensitivity/`
- Sensitivity-analysis tables in `tables/rolling_window_eval/`

These outputs are generated locally and are not tracked by Git. The `.gitignore`
file excludes `data/`, `*.csv`, `*.png`, and `*.pdf`.

## Published Models

The parameters for reproducing the published projections (Figures 5–7) are in
[`configs/published_model/best_params.json`](configs/published_model/best_params.json).
Run the commands below from the repository root with `BABF_env` activated and
the NFI–CH2018 input CSVs prepared as described above. Reproducing the paper
requires the original data.

### Forecast Through 2099

This command trains models with the published parameters and forecasts all
three RCP scenarios through 2099:

```bash
python scripts/prediction_until_2099.py
```

Results are saved in `data/predictions/Predictions_<RCP>/`.

### Predict with Saved Weights

Download the
[weights archive](https://github.com/drdrddrd/BABF-forest-forecasting/releases/download/model-weights-v1.0.0/published_projection_ensemble_weights.tar.xz)
into `model_weights/`. Then verify and extract it:

```bash
(cd model_weights && shasum -a 256 -c published_projection_ensemble_weights.tar.xz.sha256)
tar -xJf model_weights/published_projection_ensemble_weights.tar.xz -C model_weights
```

The example below loads saved models for one seed and predicts one inventory
step for three plots under RCP85, without training.

<details>
<summary>Python example</summary>

```python
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

weights = Path("model_weights/published_projection_ensemble")
manifest = json.loads((weights / "manifest.json").read_text())
scenario = "RCP85"
metadata = manifest["scenario_training"][scenario]
data = pd.read_csv(
    f"data/preprocessed/final_nfi_ch2018_merged/NFI_with_Climate_Averages_{scenario}.csv"
)
rows = data.loc[data["INVNR"].eq(550)].head(3).copy()

# Retain the fitted category levels, even when using only a subset of plots.
for column, levels in metadata["categories"].items():
    if not rows[column].dropna().isin(levels).all():
        raise ValueError(f"Unknown category in {column}")
    rows[column] = pd.Categorical(rows[column], categories=levels)
X = pd.get_dummies(rows, columns=list(metadata["categories"]), drop_first=True)
X = X[metadata["feature_names_in_order"]]

predictions = {}
for target, filename, upper in [
    ("BASFPH", "basal_area.json", 100),
    ("HWSW_prop", "broadleaf_fraction.json", 1),
]:
    model = xgb.Booster()
    model.load_model(weights / f"models/{scenario}/seed_71369/{filename}")
    predictions[target] = np.clip(model.inplace_predict(X), 0, upper)

print(pd.DataFrame(predictions, index=rows.index))
```

</details>

To forecast through 2099 with saved weights, repeat the prediction step using
the updated basal area, broadleaf fraction, and next interval's climate inputs.
Use all 100 seeds for ensemble results.

## Citations

When using this repository, cite the associated paper:

- Rieder, Dea; Costa, Maximiliano; Temperli, Christian; Giardina, Francesco; and
  Vaccario, Giacomo (2026). Machine learning for modeling forest dynamics using
  National Forest Inventory and climate data. Ecological Modelling, 521, 111763.
  https://doi.org/10.1016/j.ecolmodel.2026.111763

When using the full workflow, also cite the underlying datasets according to
their data-provider requirements. At minimum, this project depends on:

- NFI: WSL, 2025: Schweizerisches Landesforstinventar LFI. Daten der
  Erhebung(en) 1993-95 / 2004-06 / 2009-2017 / 2018-2024. Christian Temperli.
  Eidg. Forschungsanstalt, Birmensdorf.
- CH2018: CH2018 Project Team (2018): CH2018 - Climate Scenarios for
  Switzerland. National Centre for Climate Services.
  doi: `10.18751/Climate/Scenarios/CH2018/1.0`.
- swisstopo Overview Maps: Federal Office of Topography (swisstopo). Overview
  maps of Switzerland, 2024. Wabern, Switzerland.
  https://www.swisstopo.admin.ch/de/ubersichtskarten-der-schweiz
- swisstopo swissBOUNDARIES3D: Federal Office of Topography (swisstopo).
  swissBOUNDARIES3D: Administrative boundaries of Switzerland and Liechtenstein,
  2024. Wabern, Switzerland.
  https://www.swisstopo.admin.ch/en/landscape-model-swissboundaries3d

## Author

Dea Rieder, ETH Zurich, 2026.
