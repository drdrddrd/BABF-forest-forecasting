#!/usr/bin/env python3
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --- Configuration ---
DATA_DIR = PROJECT_ROOT / "data"
SCENARIOS = ["RCP26", "RCP45", "RCP85"]
SIMULATIONS_BY_RCP = {
    "RCP26": [
        "CLMCOM-CCLM4_ECEARTH_EUR11_RCP26",
        "DMI-HIRHAM5_ECEARTH_EUR11_RCP26",
        "SMHI-RCA4_MPIESM_EUR11_RCP26",
    ],
    "RCP45": [
        "CLMCOM-CCLM4_ECEARTH_EUR11_RCP45",
        "DMI-HIRHAM5_ECEARTH_EUR11_RCP45",
        "SMHI-RCA4_MPIESM_EUR11_RCP45",
    ],
    "RCP85": [
        "CLMCOM-CCLM4_ECEARTH_EUR11_RCP85",
        "DMI-HIRHAM5_ECEARTH_EUR11_RCP85",
        "SMHI-RCA4_MPIESM_EUR11_RCP85",
    ],
}
RCP_WARMING_BY_2099 = {"RCP26": 1.0, "RCP45": 2.0, "RCP85": 4.0}

NUM_PLOTS = int(os.getenv("BABF_MOCK_NUM_PLOTS", "120"))
SEED = int(os.getenv("BABF_MOCK_SEED", "35"))

INVNR_TO_YEAR = {
    150: 1984,
    250: 1994,
    350: 2005,
    450: 2012,
    550: 2021,
    650: 2031,
    750: 2041,
    850: 2050,
    950: 2060,
    1050: 2069,
    1150: 2079,
    1250: 2089,
    1350: 2099,
}
HISTORICAL_INVNR_MAX = 550
CLIMATE_YEARS = range(1981, 2100)

CLIMATE_METRIC_COLS = [
    "dry_days_count",
    "frost_days_count",
    "gdd_sum",
    "pr_sum",
    "pr_variance",
    "tas_mean",
    "tas_variance",
    "tasmax_mean",
    "tasmax_variance",
    "tasmin_mean",
    "tasmin_variance",
]

NFI_COLUMNS = [
    "CLNR",
    "DATUMF",
    "INVNR",
    "BASFPH",
    "BEWIRTINT1",
    "ASPECT25",
    "SLOPE25",
    "PH",
    "Z25",
    "X",
    "Y",
    "NAISHSTKOMB",
    "HW",
    "SW",
    "HWSW_prop",
    "INVYR",
    "BASFPH_squared",
    "Time_Diff_years",
    "BASFPH_next_INVNR",
    "HWSW_prop_next_INVNR",
    "BASFPH_difference",
]
COORD_COLUMNS = NFI_COLUMNS + [
    "closest_CH2018_lon_4326",
    "closest_CH2018_lat_4326",
    "distance_to_CH2018_m",
]
FINAL_COLUMNS = NFI_COLUMNS + [
    "closest_CH2018_lon_4326",
    "closest_CH2018_lat_4326",
] + [f"mean_{col}" for col in CLIMATE_METRIC_COLS]
YEARLY_COLUMNS = [
    "CLNR",
    "simulation",
    "year",
    "closest_CH2018_lon_4326",
    "closest_CH2018_lat_4326",
] + CLIMATE_METRIC_COLS
ENSEMBLE_COLUMNS = [
    "CLNR",
    "year",
] + CLIMATE_METRIC_COLS + [
    "closest_CH2018_lon_4326",
    "closest_CH2018_lat_4326",
]


def _round_frame(df):
    float_cols = df.select_dtypes(include=["floating"]).columns
    df[float_cols] = df[float_cols].round(4)
    return df


def _make_site_data(rng):
    clnrs = np.arange(10001, 10001 + NUM_PLOTS)
    x = rng.uniform(485000, 830000, NUM_PLOTS)
    y = rng.uniform(75000, 295000, NUM_PLOTS)
    elevation = rng.normal(1150, 420, NUM_PLOTS).clip(320, 2450)

    lon = 5.95 + (x - 485000) / 345000 * 4.4 + rng.normal(0, 0.015, NUM_PLOTS)
    lat = 45.85 + (y - 75000) / 220000 * 2.15 + rng.normal(0, 0.015, NUM_PLOTS)

    return pd.DataFrame(
        {
            "CLNR": clnrs,
            "ASPECT25": rng.uniform(0, 360, NUM_PLOTS),
            "SLOPE25": rng.uniform(2, 48, NUM_PLOTS),
            "PH": rng.normal(6.2, 0.8, NUM_PLOTS).clip(3.4, 8.5),
            "Z25": elevation,
            "X": x,
            "Y": y,
            "BEWIRTINT1": rng.choice([1.0, 2.0, 3.0], NUM_PLOTS, p=[0.58, 0.14, 0.28]),
            "NAISHSTKOMB": rng.choice(np.arange(1, 11), NUM_PLOTS),
            "closest_CH2018_lon_4326": np.round(lon * 48) / 48,
            "closest_CH2018_lat_4326": np.round(lat * 48) / 48,
            "distance_to_CH2018_m": rng.uniform(35, 1450, NUM_PLOTS),
            "_base_basal": rng.normal(30, 10, NUM_PLOTS).clip(3, 82),
            "_growth_rate": rng.normal(1.4, 1.0, NUM_PLOTS),
            "_base_hwsw": rng.beta(2.2, 2.0, NUM_PLOTS),
        }
    )


def _make_date(year, rng, future):
    if future:
        return f"01.07.{year}"
    day = int(rng.integers(1, 28))
    month = int(rng.integers(4, 11))
    return f"{day:02d}.{month:02d}.{year}"


def _make_nfi_data(site_df, rng):
    rows = []
    invnrs = list(INVNR_TO_YEAR)
    historical_invnrs = [invnr for invnr in invnrs if invnr <= HISTORICAL_INVNR_MAX]

    for _, site in site_df.iterrows():
        plot_phase = (int(site["CLNR"]) % 17) / 3.0
        for invnr in invnrs:
            year = INVNR_TO_YEAR[invnr]
            future = invnr > HISTORICAL_INVNR_MAX

            if future:
                basfph = np.nan
                hwsw_prop = np.nan
                hw = np.nan
                sw = np.nan
                basfph_squared = np.nan
            else:
                hist_idx = historical_invnrs.index(invnr)
                management_effect = -0.8 * hist_idx if site["BEWIRTINT1"] == 1.0 else 0.35 * hist_idx
                basfph = (
                    site["_base_basal"]
                    + site["_growth_rate"] * hist_idx
                    + management_effect
                    + 2.2 * np.sin(hist_idx + plot_phase)
                    + rng.normal(0, 2.4)
                )
                basfph = float(np.clip(basfph, 0.5, 98.0))
                hwsw_prop = (
                    site["_base_hwsw"]
                    + 0.018 * hist_idx
                    - 0.00004 * (site["Z25"] - 1200)
                    + rng.normal(0, 0.035)
                )
                hwsw_prop = float(np.clip(hwsw_prop, 0.02, 0.98))
                hw = basfph * hwsw_prop
                sw = basfph * (1.0 - hwsw_prop)
                basfph_squared = basfph**2

            rows.append(
                {
                    "CLNR": int(site["CLNR"]),
                    "DATUMF": _make_date(year, rng, future),
                    "INVNR": invnr,
                    "BASFPH": basfph,
                    "BEWIRTINT1": float(site["BEWIRTINT1"]),
                    "ASPECT25": site["ASPECT25"],
                    "SLOPE25": site["SLOPE25"],
                    "PH": site["PH"],
                    "Z25": site["Z25"],
                    "X": site["X"],
                    "Y": site["Y"],
                    "NAISHSTKOMB": int(site["NAISHSTKOMB"]),
                    "HW": hw,
                    "SW": sw,
                    "HWSW_prop": hwsw_prop,
                    "INVYR": year,
                    "BASFPH_squared": basfph_squared,
                    "Time_Diff_years": 10.0,
                    "closest_CH2018_lon_4326": site["closest_CH2018_lon_4326"],
                    "closest_CH2018_lat_4326": site["closest_CH2018_lat_4326"],
                    "distance_to_CH2018_m": site["distance_to_CH2018_m"],
                }
            )

    df = pd.DataFrame(rows).sort_values(["CLNR", "INVNR"]).reset_index(drop=True)
    df["BASFPH_next_INVNR"] = df.groupby("CLNR")["BASFPH"].shift(-1)
    df["HWSW_prop_next_INVNR"] = df.groupby("CLNR")["HWSW_prop"].shift(-1)
    df["BASFPH_difference"] = df["BASFPH"] - df["BASFPH_next_INVNR"]
    return _round_frame(df)


def _climate_values(site, year, rcp, simulation_index, rng):
    years_since_1981 = year - 1981
    trend_fraction = years_since_1981 / (2099 - 1981)
    warming = RCP_WARMING_BY_2099[rcp] * trend_fraction
    elevation_km = (site["Z25"] - 1000) / 1000
    sim_offset = (simulation_index - 1) * 0.12
    cycle = np.sin((year - 1981) / 5.5 + int(site["CLNR"]) % 9)

    tas_mean = 8.7 - 4.6 * elevation_km + warming + sim_offset + rng.normal(0, 0.22)
    tasmax_mean = tas_mean + 4.2 + rng.normal(0, 0.12)
    tasmin_mean = tas_mean - 3.8 + rng.normal(0, 0.12)
    pr_sum = 980 + 260 * elevation_km + 35 * cycle - 28 * warming + rng.normal(0, 65)
    dry_days_count = 220 - 18 * elevation_km + 8 * warming - 9 * cycle + rng.normal(0, 7)
    frost_days_count = 95 + 42 * elevation_km - 5.5 * warming + rng.normal(0, 8)
    gdd_sum = 1650 - 560 * elevation_km + 120 * warming + 35 * cycle + rng.normal(0, 75)

    return {
        "dry_days_count": np.clip(dry_days_count, 30, 330),
        "frost_days_count": np.clip(frost_days_count, 0, 220),
        "gdd_sum": np.clip(gdd_sum, 120, 3600),
        "pr_sum": np.clip(pr_sum, 300, 2600),
        "pr_variance": np.clip(22 + 4 * cycle + rng.normal(0, 2.5), 3, 65),
        "tas_mean": tas_mean,
        "tas_variance": np.clip(47 + 3 * cycle + rng.normal(0, 2.5), 18, 90),
        "tasmax_mean": tasmax_mean,
        "tasmax_variance": np.clip(68 + 4 * cycle + rng.normal(0, 4), 25, 120),
        "tasmin_mean": tasmin_mean,
        "tasmin_variance": np.clip(39 + 3 * cycle + rng.normal(0, 3), 15, 95),
    }


def _make_yearly_climate(site_df, rng):
    rows = []
    site_records = site_df.to_dict("records")
    for rcp in SCENARIOS:
        for simulation_index, simulation in enumerate(SIMULATIONS_BY_RCP[rcp]):
            for year in CLIMATE_YEARS:
                for site in site_records:
                    row = {
                        "CLNR": int(site["CLNR"]),
                        "simulation": simulation,
                        "year": year,
                        "closest_CH2018_lon_4326": site["closest_CH2018_lon_4326"],
                        "closest_CH2018_lat_4326": site["closest_CH2018_lat_4326"],
                    }
                    row.update(_climate_values(site, year, rcp, simulation_index, rng))
                    rows.append(row)

    return _round_frame(pd.DataFrame(rows)[YEARLY_COLUMNS])


def _make_ensemble_files(yearly_df):
    ensemble_by_rcp = {}
    yearly_with_rcp = yearly_df.copy()
    yearly_with_rcp["rcp"] = yearly_with_rcp["simulation"].str.extract(r"(RCP\d{2})")
    agg_spec = {col: "mean" for col in CLIMATE_METRIC_COLS}
    agg_spec.update(
        {
            "closest_CH2018_lon_4326": "first",
            "closest_CH2018_lat_4326": "first",
        }
    )

    for rcp in SCENARIOS:
        ensemble_df = (
            yearly_with_rcp[yearly_with_rcp["rcp"] == rcp]
            .groupby(["CLNR", "year"], as_index=False)
            .agg(agg_spec)
        )
        ensemble_by_rcp[rcp] = _round_frame(ensemble_df[ENSEMBLE_COLUMNS])

    return ensemble_by_rcp


def _add_climate_to_nfi(nfi_with_coords, ensemble_df):
    climate_at_year = ensemble_df.rename(
        columns={"year": "INVYR", **{col: f"mean_{col}" for col in CLIMATE_METRIC_COLS}}
    )
    climate_cols = ["CLNR", "INVYR"] + [f"mean_{col}" for col in CLIMATE_METRIC_COLS]
    final_df = nfi_with_coords.drop(columns=["distance_to_CH2018_m"]).merge(
        climate_at_year[climate_cols],
        on=["CLNR", "INVYR"],
        how="left",
    )
    return _round_frame(final_df[FINAL_COLUMNS])


def _write_hyperparameters():
    payload = {
        "targets": {
            "BASFPH_next_INVNR": {
                "lasso": {"alpha": 0.05},
                "xgboost": {
                    "best_params": {
                        "n_estimators": 25,
                        "learning_rate": 0.08,
                        "max_depth": 3,
                        "min_child_weight": 1,
                        "subsample": 0.9,
                        "colsample_bytree": 0.9,
                        "gamma": 0.0,
                        "reg_lambda": 1.0,
                        "reg_alpha": 0.0,
                    }
                },
            },
            "HWSW_prop_next_INVNR": {
                "lasso": {"alpha": 0.005},
                "xgboost": {
                    "best_params": {
                        "n_estimators": 25,
                        "learning_rate": 0.08,
                        "max_depth": 3,
                        "min_child_weight": 1,
                        "subsample": 0.9,
                        "colsample_bytree": 0.9,
                        "gamma": 0.0,
                        "reg_lambda": 1.0,
                        "reg_alpha": 0.0,
                    }
                },
            },
        }
    }

    for config_name in ["nfi1", "nfi12", "nfi123"]:
        output_dir = DATA_DIR / "hyperparameters" / config_name
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "best_params.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def generate_mock_files():
    rng = np.random.default_rng(SEED)

    nfi_dir = DATA_DIR / "preprocessed" / "nfi"
    ch2018_dir = DATA_DIR / "preprocessed" / "ch2018"
    final_dir = DATA_DIR / "preprocessed" / "final_nfi_ch2018_merged"
    prediction_dir = DATA_DIR / "predictions"
    figure_dirs = [
        PROJECT_ROOT / "figures" / "model_eval",
        PROJECT_ROOT / "figures" / "management_sensitivity",
        PROJECT_ROOT / "figures" / "nfi_ch2018",
        PROJECT_ROOT / "figures" / "predictions",
    ]
    table_dirs = [
        PROJECT_ROOT / "tables" / "rolling_window_eval",
    ]

    for directory in [nfi_dir, ch2018_dir, final_dir, prediction_dir, *figure_dirs, *table_dirs]:
        directory.mkdir(parents=True, exist_ok=True)

    print("--- Generating mock BABF data ---")
    print(f"Data directory: {DATA_DIR}")
    print(f"Plots: {NUM_PLOTS}")

    site_df = _make_site_data(rng)
    nfi_with_coords = _make_nfi_data(site_df, rng)
    nfi_processed = nfi_with_coords[NFI_COLUMNS]
    nfi_coords = nfi_with_coords[COORD_COLUMNS]

    nfi_processed.to_csv(nfi_dir / "nfi_processed_data.csv", index=False, float_format="%.4f")
    nfi_coords.to_csv(
        DATA_DIR / "preprocessed" / "NFI_CH2018_closest_coords.csv",
        index=False,
        float_format="%.4f",
    )

    yearly_df = _make_yearly_climate(site_df, rng)
    yearly_df.to_csv(ch2018_dir / "CH2018_yearly_metrics.csv", index=False, float_format="%.4f")

    ensemble_by_rcp = _make_ensemble_files(yearly_df)
    for rcp, ensemble_df in ensemble_by_rcp.items():
        ensemble_path = ch2018_dir / f"CH2018_ensemble_mean_multiple_vars_{rcp}.csv"
        ensemble_df.to_csv(ensemble_path, index=False, float_format="%.4f")

        final_df = _add_climate_to_nfi(nfi_with_coords, ensemble_df)
        final_path = final_dir / f"NFI_with_Climate_Averages_{rcp}.csv"
        final_df.to_csv(final_path, index=False, float_format="%.4f")
        print(f"Saved {final_path.relative_to(PROJECT_ROOT)} ({len(final_df)} rows)")

    pr_mock = ensemble_by_rcp["RCP26"][
        ["year", "closest_CH2018_lat_4326", "closest_CH2018_lon_4326", "pr_sum"]
    ].rename(
        columns={
            "closest_CH2018_lat_4326": "lat",
            "closest_CH2018_lon_4326": "lon",
            "pr_sum": "yearly_precipitation_sum",
        }
    )
    pr_mock.drop_duplicates(["year", "lat", "lon"]).to_csv(
        ch2018_dir / "CH2018_ensemble_mean_pr_RCP26.csv",
        index=False,
        float_format="%.4f",
    )

    _write_hyperparameters()
    print("Saved mock NFI, CH2018, merged NFI/climate, and hyperparameter files.")
    print("Raw CH2018 NetCDF files are intentionally not generated; the Euler output CSV is.")


if __name__ == "__main__":
    generate_mock_files()
