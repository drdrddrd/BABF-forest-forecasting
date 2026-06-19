#!/usr/bin/env python3
import xarray as xr
import pandas as pd
from pathlib import Path
import re
import time
import os
from multiprocessing import Pool, cpu_count
import sys

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CH2018_RAW_DIR = DATA_DIR / "raw" / "ch2018"
NFI_COORDS_FILE = DATA_DIR / "preprocessed" / "NFI_CH2018_closest_coords.csv"
TAS_DIR = CH2018_RAW_DIR / "tas"
TASMIN_DIR = CH2018_RAW_DIR / "tasmin"
TASMAX_DIR = CH2018_RAW_DIR / "tasmax"
PR_DIR = CH2018_RAW_DIR / "pr"
OUTPUT_DIR = DATA_DIR / "preprocessed" / "ch2018"
OUTPUT_FILENAME = "CH2018_yearly_metrics.csv"

# --- Parameters ---
GDD_BASE_TEMP = 5.0
DRY_DAY_THRESHOLD = 1.0
FROST_DAY_THRESHOLD = 0.0

def get_unique_simulations(base_dir):
    """Parses filenames to find unique simulation identifiers."""
    pattern = re.compile(r"CH2018_[a-z]+_(.*?)_QMgrid")
    simulations = set()
    for f in base_dir.glob("*.nc"):
        match = pattern.search(f.name)
        if match:
            simulations.add(match.group(1))
    return sorted(list(simulations))

def process_coordinate(args_tuple):
    """
    Processes a single coordinate's data. This function expects
    pre-loaded xarray DataArrays and performs only the calculations.
    """
    lon, lat, simulation_name, tas_ts, tasmin_ts, tasmax_ts, pr_ts = args_tuple
    
    try:
        # Calculate yearly metrics
        tas_yearly_mean = tas_ts.resample(time='YS').mean()
        tas_yearly_var = tas_ts.resample(time='YS').var()
        tasmin_yearly_mean = tasmin_ts.resample(time='YS').mean()
        tasmin_yearly_var = tasmin_ts.resample(time='YS').var()
        tasmax_yearly_mean = tasmax_ts.resample(time='YS').mean()
        tasmax_yearly_var = tasmax_ts.resample(time='YS').var()
        pr_yearly_sum = pr_ts.resample(time='YS').sum()
        pr_yearly_var = pr_ts.resample(time='YS').var()

        # Calculate Growing Degree Days (GDD)
        daily_avg_temp = (tasmax_ts + tasmin_ts) / 2
        daily_gdd = (daily_avg_temp - GDD_BASE_TEMP).clip(min=0)
        gdd_yearly_sum = daily_gdd.resample(time='YS').sum()

        # Calculate count-based metrics
        dry_days_yearly_count = (pr_ts < DRY_DAY_THRESHOLD).resample(time='YS').sum()
        frost_days_yearly_count = (tasmin_ts < FROST_DAY_THRESHOLD).resample(time='YS').sum()

        # Combine into one Dataset.
        yearly_ds = xr.Dataset({
            'tas_mean': tas_yearly_mean, 'tas_variance': tas_yearly_var,
            'tasmin_mean': tasmin_yearly_mean, 'tasmin_variance': tasmin_yearly_var,
            'tasmax_mean': tasmax_yearly_mean, 'tasmax_variance': tasmax_yearly_var,
            'pr_sum': pr_yearly_sum, 'pr_variance': pr_yearly_var,
            'gdd_sum': gdd_yearly_sum, 'dry_days_count': dry_days_yearly_count,
            'frost_days_count': frost_days_yearly_count,
        })

        # To DataFrame, add metadata.
        df = yearly_ds.to_dataframe()
        df['year'] = df.index.year
        df = df.reset_index(drop=True)
        df['simulation'] = simulation_name
        df['lon'] = lon
        df['lat'] = lat

        return (lon, lat, simulation_name, df, None)
    except Exception as e:
        return (lon, lat, simulation_name, None, f"Calculation failed for lon={lon}, lat={lat}: {e}")

if __name__ == "__main__":
    start_time = time.time()
    
    try:
        num_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
    except KeyError:
        print("WARNING: SLURM_CPUS_PER_TASK not set. Using all available CPUs.")
        num_cpus = cpu_count()
    print(f"--- Starting Yearly Metrics with {num_cpus} parallel workers ---", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n1. Loading target coordinates from: {NFI_COORDS_FILE}", flush=True)
    coords_df = pd.read_csv(NFI_COORDS_FILE)
    # Process only unique grid points to avoid redundant calculations
    unique_coords_df = coords_df[['closest_CH2018_lon_4326', 'closest_CH2018_lat_4326']].drop_duplicates().reset_index(drop=True)
    print(f"Found {len(unique_coords_df)} unique CH2018 coordinates to process.", flush=True)


    print("\n2. Identifying unique climate simulations...", flush=True)
    simulations = get_unique_simulations(PR_DIR)
    print(f"Found {len(simulations)} simulations to process.", flush=True)

    all_results = []
    

    # Main loop: process one climate simulation at a time
    for sim_idx, sim_name in enumerate(simulations):
        sim_start_time = time.time()
        print(f"\n--- Processing Simulation {sim_idx + 1}/{len(simulations)}: {sim_name} ---", flush=True)
        
        # NetCDF files per variable for this simulation.
        simulation_files_map = {
            'tas': sorted(list(TAS_DIR.glob(f"*_{sim_name}_*.nc"))),
            'tasmin': sorted(list(TASMIN_DIR.glob(f"*_{sim_name}_*.nc"))),
            'tasmax': sorted(list(TASMAX_DIR.glob(f"*_{sim_name}_*.nc"))),
            'pr': sorted(list(PR_DIR.glob(f"*_{sim_name}_*.nc"))),
        }
        
        if not all(simulation_files_map.values()):
            print(f"  WARNING: Missing one or more variable files for {sim_name}. Skipping.", flush=True)
            continue
            
        # Load all simulation data into memory once.
        print("  Loading simulation data into memory (this may take a moment)...", flush=True)
        try:
            with xr.open_mfdataset(simulation_files_map['tas']) as ds:
                tas_all = ds['tas'].load()
            with xr.open_mfdataset(simulation_files_map['tasmin']) as ds:
                tasmin_all = ds['tasmin'].load()
            with xr.open_mfdataset(simulation_files_map['tasmax']) as ds:
                tasmax_all = ds['tasmax'].load()
            with xr.open_mfdataset(simulation_files_map['pr']) as ds:
                pr_all = ds['pr'].load()
        except Exception as e:
            print(f"  FATAL ERROR: Failed to load data for simulation {sim_name}. Error: {e}. Skipping.", flush=True)
            continue
        print("  Data loaded.", flush=True)

        # Prepare tasks by pre-selecting the data for each coordinate.
        tasks_for_pool = []
        for _, row in unique_coords_df.iterrows():
            lon = row['closest_CH2018_lon_4326']
            lat = row['closest_CH2018_lat_4326']
            tas_ts = tas_all.sel(lon=lon, lat=lat, method='nearest')
            tasmin_ts = tasmin_all.sel(lon=lon, lat=lat, method='nearest')
            tasmax_ts = tasmax_all.sel(lon=lon, lat=lat, method='nearest')
            pr_ts = pr_all.sel(lon=lon, lat=lat, method='nearest')
            tasks_for_pool.append((lon, lat, sim_name, tas_ts, tasmin_ts, tasmax_ts, pr_ts))
        
        print(f"  Dispatching {len(tasks_for_pool)} coordinate tasks to {num_cpus} cpus for calculation...", flush=True)
        with Pool(processes=num_cpus) as pool:
            results_for_this_sim = pool.map(process_coordinate, tasks_for_pool)
        
        successful_dfs = []
        for lon, lat, sim, df, error in results_for_this_sim:
            if error:
                print(f"  ERROR processing {sim}: {error}", flush=True)
            else:
                successful_dfs.append(df)
        
        num_expected = len(tasks_for_pool)
        num_successful = len(successful_dfs)
        print(f"\n  VERIFICATION for {sim_name}:")
        print(f"    - Expected coordinates: {num_expected}")
        print(f"    - Successfully processed: {num_successful}")
        if num_successful < num_expected:
            print(f"    - WARNING: {num_expected - num_successful} coordinates failed to process for this simulation.", flush=True)
        else:
            print(f"    - SUCCESS: All coordinates processed successfully.", flush=True)
        
        all_results.extend(successful_dfs)
        sim_end_time = time.time()
        print(f"  Finished simulation in {(sim_end_time - sim_start_time)/60:.2f} minutes.", flush=True)


    print("\n--- Aggregating all results ---", flush=True)
    if not all_results:
        print("FATAL: No results were successfully generated. Exiting.", flush=True)
        sys.exit(1)
    
    final_df = pd.concat(all_results, ignore_index=True)
    
    print("\n--- Merging with NFI identifiers ---", flush=True)
    # Match coord column names for the merge.
    final_df.rename(columns={'lon': 'closest_CH2018_lon_4326', 'lat': 'closest_CH2018_lat_4326'}, inplace=True)
    
    # Unique CLNR -> CH2018 coordinate mapping.
    nfi_identifiers = coords_df[['CLNR', 'closest_CH2018_lon_4326', 'closest_CH2018_lat_4326']].drop_duplicates()
    merged_df = pd.merge(nfi_identifiers, final_df, on=['closest_CH2018_lon_4326', 'closest_CH2018_lat_4326'])
    
    # Identifiers first, then sorted metrics.
    id_cols = ['CLNR', 'simulation', 'year', 'closest_CH2018_lon_4326', 'closest_CH2018_lat_4326']
    metric_cols = [col for col in merged_df.columns if col not in id_cols]
    final_ordered_df = merged_df[id_cols + sorted(metric_cols)]


    output_path = OUTPUT_DIR / OUTPUT_FILENAME
    print(f"\n5. Saving final merged data to: {output_path}", flush=True)
    final_ordered_df.to_csv(output_path, index=False, float_format='%.4f')

    end_time = time.time()
    print("\n--- Script Finished ---", flush=True)
    print(f"Total execution time: {(end_time - start_time) / 60:.2f} minutes.", flush=True)
