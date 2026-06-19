import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from pathlib import Path
import json
import warnings
from joblib import Parallel, delayed
import traceback

from forecasting_helpers import (
    FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGETS, CLIP_RANGES,
    get_model, load_xgboost_hyperparameters,
)

# suppress FutureWarnings and chained assignment warnings
warnings.filterwarnings('ignore', category=FutureWarning) 
pd.options.mode.chained_assignment = None 

# --- Configuration ---

PROJECT_ROOT = Path(__file__).resolve().parents[1]


RCP_SCENARIOS = ["RCP26", "RCP45", "RCP85"]
N_SEEDS = 100
RANDOMSEED = 35
N_JOBS = 8

DATA_DIR = PROJECT_ROOT / "data"
base_dir = DATA_DIR / "preprocessed" / "final_nfi_ch2018_merged"
HYPERPARAMETERS_FILE = DATA_DIR / "hyperparameters" / "nfi123" / "best_params.json"


def process_single_seed(
    seed,
    train_data,
    prediction_data,
    final_feature_cols,
    original_data_for_saving,
    hyperparameters,
    output_dir,
    output_filename_template,
):
    """
    Worker function for a single seed run. Trains models and performs iterative prediction.
    Receives pre-processed dataframes and the original, unmodified dataframe for final saving.

    The iterative prediction is batched across plots: at every time step, all plots that
    still have a step to take are predicted in a single call. Plots are independent of each
    other (the only dependency is within a plot, where a year's prediction feeds the next
    year's BASFPH/HWSW_prop features), so this produces numerically identical results to a
    per-plot, per-row loop while drastically cutting the number of model.predict() calls.
    """
    try:
        X_train = train_data[final_feature_cols]
        y_train = train_data[TARGETS]

        # One independent model per target.
        model_basfph = get_model('BASFPH_next_INVNR', seed, hyperparameters)
        model_basfph.fit(X_train, y_train['BASFPH_next_INVNR'])

        model_hwsw = get_model('HWSW_prop_next_INVNR', seed, hyperparameters)
        model_hwsw.fit(X_train, y_train['HWSW_prop_next_INVNR'])

        # Feature matrix aligned row-for-row with prediction_data; BASFPH/HWSW_prop are
        # both feature columns, so updating them in-place feeds the next step.
        feat = prediction_data[final_feature_cols].to_numpy(dtype=np.float64).copy()
        col_basfph = final_feature_cols.index('BASFPH')
        col_hwsw = final_feature_cols.index('HWSW_prop')

        invnr_values = prediction_data['INVNR'].to_numpy()
        basfph_present = prediction_data['BASFPH'].notna().to_numpy()

        # For each plot, build the sequence of row positions from its valid start onward.
        plot_seqs = []
        max_steps = 0
        for positions in prediction_data.groupby('CLNR').indices.values():
            mask = (invnr_values[positions] == 550) & basfph_present[positions]
            if not mask.any():
                continue  # No valid starting point
            start = int(np.argmax(mask))  # first row where INVNR == 550 and BASFPH not NaN
            seq = positions[start:]
            if len(seq) < 2:
                continue  # Nothing to predict forward
            plot_seqs.append(seq)
            max_steps = max(max_steps, len(seq) - 1)

        for k in range(max_steps):
            cur_pos = [seq[k] for seq in plot_seqs if k + 1 < len(seq)]
            nxt_pos = [seq[k + 1] for seq in plot_seqs if k + 1 < len(seq)]
            cur_pos = np.asarray(cur_pos)
            nxt_pos = np.asarray(nxt_pos)

            X_step = feat[cur_pos]
            predicted_basfph = np.clip(model_basfph.predict(X_step, validate_features=False), *CLIP_RANGES['BASFPH_next_INVNR'])
            predicted_hwsw_prop = np.clip(model_hwsw.predict(X_step, validate_features=False), *CLIP_RANGES['HWSW_prop_next_INVNR'])

            feat[nxt_pos, col_basfph] = predicted_basfph
            feat[nxt_pos, col_hwsw] = predicted_hwsw_prop

        predicted_values_df = pd.DataFrame({
            'CLNR': prediction_data['CLNR'].to_numpy(),
            'INVYR': prediction_data['INVYR'].to_numpy(),
            'BASFPH': feat[:, col_basfph],
            'HWSW_prop': feat[:, col_hwsw],
        })

        # Swap the original BASFPH/HWSW_prop columns for the predicted ones.
        output_df = original_data_for_saving.drop(columns=['BASFPH', 'HWSW_prop'], errors='ignore')
        final_df = pd.merge(output_df, predicted_values_df, on=['CLNR', 'INVYR'], how='left')
        
        
        current_output_path = output_dir / output_filename_template.format(seed=seed)
        final_df.to_csv(current_output_path, index=False, float_format='%.4f')
        return True
    
    except Exception as e:
        print(f"Error in seed {seed}: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("--- Running Parallel Forest Prediction Script ---")
    hyperparameters = load_xgboost_hyperparameters(HYPERPARAMETERS_FILE)
    print(f"Loaded XGBoost hyperparameters from: {HYPERPARAMETERS_FILE}")

    master_rng = np.random.RandomState(RANDOMSEED)
    SEEDS = master_rng.randint(0, 100000, N_SEEDS)

    for RCP_SCENARIO in RCP_SCENARIOS:
        print(f"\n{'='*25} Processing {RCP_SCENARIO} {'='*25}")
        input_file = base_dir / f"NFI_with_Climate_Averages_{RCP_SCENARIO}.csv"
        output_dir = DATA_DIR / "predictions" / f"Predictions_{RCP_SCENARIO}"
        output_filename_template = f"NFI_{RCP_SCENARIO}_prediction_seed_{{seed}}.csv"
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Load and prepare data ---
        try:
            full_data = pd.read_csv(input_file)
            print(f"Data loaded successfully. Shape: {full_data.shape}")
        except FileNotFoundError:
            print(f"FATAL: The file was not found at {input_file}. Skipping.")
            continue

        # Keep rows where BASFPH <= 100 or is NaN (future rows to be predicted).
        full_data = full_data[(full_data['BASFPH'].isna()) | (full_data['BASFPH'] <= 100)].copy()
        full_data = full_data[(full_data['BASFPH_next_INVNR'].isna()) | (full_data['BASFPH_next_INVNR'] <= 100)].copy()

        # Keep an unprocessed copy for the final save.
        original_data_for_saving = full_data.copy()

        full_data_processed = pd.get_dummies(full_data, columns=CATEGORICAL_FEATURES, drop_first=True)
        print("One-hot encoding performed on the entire dataset.")

        ohe_features = [col for col in full_data_processed.columns if col.startswith(tuple(cat + '_' for cat in CATEGORICAL_FEATURES))]
        FINAL_FEATURE_COLS = NUMERIC_FEATURES + ohe_features
        print(f"Total features after one-hot encoding: {len(FINAL_FEATURE_COLS)}")

        train_data_processed = full_data_processed.dropna(subset=TARGETS).copy()
        train_data_processed.dropna(subset=FINAL_FEATURE_COLS, inplace=True)
        print(f"Prepared {len(train_data_processed)} historical records for training.")
        if train_data_processed.empty:
            print("FATAL: No complete historical records available for training. Skipping.")
            continue

        # Future rows (NaN targets) stay in and get filled by the iterative prediction.
        prediction_data_processed = full_data_processed.sort_values(by=['CLNR', 'INVYR'], inplace=False, ignore_index=True)

        # --- Run parallel predictions ---
        print(f"\nStarting parallel execution on {N_JOBS} cores for {N_SEEDS} seeds...")
        results = Parallel(n_jobs=N_JOBS, prefer="threads")(
            delayed(process_single_seed)(
                seed,
                train_data_processed,
                prediction_data_processed,
                FINAL_FEATURE_COLS,
                original_data_for_saving,
                hyperparameters,
                output_dir,
                output_filename_template,
            )
            for seed in tqdm(SEEDS, desc=f"{RCP_SCENARIO} Progress")
        )

        successful_runs = sum(results)
        print(f"\nTotal Successful Runs: {successful_runs} / {N_SEEDS}")
        if successful_runs < N_SEEDS:
            print("Some runs failed. Please check logs for errors.")
        print(f"Output files saved in: {output_dir}")

    print("\n--- Script Finished ---")
