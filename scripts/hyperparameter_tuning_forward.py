import json
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from pathlib import Path

from sklearn.linear_model import lasso_path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from forecasting_helpers import (
    FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGETS, RANDOM_STATE,
)

# --- Configuration ---

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FILE_PATH = DATA_DIR / "preprocessed" / "final_nfi_ch2018_merged" / "NFI_with_Climate_Averages_RCP45.csv"
HYPERPARAMETERS_DIR = DATA_DIR / "hyperparameters"

STRATA_COL = 'INVNR'

# Forward (walk-forward) configurations: name → (train INVNRs, test INVNRs)
#   nfi1   train -> nfi2  test
#   nfi12  train -> nfi3  test
#   nfi123 train -> nfi4  test
# (INVNR 150..450 = NFI1..4)
FORWARD_CONFIGS = {
    'nfi1':   {'train': [150],            'test': [250]},
    'nfi12':  {'train': [150, 250],       'test': [350]},
    'nfi123': {'train': [150, 250, 350],  'test': [450]},
}

# LassoCV settings
ALPHA_NUM_STEPS = 10000
ALPHA_LOGSPACE_START = -6
ALPHA_LOGSPACE_END = 2
LASSO_MAX_ITER = 1000

# Optuna settings
N_TRIALS_OPTUNA = 300


# --- Data loading and preprocessing ---

def load_and_preprocess_forward(filepath, train_invnrs, test_invnrs):
    """
    Loads data, filters to the train + test INVNR values, drops NAs, and one-hot
    encodes the combined frame (so train/test share identical dummy columns), then
    splits into a forward train/test pair based on INVNR.
    Returns X_train, y_train, X_test, y_test.
    """
    keep_invnrs = list(train_invnrs) + list(test_invnrs)
    print(f"\nLoading data from {filepath} (train INVNRs: {train_invnrs}, test INVNRs: {test_invnrs}) ...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None, None, None, None

    required_cols = FEATURES + TARGETS + [STRATA_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns not found: {missing_cols}")

    initial_rows = len(df)
    print(f"Initial rows: {initial_rows}")

    df = df[(df['BASFPH'] <= 100) & (df['BASFPH_next_INVNR'] <= 100)].copy()
    df = df[df['INVNR'].isin(keep_invnrs)].copy()

    columns_to_check = FEATURES + TARGETS
    df_clean = df.dropna(subset=columns_to_check).copy()
    print(f"Rows after filtering and dropping NAs: {len(df_clean)} ({initial_rows - len(df_clean)} removed)")

    df_processed = pd.get_dummies(df_clean, columns=CATEGORICAL_FEATURES, drop_first=True)

    ohe_features = [col for col in df_processed.columns if col.startswith(tuple(c + '_' for c in CATEGORICAL_FEATURES))]
    final_features = NUMERIC_FEATURES + ohe_features

    train_mask = df_processed[STRATA_COL].isin(train_invnrs)
    test_mask = df_processed[STRATA_COL].isin(test_invnrs)

    X_train = df_processed.loc[train_mask, final_features]
    y_train = df_processed.loc[train_mask, TARGETS]
    X_test = df_processed.loc[test_mask, final_features]
    y_test = df_processed.loc[test_mask, TARGETS]

    print(f"Train feature matrix shape: {X_train.shape}")
    print(f"Test feature matrix shape: {X_test.shape}")
    return X_train, y_train, X_test, y_test


# --- Model tuning (forward / walk-forward split) ---

def tune_lasso_forward(X_train, y_train_series, X_test, y_test_series):
    """
    Selects the Lasso alpha that minimises RMSE on the forward (held-out) test
    inventory rather than on random CV folds.
    """
    print(f"\n--- Tuning Lasso (forward) for target: {y_train_series.name} ---")
    alphas = np.logspace(ALPHA_LOGSPACE_START, ALPHA_LOGSPACE_END, ALPHA_NUM_STEPS)

    # Scale once on the train inventories.
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Centre the target so the centred-X path is equivalent to fitting an intercept;
    # the intercept then equals the train-target mean.
    y_train_mean = y_train_series.to_numpy().mean()
    y_train_c = y_train_series.to_numpy() - y_train_mean

    # Compute the whole regularisation path in one coordinate-descent sweep
    # (warm-started across alphas) instead of refitting per alpha.
    alphas_out, coefs, _ = lasso_path(
        X_train_s, y_train_c, alphas=alphas, max_iter=LASSO_MAX_ITER
    )

    # preds: (n_test_samples, n_alphas); rmse per alpha across the test inventory.
    preds = X_test_s @ coefs + y_train_mean
    rmses = np.sqrt(((preds - y_test_series.to_numpy()[:, None]) ** 2).mean(axis=0))

    best_idx = int(np.argmin(rmses))
    best_alpha = float(alphas_out[best_idx])
    print(f"Best alpha: {best_alpha:.6f} (test RMSE: {rmses[best_idx]:.4f})")
    return best_alpha


def tune_xgboost_optuna_forward(X_train, y_train_series, X_test, y_test_series, target_name):
    """
    Tunes XGBoost with Optuna, minimising RMSE on the forward (held-out) test
    inventory rather than on random CV folds.
    """
    print(f"\n--- Tuning XGBoost with Optuna (forward) for target: {target_name} ---")

    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'reg_alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train_series)
        preds = model.predict(X_test)
        return np.sqrt(mean_squared_error(y_test_series, preds))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS_OPTUNA, show_progress_bar=True)
    print(f"Best RMSE: {study.best_value:.4f}")
    return study.best_params


# --- Save results ---

def save_results_to_json(results_dict, output_dir):
    """
    Saves tuning results to best_params.json matching the format used in
    data/hyperparameters/<config>/best_params.json.
    """
    payload = {"targets": {}}
    for target, models in results_dict.items():
        payload["targets"][target] = {
            "lasso": {"alpha": models["lasso_alpha"]},
            "xgboost": {"best_params": models["xgboost_params"]},
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "best_params.json"
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved hyperparameters to: {output_path}")


# --- Main ---

def run_tuning_for_config(config_name, train_invnrs, test_invnrs):
    print("\n" + "="*60)
    print(f"FORWARD CONFIG: {config_name}  (train: {train_invnrs} -> test: {test_invnrs})")
    print("="*60)

    X_train, y_train, X_test, y_test = load_and_preprocess_forward(
        FILE_PATH, train_invnrs, test_invnrs
    )
    if X_train is None:
        return

    all_best_params = {}

    for target in TARGETS:
        print("\n" + "-"*50)
        print(f"Target: {target}")
        print("-"*50)

        y_train_series = y_train[target]
        y_test_series = y_test[target]
        all_best_params[target] = {}

        all_best_params[target]['lasso_alpha'] = tune_lasso_forward(
            X_train, y_train_series, X_test, y_test_series
        )
        all_best_params[target]['xgboost_params'] = tune_xgboost_optuna_forward(
            X_train, y_train_series, X_test, y_test_series, target
        )

    output_dir = HYPERPARAMETERS_DIR / config_name
    save_results_to_json(all_best_params, output_dir)


def main():
    for config_name, cfg in FORWARD_CONFIGS.items():
        run_tuning_for_config(config_name, cfg['train'], cfg['test'])


if __name__ == '__main__':
    main()
