"""Shared forecasting helpers for the BABF rolling-window evaluation project.

This module is the single home for the constants and modelling logic that the
notebooks and scripts previously re-implemented from scratch. The canonical
versions live here; `notebooks/rolling_window_evaluation.ipynb` is the source of
truth for this logic and only keeps its presentation (table / plot) code locally.

Import patterns
---------------
- Notebooks (run from their own directory) add the repo root to ``sys.path`` and
  import via the package path, e.g.::

      import sys; sys.path.append('..')          # '../..' from sensitivity_analysis/
      from scripts.forecasting_helpers import load_and_clean, train_scope

- Scripts in this directory run standalone, so they import directly::

      from forecasting_helpers import FEATURES, get_model
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --- Constants ---

# Match the deployment / iterative-prediction seed for reproducibility.
RANDOM_STATE = 35

FEATURES = [
    'mean_dry_days_count', 'mean_frost_days_count', 'mean_gdd_sum',
    'mean_pr_sum', 'mean_pr_variance', 'mean_tas_mean', 'mean_tas_variance',
    'mean_tasmax_mean', 'mean_tasmax_variance', 'mean_tasmin_mean',
    'mean_tasmin_variance', 'HWSW_prop',
    'Time_Diff_years', 'BASFPH', 'BEWIRTINT1', 'ASPECT25',
    'SLOPE25', 'PH', 'Z25', 'NAISHSTKOMB',
]
CATEGORICAL_FEATURES = ['BEWIRTINT1', 'NAISHSTKOMB']
NUMERIC_FEATURES = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]
TARGETS = ['BASFPH_next_INVNR', 'HWSW_prop_next_INVNR']
GROUPING_COLUMNS = ['BEWIRTINT1', 'NAISHSTKOMB']

# Rolling-window train/test inventory rounds (INVNR 150..450 = NFI1..NFI4).
NFI12_TRAIN_INVNRS = [150, 250]
NFI123_TRAIN_INVNRS = [150, 250, 350]
NFI12_TEST_INVNR = 350    # NFI12 one-step test: 350 -> 450 (NFI4)
NFI123_TEST_INVNR = 450   # NFI123 one-step test: 450 -> 550 (NFI5)

CLIP_RANGES = {
    'BASFPH_next_INVNR': (0, 100),
    'HWSW_prop_next_INVNR': (0, 1),
}
PERSISTENCE_MAP = {
    'BASFPH_next_INVNR': 'BASFPH',
    'HWSW_prop_next_INVNR': 'HWSW_prop',
}


# --- Hyperparameter loading ---

def load_best_hyperparams(filepath):
    """Load Lasso + XGBoost best params per target from a ``best_params.json``."""
    with Path(filepath).open('r', encoding='utf-8') as f:
        payload = json.load(f)
    hyperparams = {}
    for target_name, target_payload in payload['targets'].items():
        hyperparams[target_name] = {
            'Lasso':   {'alpha': target_payload['lasso']['alpha']},
            'XGBoost': target_payload['xgboost']['best_params'].copy(),
        }
    return hyperparams


def load_xgboost_hyperparameters(filepath):
    """Load only the tuned XGBoost params per target: ``{target: best_params}``."""
    with Path(filepath).open('r', encoding='utf-8') as f:
        payload = json.load(f)
    return {
        target_name: target_payload['xgboost']['best_params']
        for target_name, target_payload in payload['targets'].items()
    }


# --- Data loading & encoding ---

def load_and_clean(filepath, verbose=False):
    """Load the merged NFI/CH2018 CSV and apply the rolling-window cleaning.

    Filters BASFPH/BASFPH_next_INVNR to <= 100, keeps INVNR in {150, 250, 350,
    450} (NFI1..NFI4) and drops rows with NaNs in FEATURES + TARGETS.
    """
    df = pd.read_csv(filepath)
    if verbose:
        print('Loaded', len(df), 'rows.')
    df = df[df['BASFPH'] <= 100].copy()
    df = df[df['BASFPH_next_INVNR'] <= 100].copy()
    df = df[df['INVNR'].isin([150, 250, 350, 450])].copy()
    df = df.dropna(subset=FEATURES + TARGETS).copy()
    if verbose:
        print('After cleaning:', len(df), 'rows | unique CLNR:', df['CLNR'].nunique())
        for inv in [150, 250, 350, 450]:
            print('   INVNR', inv, ':', int((df['INVNR'] == inv).sum()), 'rows')
    return df


def lock_categories(train_df):
    """Capture the observed categories per categorical feature on the train slice."""
    return {col: pd.Series(train_df[col]).dropna().unique() for col in CATEGORICAL_FEATURES}


def encode_features(df_slice, cats_map, final_features):
    """One-hot encode a slice using locked train categories, aligned to final_features."""
    df2 = df_slice.copy()
    for col in CATEGORICAL_FEATURES:
        df2[col] = pd.Categorical(df2[col], categories=cats_map[col])
    ohe = pd.get_dummies(df2, columns=CATEGORICAL_FEATURES, drop_first=True)
    return ohe.reindex(columns=final_features, fill_value=0)


def build_model_inputs(df, train_invnrs):
    """Lock categories on the train slice and assemble X_train / y_train + metadata."""
    train_df = df[df['INVNR'].isin(train_invnrs)].copy()
    cats_map = lock_categories(train_df)
    tmp = train_df.copy()
    for col in CATEGORICAL_FEATURES:
        tmp[col] = pd.Categorical(tmp[col], categories=cats_map[col])
    train_ohe = pd.get_dummies(tmp, columns=CATEGORICAL_FEATURES, drop_first=True)
    prefixes = tuple(c + '_' for c in CATEGORICAL_FEATURES)
    ohe_features = [c for c in train_ohe.columns if c.startswith(prefixes)]
    final_features = NUMERIC_FEATURES + ohe_features
    return {
        'train_df': train_df,
        'X_train': train_ohe[final_features],
        'y_train': train_df[TARGETS].copy(),
        'cats_map': cats_map,
        'final_features': final_features,
    }


# --- Models ---

def make_models(hyperparams, target):
    """Build a Lasso pipeline and a seeded XGBoost regressor for one target."""
    return {
        'Lasso': Pipeline([
            ('scaler', StandardScaler()),
            ('lasso', Lasso(alpha=hyperparams[target]['Lasso']['alpha'], max_iter=10000)),
        ]),
        'XGBoost': xgb.XGBRegressor(
            **hyperparams[target]['XGBoost'],
            objective='reg:squarederror', random_state=RANDOM_STATE, n_jobs=-1,
            importance_type='gain',
        ),
    }


def get_model(target_name, seed, hyperparameters):
    """Initialise a single-threaded, seeded XGBoost model for iterative prediction.

    ``hyperparameters`` is the XGBoost-only mapping from
    :func:`load_xgboost_hyperparameters`.
    """
    params = hyperparameters[target_name].copy()
    params['random_state'] = seed
    params['n_jobs'] = 1  # Each model runs single-threaded inside a parallel job.
    return xgb.XGBRegressor(**params)


def train_scope(scope, hyperparams):
    """Fit one Lasso and one XGBoost per target; return (models, importances).

    ``models[target][algo]`` is a fitted estimator. ``importances[target][algo]``
    is a feature-importance Series indexed by ``scope['final_features']``.
    """
    models, importances = {}, {}
    for target in TARGETS:
        models[target], importances[target] = {}, {}
        for algo, model in make_models(hyperparams, target).items():
            model.fit(scope['X_train'], scope['y_train'][target])
            models[target][algo] = model
            if algo == 'Lasso':
                imp = np.abs(model.named_steps['lasso'].coef_)
            else:
                imp = model.feature_importances_
            importances[target][algo] = pd.Series(
                imp, index=scope['final_features']
            ).sort_values(ascending=False)
    return models, importances


def predict_clip(model, X, target):
    """Predict and clip to the target's valid range (CLIP_RANGES)."""
    lo, hi = CLIP_RANGES[target]
    return np.clip(model.predict(X), lo, hi)


# --- Metrics ---

def calculate_all_metrics(y_true, y_pred, group_name='Overall'):
    """Basic regression metrics as a dict for one group."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    metrics = {'Group': group_name, 'N': len(y_true)}
    mean_observed = y_true.mean()
    metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['R_squared'] = r2_score(y_true, y_pred)
    metrics['Bias'] = np.mean(y_pred - y_true)
    if mean_observed != 0:
        metrics['%RMSE'] = (metrics['RMSE'] / abs(mean_observed)) * 100
        metrics['%Bias'] = (metrics['Bias'] / mean_observed) * 100
    else:
        metrics['%RMSE'], metrics['%Bias'] = np.nan, np.nan
    return metrics
