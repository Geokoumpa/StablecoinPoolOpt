import logging
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
import optuna

logger = logging.getLogger(__name__)

# Extended macroeconomic features from notebook
EXOG_BASE = ['eth_open',
        'btc_open',
        'gas_price_gwei',
        'tvl_usd',
        'apy_7d',
        '1-Month Treasury Yield',
        '1-Year Treasury Bills Yield',
        '10-Year Treasury Minus 2-Year Treasury Spread',
        '10-Year Treasury Minus 3-Month Treasury Spread',
        '10-Year Treasury Yield', '2-Year Treasury Yield',
        '3-Month Treasury Yield', '30-Year Treasury Yield',
        '6-Month Treasury Yield',
        'Credit Suisse NASDAQ Gold FLOWS103 Price Index',
        'Federal Funds Effective Rate',
        'ICE BofA US High Yield Index Effective Yield',
        'M2 Not Seasonally Adjusted', 'M2 Seasonally Adjusted',
        'Nasdaq 100 Index', 'Nominal Broad U.S. Dollar Index',
        'Reverse Repo Yield (Overnight Reverse Repurchase Award Rate)',
        'S&P 500 Index (Daily Close)', 'SOFR 180-Day Average',
        'SOFR 30-Day Average', 'SOFR 90-Day Average', 'SOFR Index',
        'Secured Overnight Financing Rate']

LAG_SETS  = {
    'eth_open':        [7, 30],
    'btc_open':        [7, 30],
    'gas_price_gwei':  [7, 30],
    'tvl_usd':         [7, 30],
    'apy_7d':          [7, 30],
    '1-Month Treasury Yield': [7, 30],
    '1-Year Treasury Bills Yield': [7, 30],
    '10-Year Treasury Minus 2-Year Treasury Spread': [7, 30],
    '10-Year Treasury Minus 3-Month Treasury Spread': [7, 30],
    '10-Year Treasury Yield': [7, 30],
    '2-Year Treasury Yield': [7, 30],
    '3-Month Treasury Yield': [7, 30],
    '30-Year Treasury Yield': [7, 30],
    '6-Month Treasury Yield': [7, 30],
    'Credit Suisse NASDAQ Gold FLOWS103 Price Index': [7, 30],
    'Federal Funds Effective Rate': [7, 30],
    'ICE BofA US High Yield Index Effective Yield': [7, 30],
    'M2 Not Seasonally Adjusted': [7, 30],
    'M2 Seasonally Adjusted': [7, 30],
    'Nasdaq 100 Index': [7, 30],
    'Nominal Broad U.S. Dollar Index': [7, 30],
    'Reverse Repo Yield (Overnight Reverse Repurchase Award Rate)': [7, 30],
    'S&P 500 Index (Daily Close)': [7, 30],
    'SOFR 180-Day Average': [7, 30],
    'SOFR 30-Day Average': [7, 30],
    'SOFR 90-Day Average': [7, 30],
    'SOFR Index': [7, 30],
    'Secured Overnight Financing Rate': [7, 30]
}

def _get_feature_cols_for_training(df: pd.DataFrame, 
                             extra_drop: Optional[set] = None) -> List[str]:
    """Get numeric feature columns for model training."""
    if extra_drop is None:
        extra_drop = set()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in extra_drop]

def tune_apy_params_with_optuna(train_df: pd.DataFrame,
                                feature_cols: list,
                                val_frac: float = 0.3,
                                n_trials: int = 30,
                                timeout: int | None = None,
                                seed: int = 123) -> dict:
    """
    Returns best RandomForest params found by Optuna (minimizing blocked-CV MAE).
    """
    # Time-based train/valid split
    df = train_df.sort_values('asof').reset_index(drop=True).copy()
    n = len(df)
    split_idx = int(n * (1.0 - val_frac))
    if split_idx <= 0 or split_idx >= n:
        split_idx = int(n * 0.8)

    X_train = df.iloc[:split_idx][feature_cols].fillna(0.0)
    y_train = df.iloc[:split_idx]['target_apy_t1'].astype(float)
    X_valid = df.iloc[split_idx:][feature_cols].fillna(0.0)
    y_valid = df.iloc[split_idx:]['target_apy_t1'].astype(float)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 1, 24),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]
            ),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "n_jobs": -1,
            "random_state": seed,
        }

        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        mae = mean_absolute_error(y_valid, y_pred)
        return mae

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params.update({"n_jobs": -1, "random_state": seed})

    logger.info(f"Best RF params: {best_params}")
    logger.info(f"Best validation MAE: {study.best_value}")

    return best_params


def fit_global_panel_model(panel_df: pd.DataFrame,
                        target_col: str = 'target_apy_t1',
                        n_trials: int = 30,
                        val_frac: float = 0.3,
                        random_state: int = 123) -> Tuple[RandomForestRegressor, List[str], optuna.Study, LabelEncoder, pd.DataFrame]:
    """
    Fit RandomForest global model with Optuna tuning to predict target_col.
    Enhanced version based on notebook implementation.

    Returns:
      - best_model: fitted RandomForestRegressor on ALL data with best params
      - feat_cols: list of feature names
      - study: Optuna study (for inspection)
      - le_pool_encoder: LabelEncoder for pool_id
      - df: processed dataframe with encodings
    """
    df = panel_df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[target_col])

    # Ensure sorted by time if 'asof' exists
    if 'asof' in df.columns:
        df = df.sort_values('asof')

    # Encode pool_id
    le_pool_encoder = LabelEncoder()
    df["pool_id_code"] = le_pool_encoder.fit_transform(df["pool_id"].astype(str))

    # Encode instability_group if present
    if 'instability_group' in df.columns:
        instab_map = {
            "insufficient": -1,
            "low": 0,
            "medium": 1,
            "high": 2
        }
        df["instability_group_code"] = df["instability_group"].map(instab_map).astype("Int64")

    # Feature selection
    drop_like = {
        target_col, 'target_tvl_t1',  # targets
        'pred_global_apy', 'cold_start_flag',
        'target_date', 'asof', 'pool_id', 'hour'
    }

    feat_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in drop_like
    ]

    logger.info(f"Training global RF model with {len(feat_cols)} features")

    # Time-based train/valid split for Optuna
    n = len(df)
    split_idx = int(n * (1.0 - val_frac))
    if split_idx <= 0 or split_idx >= n:
        split_idx = int(n * 0.8)

    X_train = df.iloc[:split_idx][feat_cols].fillna(0.0)
    y_train = df.iloc[:split_idx][target_col].astype(float)
    X_valid = df.iloc[split_idx:][feat_cols].fillna(0.0)
    y_valid = df.iloc[split_idx:][target_col].astype(float)

    # Apply data cleaning to validation sets
    def clean_dataframe(X):
        # Replace infinities
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Clip extreme values
        for col in X_clean.columns:
            if X_clean[col].dtype in [np.float64, np.float32]:
                if X_clean[col].std() > 0:
                    mean_val = X_clean[col].mean()
                    std_val = X_clean[col].std()
                    lower_bound = mean_val - 10 * std_val
                    upper_bound = mean_val + 10 * std_val
                    X_clean[col] = X_clean[col].clip(lower_bound, upper_bound)
                else:
                    X_clean[col] = X_clean[col].clip(-1e6, 1e6)
        return X_clean

    X_train = clean_dataframe(X_train)
    X_valid = clean_dataframe(X_valid)

    # Optuna objective
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 1, 24),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]
            ),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "n_jobs": -1,
            "random_state": random_state,
        }

        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        mae = mean_absolute_error(y_valid, y_pred)
        return mae

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params.update({"n_jobs": -1, "random_state": random_state})

    # Refit best model on ALL data
    X_all = df[feat_cols].fillna(0.0)
    y_all = df[target_col].astype(float)

    # Data cleaning: handle infinities and extreme values for training
    X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Clip extreme values to prevent float32 overflow
    for col in X_all.columns:
        if X_all[col].dtype in [np.float64, np.float32]:
            if X_all[col].std() > 0:
                mean_val = X_all[col].mean()
                std_val = X_all[col].std()
                # Clip to mean ± 10*std to handle outliers
                lower_bound = mean_val - 10 * std_val
                upper_bound = mean_val + 10 * std_val
                X_all[col] = X_all[col].clip(lower_bound, upper_bound)
            else:
                # If no variation, clip to reasonable range
                X_all[col] = X_all[col].clip(-1e6, 1e6)

    best_model = RandomForestRegressor(**best_params)
    best_model.fit(X_all, y_all)

    logger.info(f"Best RF params: {best_params}")
    logger.info(f"Best validation MAE: {study.best_value}")
    logger.info("Global RF model training completed")

    return best_model, feat_cols, study, le_pool_encoder, df

def make_tvl_oof(panel_df: pd.DataFrame, n_splits: int = 5) -> Tuple[pd.DataFrame, LGBMRegressor, float]:
    """
    Create leakage-free OOF predictions for target_tvl_t1 using time-ordered blocked CV.
    Returns:
      - panel_df with new column 'tvl_hat_t1_oof'
      - fitted TVL model
      - OOF MAE score
    """
    df = panel_df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Only use rows with TVL targets
    tvl_df = df.dropna(subset=['target_tvl_t1']).copy()
    
    if tvl_df.empty:
        logger.warning("No TVL data available for stacking")
        return df, None, np.nan

    # Sort by time for proper CV splitting and keep track of original indices
    tvl_df = tvl_df.sort_values(['asof', 'pool_id'])
    tvl_df_original_indices = tvl_df.index.copy()
    
    # indices per time block
    n = len(tvl_df)
    fold_sizes = np.full(n_splits, n // n_splits, dtype=int)
    fold_sizes[: n % n_splits] += 1
    starts = np.cumsum(np.r_[0, fold_sizes[:-1]])
    ends = starts + fold_sizes

    # Feature selection (same as global model)
    drop_like = {
        'pool_id', 'asof', 'target_date', 'cold_start_flag',
        'target_apy_t1', 'target_tvl_t1', 'pred_global_apy'
    }
    
    feat_cols = _get_feature_cols_for_training(tvl_df, extra_drop=drop_like)
    
    oof_predictions = np.full(len(tvl_df), np.nan)
    maes = []

    logger.info(f"Running {n_splits}-fold time-series CV for TVL stacking")
    
    for i in range(1, n_splits):
        # train on blocks [0 .. i-1], predict block i
        train_idx = np.r_[0:ends[i-1]]
        val_idx = np.r_[ends[i-1]:ends[i]]
        
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
            
        train_df = tvl_df.iloc[train_idx]
        val_df = tvl_df.iloc[val_idx]
        
        if train_df.empty or val_df.empty:
            continue
            
        # Train TVL model
        X_train = train_df[feat_cols].fillna(0.0)
        y_train = train_df['target_tvl_t1'].astype(float)
        
        X_val = val_df[feat_cols].fillna(0.0)
        y_val = val_df['target_tvl_t1'].astype(float)
        
        # Ensure numeric types
        for c in X_train.columns:
            if not np.issubdtype(X_train[c].dtype, np.number):
                X_train[c] = pd.to_numeric(X_train[c], errors='coerce')
        for c in X_val.columns:
            if not np.issubdtype(X_val[c].dtype, np.number):
                X_val[c] = pd.to_numeric(X_val[c], errors='coerce')
                
        X_train = X_train.fillna(0.0)
        X_val = X_val.fillna(0.0)
        
        tvl_model = LGBMRegressor(
            objective='regression',
            n_estimators=800,
            num_leaves=64,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=123,
            force_col_wise=True,
            verbosity=-1
        )
        
        tvl_model.fit(X_train, y_train)
        preds = tvl_model.predict(X_val)
        
        mae = mean_absolute_error(y_val, preds)
        maes.append(mae)
        
        # Store OOF predictions
        oof_predictions[val_idx] = preds
        
        logger.info(f"Fold {i}: MAE={mae:.4f}, train_size={len(train_idx)}, val_size={len(val_idx)}")

    # Add OOF predictions to original dataframe using the original indices
    # Initialize with NaN for all rows
    df['tvl_hat_t1_oof'] = np.nan
    # Map the OOF predictions back to their original positions
    df.loc[tvl_df_original_indices, 'tvl_hat_t1_oof'] = oof_predictions
    
    # Fit final model on all data
    X_all = tvl_df[feat_cols].fillna(0.0)
    y_all = tvl_df['target_tvl_t1'].astype(float)
    
    for c in X_all.columns:
        if not np.issubdtype(X_all[c].dtype, np.number):
            X_all[c] = pd.to_numeric(X_all[c], errors='coerce')
    X_all = X_all.fillna(0.0)
    
    final_model = LGBMRegressor(
        objective='regression',
        n_estimators=800,
        num_leaves=64,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=123,
        force_col_wise=True,
        verbosity=-1
    )
    
    final_model.fit(X_all, y_all)
    
    oof_mae = float(np.mean(maes)) if maes else np.nan
    logger.info(f"TVL stacking completed with OOF MAE: {oof_mae}")
    
    return df, final_model, oof_mae