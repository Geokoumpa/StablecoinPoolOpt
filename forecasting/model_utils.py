import logging
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

def _get_feature_cols_for_training(df: pd.DataFrame, 
                             extra_drop: Optional[set] = None) -> List[str]:
    """Get numeric feature columns for model training."""
    if extra_drop is None:
        extra_drop = set()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in extra_drop]

def fit_global_panel_model(panel_df: pd.DataFrame, 
                        target_col: str = 'target_apy_t1') -> Tuple[LGBMRegressor, List[str]]:
    """
    Fit LightGBM global model to predict target_col.
    Automatically includes tvl_hat_t1_oof if present as stacked feature.
    """
    df = panel_df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[target_col])
    
    if df.empty:
        raise ValueError("No training data available after cleaning")

    # Feature selection
    drop_like = {
        'pool_id', 'asof', 'target_date', 'cold_start_flag',
        'target_apy_t1', 'target_tvl_t1', 'pred_global_apy',
        'tvl_hat_t1_oof'  # will be added if present
    }
    
    feat_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in drop_like
    ]

    # Auto-include TVL stacking if available
    if 'tvl_hat_t1_oof' in df.columns:
        feat_cols.append('tvl_hat_t1_oof')
        logger.info("Including TVL stacking feature: tvl_hat_t1_oof")

    logger.info(f"Training global model with {len(feat_cols)} features")
    
    X = df[feat_cols].fillna(0.0)
    y = df[target_col].astype(float)

    # Ensure all features are numeric
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors='coerce')
    X = X.fillna(0.0)

    model = LGBMRegressor(
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

    model.fit(X, y)
    logger.info("Global model training completed")
    
    return model, feat_cols

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