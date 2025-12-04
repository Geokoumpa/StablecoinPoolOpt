import logging
import pandas as pd
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

def classify_instability(row, low_thr, high_thr):
    """Classify pool into instability group based on volatility."""
    if row["history_days"] < 7 or pd.isna(row["instability_3d"]):
        return "insufficient"

    val = row["instability_3d"]

    if val < low_thr:
        return "low"
    elif val < high_thr:
        return "medium"
    else:
        return "high"


def add_neighbor_features(panel_df: pd.DataFrame, group_col: Optional[str] = None) -> pd.DataFrame:
    """
    Enhanced version with stability features and instability classification.

    For each (date, pool), add group-level neighbor stats and stability features.

    Intended usage: predicting t+1 actual_apy using features at t.
    We provide both same-day (t) and past-only (t-1) variants to let you choose.

    Features (suffix _nbr):
      - group_tvl_sum_t_nbr                 (sum TVL at t)
      - group_apy_mean_t_nbr / median / std (based on apy_7d at t)
      - tvl_share_nbr                       (pool TVL share within its group at t)
      - apy_rank_nbr                        (normalized rank of apy_7d within its group at t, 0..1)
      - grp_ex_mean_t_nbr                   (group mean apy_7d at t excluding the pool)
      - grp_ex_mean_7d_nbr                  (7d rolling mean of grp_ex_mean_t by group)
      - *_lag1 counterparts for past-only neighbor stats (computed from t-1)

    Stability Features:
      - apy_pct_change: Daily % change of APY per pool
      - apy_instability_abs: Absolute change = "instability" at each day
      - instability_3d: 3-day rolling instability using only past days
      - instability_3d_norm: Normalized instability [0,1] per pool
      - history_days: Number of days since first observation
      - instability_group: Classification (low/medium/high/insufficient)

    Notes:
      - If `group_col` is None or missing, uses instability-based grouping
      - This function uses only columns: ['date','pool_id','apy_7d','tvl_usd', 'actual_apy', group_col].
        Make sure they exist upstream (we'll create empties if missing to avoid crashes).
    """
    df = panel_df.copy()

    # Add Stability features
    # 1) Daily % change of APY per pool
    df['apy_pct_change'] = (
        df.groupby('pool_id')['actual_apy']
            .pct_change(fill_method=None)                 # (APY_t - APY_{t-1}) / APY_{t-1}
    )

    # 2) Absolute change = "instability" at each day
    df['apy_instability_abs'] = df['apy_pct_change'].abs()

    # 3) 3-day rolling instability *using only past days*:
    #    value at t = mean(|pct_change| at t-1, t-2, t-3)
    df['instability_3d'] = (
        df.groupby('pool_id')['apy_instability_abs']
            .transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
    )

    # 4) Normalize to [0,1] per pool (nice as a feature)
    df['instability_3d_norm'] = (
        df.groupby('pool_id')['instability_3d']
            .transform(lambda s: (s - s.min()) / (s.max() - s.min() + 1e-9))
    )

    # 5) Number of days since first observation
    df["history_days"] = df.groupby("pool_id").cumcount()

    # 6) Instability thresholds and classification
    mask = (df["history_days"] >= 7) & df["instability_3d"].notna()
    valid_instability = df.loc[mask, "instability_3d"]

    if len(valid_instability) > 0:
        # Typical quantiles: 50% and 85% (same good practice as volatility)
        low_thr, high_thr = valid_instability.quantile([0.50, 0.85]).tolist()

        # Classify instability into separate column
        df["instability_group"] = df.apply(lambda x: classify_instability(x, low_thr, high_thr), axis=1)
    else:
        df["instability_group"] = "insufficient"

    # Ensure required columns exist
    req = ['date', 'pool_id', 'apy_7d', 'tvl_usd']
    for c in req:
        if c not in df.columns:
            df[c] = np.nan

    # Handle grouping column
    if group_col is None or group_col not in df.columns:
        _grp_col = '__ALL__'
        df[_grp_col] = 'ALL'
    else:
        _grp_col = group_col

    # Normalize dates (daily) and sort
    if "date" in df.columns:
        df["asof"] = pd.to_datetime(df["date"], utc=True)
    elif "asof" in df.columns:
        df["asof"] = pd.to_datetime(df["asof"], utc=True)
    else:
        raise ValueError("Neither 'date' nor 'asof' exists in df in add_neighbor_features")

    df['asof'] = pd.to_datetime(df['asof'], utc=True).dt.normalize()
    df = df.sort_values(['asof', _grp_col, 'pool_id'])

    # -------- SAME-DAY (t) group stats (OK for t+1 forecasts) --------
    grp_t = df.groupby(['asof', _grp_col], dropna=False)

    group_tvl_sum_t   = grp_t['tvl_usd'].transform('sum')    .rename('group_tvl_sum_t_nbr')
    group_apy_mean_t  = grp_t['apy_7d'].transform('mean')   .rename('group_apy_mean_t_nbr')
    group_apy_median_t= grp_t['apy_7d'].transform('median') .rename('group_apy_median_t_nbr')
    group_apy_std_t   = grp_t['apy_7d'].transform('std')    .rename('group_apy_std_t_nbr')

    df = pd.concat([df, group_tvl_sum_t, group_apy_mean_t, group_apy_median_t, group_apy_std_t], axis=1)

    # TVL share inside group at t
    denom = df['group_tvl_sum_t_nbr'].replace(0, np.nan)
    df['tvl_share_nbr'] = (df['tvl_usd'] / denom).fillna(0.0)

    # Normalized rank (0..1) of apy_7d within group at t (includes pool itself)
    def _rank_norm(x):
        r = x.rank(method='average', na_option='keep')
        return (r - 1) / max(len(r) - 1, 1)

    df['apy_rank_nbr'] = grp_t['apy_7d'].transform(_rank_norm)

    # Exclude-this-pool group mean at t
    group_count_t = grp_t['apy_7d'].transform('count')
    group_sum_t   = grp_t['apy_7d'].transform('sum')

    # Add these to dataframe before using them
    df['group_count_t_nbr'] = group_count_t
    df['group_sum_t_nbr'] = group_sum_t

    excl_mean = (df['group_sum_t_nbr'] - df['apy_7d']) / (df['group_count_t_nbr'] - 1).replace([np.inf, -np.inf], np.nan)
    df['grp_ex_mean_t_nbr'] = excl_mean.fillna(df['group_apy_mean_t_nbr'])

    # Build group-daily series for rolling calcs (dedup per (date, group))
    g_daily = (
        df[['asof', _grp_col, 'grp_ex_mean_t_nbr']]
        .drop_duplicates(['asof', _grp_col])
        .sort_values(['asof', _grp_col])
    )

    # 7d rolling of excl-mean (per group)
    g_daily['grp_ex_mean_7d_nbr'] = (
        g_daily.groupby(_grp_col)['grp_ex_mean_t_nbr']
               .transform(lambda s: s.rolling(7, min_periods=3).mean())
    )

    df = df.merge(g_daily[['asof', _grp_col, 'grp_ex_mean_7d_nbr']],
                   on=['asof', _grp_col], how='left')

    # -------- PAST-ONLY (t-1) variants (for ultra-conservative setups) --------
    # Collapse same-day stats to (date, group), then shift by 1 day per group.
    g_same = (
        df[['asof', _grp_col,
            'group_tvl_sum_t_nbr', 'group_apy_mean_t_nbr', 'group_apy_median_t_nbr',
            'group_apy_std_t_nbr', 'grp_ex_mean_t_nbr', 'grp_ex_mean_7d_nbr']]
        .drop_duplicates(['asof', _grp_col])
        .sort_values(['asof', _grp_col])
        .copy()
    )

    for col in ['group_tvl_sum_t_nbr', 'group_apy_mean_t_nbr', 'group_apy_median_t_nbr',
                'group_apy_std_t_nbr', 'grp_ex_mean_t_nbr', 'grp_ex_mean_7d_nbr']:
        g_same[col + '_lag1'] = (
            g_same.groupby(_grp_col)[col]
                  .shift(1)
        )

    # Merge lag1 columns back to main dataframe
    lag_cols = ['asof', _grp_col] + [f'{c}_lag1' for c in
                ['group_tvl_sum_t_nbr', 'group_apy_mean_t_nbr', 'group_apy_median_t_nbr',
                 'group_apy_std_t_nbr', 'grp_ex_mean_t_nbr', 'grp_ex_mean_7d_nbr']]

    df = df.merge(
        g_same[lag_cols],
        on=['asof', _grp_col],
        how='left'
    )

    # Final NA handling for neighbor block
    fill_cols = [
        'group_tvl_sum_t_nbr','group_apy_mean_t_nbr','group_apy_median_t_nbr','group_apy_std_t_nbr',
        'tvl_share_nbr','apy_rank_nbr','grp_ex_mean_t_nbr','grp_ex_mean_7d_nbr',
        'group_tvl_sum_t_nbr_lag1','group_apy_mean_t_nbr_lag1','group_apy_median_t_nbr_lag1',
        'group_apy_std_t_nbr_lag1','grp_ex_mean_t_nbr_lag1','grp_ex_mean_7d_nbr_lag1'
    ]
    for c in fill_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    return df