import logging
import pandas as pd
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

def add_neighbor_features(panel_df: pd.DataFrame, group_col: Optional[str] = None) -> pd.DataFrame:
    """
    For each (date, pool), add group-level neighbor stats.
    
    Intended usage: predicting t+1 actual_apy using features at t.
    We provide both same-day (t) and past-only (t-1) variants.
    
    Features (suffix _nbr):
      - group_tvl_sum_t_nbr                 (sum TVL at t)
      - group_apy_mean_t_nbr / median / std (based on apy_7d at t)
      - tvl_share_nbr                       (pool TVL share within its group at t)
      - apy_rank_nbr                        (normalized rank of apy_7d within its group at t, 0..1)
      - grp_ex_mean_t_nbr                   (group mean apy_7d at t excluding the pool)
      - grp_ex_mean_7d_nbr                  (7d rolling mean of grp_ex_mean_t by group)
      - *_lag1 counterparts for past-only neighbor stats (computed from t-1)
    
    Notes:
      - If `group_col` is None or missing, a single group 'ALL' is assumed (no leakage risk).
      - This function uses only columns: ['date','pool_id','apy_7d','tvl_usd', group_col].
        Make sure they exist upstream (we'll create empties if missing to avoid crashes).
    """
    df = panel_df.copy()

    # Ensure required columns exist
    req = ['date', 'pool_id', 'apy_7d', 'tvl_usd']
    for c in req:
        if c not in df.columns:
            df[c] = np.nan

    # Handle grouping column
    if group_col is None:
        _grp_col = '__ALL__'
        df[_grp_col] = 'ALL'
    else:
        _grp_col = group_col
        if _grp_col not in df.columns:
            df[_grp_col] = 'ALL'

    # Normalize dates (daily) and sort
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.normalize()
    df = df.sort_values(['date', _grp_col, 'pool_id'])

    # -------- SAME-DAY (t) group stats (OK for t+1 forecasts) --------
    grp_t = df.groupby(['date', _grp_col], dropna=False)

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
        df[['date', _grp_col, 'grp_ex_mean_t_nbr']]
        .drop_duplicates(['date', _grp_col])
        .sort_values(['date', _grp_col])
    )

    # 7d rolling of excl-mean (per group)
    g_daily['grp_ex_mean_7d_nbr'] = (
        g_daily.groupby(_grp_col)['grp_ex_mean_t_nbr']
               .transform(lambda s: s.rolling(7, min_periods=3).mean())
    )

    df = df.merge(g_daily[['date', _grp_col, 'grp_ex_mean_7d_nbr']], 
                   on=['date', _grp_col], how='left')

    # -------- PAST-ONLY (t-1) variants (for ultra-conservative setups) --------
    # Collapse same-day stats to (date, group), then shift by 1 day per group.
    g_same = (
        df[['date', _grp_col,
            'group_tvl_sum_t_nbr', 'group_apy_mean_t_nbr', 'group_apy_median_t_nbr',
            'group_apy_std_t_nbr', 'grp_ex_mean_t_nbr', 'grp_ex_mean_7d_nbr']]
        .drop_duplicates(['date', _grp_col])
        .sort_values(['date', _grp_col])
        .copy()
    )

    for col in ['group_tvl_sum_t_nbr', 'group_apy_mean_t_nbr', 'group_apy_median_t_nbr',
                'group_apy_std_t_nbr', 'grp_ex_mean_t_nbr', 'grp_ex_mean_7d_nbr']:
        g_same[col + '_lag1'] = (
            g_same.groupby(_grp_col)[col]
                  .shift(1)
        )

    # Merge lag1 columns back to main dataframe
    lag_cols = ['date', _grp_col] + [f'{c}_lag1' for c in 
                ['group_tvl_sum_t_nbr', 'group_apy_mean_t_nbr', 'group_apy_median_t_nbr',
                 'group_apy_std_t_nbr', 'grp_ex_mean_t_nbr', 'grp_ex_mean_7d_nbr']]
    
    df = df.merge(
        g_same[lag_cols],
        on=['date', _grp_col],
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