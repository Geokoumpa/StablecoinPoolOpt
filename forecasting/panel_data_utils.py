import logging
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import List, Dict, Optional
from sqlalchemy import text

from database.db_utils import get_db_connection
from forecasting.data_preprocessing import preprocess_data, create_lagged_features

# Import constants from global_forecasting to avoid circular import issues
# These are used in build_pool_feature_row
EXOG_BASE = ['eth_open', 'btc_open', 'gas_price_gwei', 'tvl_usd', 'apy_7d']
LAG_SETS = {
    'eth_open': [7, 30],
    'btc_open': [7, 30],
    'gas_price_gwei': [7, 30],
    'tvl_usd': [7, 30],
    'apy_7d': [7, 30],
}

logger = logging.getLogger(__name__)

def fetch_panel_history(asof: pd.Timestamp, pool_ids: List[str], days: int = 150,
                        group_col: str = "pool_group") -> pd.DataFrame:
    """
    Read-only: fetch last `days` of history up to `asof` for given pools.
    Returns tidy df with columns:
      date, pool_id, apy_7d, actual_apy, tvl_usd, eth_open, btc_open, gas_price_gwei, group_col
    """
    t = pd.Timestamp(asof)
    t = t.tz_localize('UTC') if t.tz is None else t.tz_convert('UTC')
    start = (t - pd.Timedelta(days=days)).normalize()

    engine = get_db_connection()
    q = f"""
        SELECT
            date,
            pool_id,
            rolling_apy_7d AS apy_7d,
            actual_apy,
            actual_tvl AS tvl_usd,
            eth_open,
            btc_open,
            gas_price_gwei,
            {group_col} AS {group_col}
        FROM pool_daily_metrics
        WHERE pool_id = ANY(:pool_ids)
          AND date >= :start_date
          AND date <= :asof_date
        ORDER BY date ASC
    """
    
    with engine.connect() as conn:
        df = pd.read_sql(
            text(q), conn,
            params={
                "pool_ids": pool_ids,
                "start_date": start.tz_convert('UTC').to_pydatetime(),
                "asof_date": t.tz_convert('UTC').to_pydatetime()
            }
        )
    
    if df.empty:
        return df

    df['date'] = pd.to_datetime(df['date'], utc=True).dt.normalize()
    return df.sort_values(['date', 'pool_id'])

def build_pool_feature_row(panel_df: pd.DataFrame,
                           pool_id: str,
                           asof: pd.Timestamp,
                           group_col: str = "pool_group") -> Dict:
    """
    Build leakage-safe feature row for (pool_id, asof) using only information
    available at end-of-day `asof`, for forecasting next-day actual_apy.
    
    Uses preprocess_data(..., exogenous_cols=EXOG_BASE) which creates *_shifted
    versions of exogenous vars to avoid leakage.
    Adds explicit lags from LAG_SETS.
    Pulls neighbor features computed on `panel_df` for the same day (t) with *_nbr names.
    """
    # Normalize timezone
    asof = pd.Timestamp(asof)
    asof = asof.tz_localize('UTC') if asof.tz is None else asof.tz_convert('UTC')
    asof_day = asof.normalize()

    # history for this pool up to asof
    hist = (panel_df.loc[panel_df['pool_id'] == pool_id]
                    .sort_values('date')
                    .copy())

    if hist.empty:
        return {}

    # ensure datetime normalized
    hist['date'] = pd.to_datetime(hist['date'], utc=True).dt.normalize()
    hist = hist.set_index('date').sort_index()

    # require the asof day to be present (features at t to predict t+1)
    if asof_day not in hist.index:
        return {}

    # ---- base + exogenous (with _shifted from preprocess_data) ----
    feat = preprocess_data(hist.reset_index(), exogenous_cols=EXOG_BASE)
    
    # add explicit lags
    for col, lags in LAG_SETS.items():
        if col in feat.columns:
            feat = create_lagged_features(feat, col, lags)

    # keep only the row at `asof`
    if asof_day not in feat.index:
        return {}
    row = feat.loc[asof_day].to_dict()

    # ---- neighbor features on the same date (already computed on panel_df) ----
    day_pool = panel_df[
        (panel_df['date'] == asof_day) & 
        (panel_df['pool_id'] == pool_id)
    ]

    if len(day_pool) > 0:
        pool_data = day_pool.iloc[0]
        
        # neighbor feature names align with add_neighbor_features() output
        nbr_cols = [
            'group_tvl_sum_t_nbr',
            'group_apy_mean_t_nbr', 'group_apy_median_t_nbr', 'group_apy_std_t_nbr',
            'tvl_share_nbr', 'apy_rank_nbr',
            'grp_ex_mean_t_nbr', 'grp_ex_mean_7d_nbr',
            'group_tvl_sum_t_nbr_lag1', 'group_apy_mean_t_nbr_lag1',
            'group_apy_median_t_nbr_lag1', 'group_apy_std_t_nbr_lag1',
            'grp_ex_mean_t_nbr_lag1', 'grp_ex_mean_7d_nbr_lag1'
        ]

        for col in nbr_cols:
            if col in pool_data and pd.notna(pool_data[col]):
                row[col] = float(pool_data[col])
            else:
                row[col] = 0.0

    # ---- calendar (no leakage) ----
    dow = int(asof_day.dayofweek)
    doy = int(asof_day.dayofyear)
    row['dow_sin'] = np.sin(2*np.pi * dow / 7.0)
    row['dow_cos'] = np.cos(2*np.pi * dow / 7.0)
    row['doy_sin'] = np.sin(2*np.pi * doy / 365.25)
    row['doy_cos'] = np.cos(2*np.pi * doy / 365.25)

    # ---- stable pool id hash (0..1) ----
    row['pool_id_hash'] = _stable_hash_0_1(pool_id, mod=1000)

    # ---- keep the group label (categorical) as-is if you want to feed it to LGBM ----
    if group_col in pool_data and pd.notna(pool_data[group_col]):
        row[group_col] = pool_data[group_col]

    return row

def _stable_hash_0_1(s: str, mod: int = 1000) -> float:
    """Deterministic hash to [0,1) based on md5."""
    import hashlib
    h = hashlib.md5(s.encode('utf-8')).hexdigest()
    val = int(h[:8], 16) % mod
    return val / float(mod)

def _get_numeric_feature_cols(df: pd.DataFrame, 
                             exclude_cols: Optional[set] = None) -> List[str]:
    """Get numeric feature columns for model training."""
    if exclude_cols is None:
        exclude_cols = set()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in exclude_cols]