import logging
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import List, Dict, Optional, Tuple
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import text
from tqdm.auto import tqdm

from database.db_utils import get_db_connection
from forecasting.panel_data_utils import fetch_panel_history, build_pool_feature_row, fetch_all_macro_daily, EXOG_BASE, LAG_SETS
from forecasting.neighbor_features import add_neighbor_features
from forecasting.model_utils import fit_global_panel_model, make_tvl_oof

logger = logging.getLogger(__name__)

# Constants from notebook
HIST_DAYS_PANEL = 150
MIN_ROWS_PANEL = 400
GROUP_COL = "pool_group"

def get_filtered_pool_ids(limit: Optional[int] = None) -> List[str]:
    """
    Fetches pool_ids from pool_daily_metrics that are not filtered out and are active.
    """
    engine = get_db_connection()
    sql = """
    SELECT DISTINCT pool_id
    FROM pool_daily_metrics
    WHERE is_filtered_out = FALSE
    """
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
        ids = df["pool_id"].tolist()
        return ids[:limit] if limit else ids

def _count_actual_history(panel_df: pd.DataFrame, pid: str, asof_norm: pd.Timestamp) -> int:
    """Count actual history for a pool up to asof date."""
    return int(
        panel_df.loc[
            (panel_df['pool_id'] == pid) &
            (panel_df['date'] <= asof_norm) &
            (panel_df['actual_apy'].notna())
        ].shape[0]
    )

def _baseline_from_actual(panel_df: pd.DataFrame, pid: str, asof_norm: pd.Timestamp) -> Optional[float]:
    """Get baseline from actual APY values (mean of last 2 values)."""
    hist = (panel_df.loc[
        (panel_df['pool_id'] == pid) &
        (panel_df['date'] <= asof_norm) &
        (panel_df['actual_apy'].notna()),
        'actual_apy'
    ].sort_index())
    
    if len(hist) >= 2:
        return float(hist.tail(2).mean())
    return None

def build_global_panel_dataset(asof_start: pd.Timestamp, asof_end: pd.Timestamp,
                          pool_ids: List[str], group_col: str = GROUP_COL,
                          hist_days: int = HIST_DAYS_PANEL) -> pd.DataFrame:
    """
    Enhanced version with macroeconomic data.
    Build training rows with target = next-day actual_apy.
    Only include pools that have >=3 valid actual_apy values by `asof` (cold-start guard).
    """
    rows = []
    days = pd.date_range(asof_start, asof_end, freq='D')
    engine = get_db_connection()

    logger.info(f"Building global panel dataset from {asof_start.date()} to {asof_end.date()}")

    # Load macroeconomic data with lags
    macro_daily = fetch_all_macro_daily(engine,
                                        start_date=asof_start - pd.Timedelta(days=hist_days),
                                        end_date=asof_end)

    if not macro_daily.empty:
        macro_daily = macro_daily.sort_values("date")
        macro_cols = [c for c in macro_daily.columns if c != "date"]

        # Example: 1d, 3d, 7d lags for all macro series
        for lag in [1, 3, 7]:
            lagged = macro_daily[macro_cols].shift(lag)
            lagged.columns = [f"{c}_lag{lag}" for c in macro_cols]
            macro_daily = pd.concat([macro_daily, lagged], axis=1)

        macro_daily = macro_daily.ffill()

    for t in tqdm(days, desc="Building panel dataset"):
        t = pd.Timestamp(t)
        t = t.tz_localize('UTC') if t.tz is None else t.tz_convert('UTC')

        panel = fetch_panel_history(t, pool_ids, days=hist_days, group_col=group_col)
        if panel.empty:
            continue

        # Merge with macroeconomic data
        if not macro_daily.empty:
            panel = panel.merge(
                macro_daily,
                on="date",
                how="inner"
            )

        panel = add_neighbor_features(panel, group_col=group_col)

        # Get realized next day (target)
        with engine.connect() as conn:
            realized = pd.read_sql(
                text("""SELECT pool_id, date, actual_apy,actual_tvl
                        FROM pool_daily_metrics
                        WHERE date = :d"""),
                conn,
                params={"d": (t + pd.Timedelta(days=1)).normalize().to_pydatetime()}
            )
        realized['date'] = pd.to_datetime(realized['date'], utc=True).dt.normalize()

        pools_today = panel.loc[panel['date'] == t.normalize(), 'pool_id'].unique().tolist()

        for pid in pools_today:
            n_hist = _count_actual_history(panel, pid, t.normalize())

            hist_apys = panel.loc[
                (panel['pool_id'] == pid) &
                (panel['apy_7d'].notna()) &
                (panel['date'] <= t)
            ]['apy_7d']

            n_valid = len(hist_apys)

            if n_hist < 2:
                continue  # do not train on unstable early windows
            elif n_valid == 2:
                # simple average baseline
                baseline = hist_apys.mean()
                rows.append({
                    'pool_id': pid,
                    'asof': t.normalize(),
                    'target_apy_t1': np.nan,   # not used for training
                    'target_tvl_t1': np.nan,   # not used for training
                    'pred_global_apy': baseline,
                    'cold_start_flag': True
                })
                continue

            # Main model branch
            feat_row = build_pool_feature_row(panel, pid, t, group_col=group_col)
            if not feat_row:
                continue

            y_next_apy = realized.loc[realized['pool_id'] == pid, 'actual_apy']
            y_next_tvl = realized.loc[realized['pool_id'] == pid, 'actual_tvl']

            target_apy = float(y_next_apy.iloc[0]) if len(y_next_apy) > 0 and pd.notna(y_next_apy.iloc[0]) else np.nan
            target_tvl = float(y_next_tvl.iloc[0]) if len(y_next_tvl) > 0 and pd.notna(y_next_tvl.iloc[0]) else np.nan

            if pd.isna(target_apy):
                continue

            feat_row.update({
                'pool_id': pid,
                'asof': t.normalize(),
                'target_apy_t1': target_apy,
                'target_tvl_t1': target_tvl,
            })
            rows.append(feat_row)

    panel = pd.DataFrame(rows)
    panel = panel.sort_values(['pool_id', 'asof']).reset_index(drop=True)

    return panel

def train_global_models(train_start: pd.Timestamp, train_end: pd.Timestamp,
                       pool_ids: List[str], group_col: str = GROUP_COL,
                       hist_days: int = HIST_DAYS_PANEL,
                       use_tvl_stacking: bool = True,
                       n_trials: int = 30) -> Tuple[RandomForestRegressor, LGBMRegressor, List[str], List[str], LabelEncoder]:
    """
    Train enhanced global models (RandomForest for APY, LightGBM for TVL) for APY and TVL prediction.
    Returns trained models and feature lists, plus pool_id encoder.
    """
    logger.info(f"Training global models from {train_start.date()} to {train_end.date()}")

    # Build panel training dataset
    panel_train = build_global_panel_dataset(
        train_start, train_end, pool_ids, group_col, hist_days
    )

    if panel_train.empty:
        raise ValueError("No training data available")

    # Train TVL model first (for stacking)
    tvl_model = None
    tvl_feature_cols = None

    if use_tvl_stacking:
        logger.info("Training TVL model with stacking approach")
        panel_with_oof, tvl_model, tvl_oof_mae = make_tvl_oof(panel_train, n_splits=5)

        # Get TVL feature columns - must exclude all non-feature columns including tvl_hat_t1_oof
        from forecasting.model_utils import _get_feature_cols_for_training
        tvl_feature_cols = _get_feature_cols_for_training(
            panel_with_oof,
            extra_drop={'pool_id', 'asof', 'target_date', 'cold_start_flag',
                       'target_apy_t1', 'target_tvl_t1', 'pred_global_apy',
                       'tvl_hat_t1_oof'}
        )

        logger.info(f"TVL model trained with OOF MAE: {tvl_oof_mae}")
    else:
        panel_with_oof = panel_train.copy()

    # Train APY model (with TVL stacking if available)
    apy_model, apy_feature_cols, study, le_pool_encoder, _ = fit_global_panel_model(
        panel_with_oof, target_col='target_apy_t1', n_trials=n_trials
    )

    logger.info(f"APY RandomForest model trained with {len(apy_feature_cols)} features")

    return apy_model, tvl_model, apy_feature_cols, tvl_feature_cols, le_pool_encoder

def predict_global_lgbm(asof: pd.Timestamp,
                      pool_ids: List[str],
                      apy_model: RandomForestRegressor,
                      apy_feature_cols: List[str],
                      group_col: str = GROUP_COL,
                      hist_days: int = HIST_DAYS_PANEL,
                      tvl_model: Optional[LGBMRegressor] = None,
                      tvl_feature_cols: Optional[List[str]] = None,
                      pool_id_encoder: Optional[LabelEncoder] = None,
                      lag_pad_days: int = 7) -> pd.DataFrame:
    """
    Enhanced prediction function with RandomForest and macro features.
    Predict next-day APY per pool for `asof + 1`.
    If tvl_model & tvl_feature_cols are provided, first predict TVL and use as feature.
    """
    logger.info(f"Predicting for {asof.date()} with {len(pool_ids)} pools")

    asof = pd.Timestamp(asof)
    asof = asof.tz_localize('UTC') if asof.tz is None else asof.tz_convert('UTC')

    # Build historical panel with macro data
    effective_days = hist_days + lag_pad_days
    start = (asof - pd.Timedelta(days=effective_days)).normalize()
    engine = get_db_connection()

    # 1) Raw per-pool history
    panel = fetch_panel_history(asof, pool_ids, days=effective_days, group_col=group_col)
    if panel.empty:
        logger.warning("No panel data available for prediction")
        return pd.DataFrame()

    panel["date"] = pd.to_datetime(panel["date"], utc=True).dt.normalize()

    # 2) Macro history + lags
    macro_daily = fetch_all_macro_daily(
        engine,
        start_date=start,
        end_date=asof.normalize()
    )

    if not macro_daily.empty:
        macro_daily = macro_daily.sort_values("date")
        macro_daily["date"] = pd.to_datetime(macro_daily["date"], utc=True).dt.normalize()

        macro_cols = [c for c in macro_daily.columns if c != "date"]

        for lag in [1, 3, 7, 30]:
            lagged = macro_daily[macro_cols].shift(lag)
            lagged.columns = [f"{c}_lag{lag}" for c in macro_cols]
            macro_daily = pd.concat([macro_daily, lagged], axis=1)

        macro_daily = macro_daily.ffill()

        # 3) Merge macros with pool time-series
        panel = panel.merge(macro_daily, on="date", how="inner")

    # 4) Encode pool_id using SAME encoder as training
    if pool_id_encoder is not None:
        known = panel["pool_id"].astype(str).isin(pool_id_encoder.classes_)
        panel.loc[known, "pool_id_code"] = pool_id_encoder.transform(
            panel.loc[known, "pool_id"].astype(str)
        )
        panel.loc[~known, "pool_id_code"] = -1  # unknown bucket
        panel["pool_id_code"] = panel["pool_id_code"].astype(int)

    # 5) Trim to last hist_days (drop the lag_pad_days)
    cutoff_start = (asof - pd.Timedelta(days=hist_days)).normalize()
    panel = panel.loc[panel["date"] >= cutoff_start].copy()

    # 6) Drop rows with missing actual_apy (needed for feature building)
    panel.dropna(subset=["actual_apy"], inplace=True)

    panel = add_neighbor_features(panel, group_col=group_col)

    rows = []
    target_date = asof.normalize() + pd.Timedelta(days=1)

    pools_today = panel.loc[panel["date"] == asof.normalize(), "pool_id"].unique().tolist()

    for pid in pools_today:
        n_hist = _count_actual_history(panel, pid, asof.normalize())

        hist_apys = panel.loc[
            (panel['pool_id'] == pid) &
            (panel['apy_7d'].notna()) &
            (panel['date'] <= asof)
        ]['apy_7d']

        n_valid = len(hist_apys)

        if n_hist < 2:
            continue  # skip prediction for unstable pools

        # Cold-start baseline branch
        elif n_valid == 2:
            baseline_apy = float(hist_apys.tail(2).mean())
            baseline_tvl = float(panel.loc[
                (panel['pool_id'] == pid) &
                (panel['date'] == asof.normalize()),
                'tvl_usd'
            ].iloc[0]) if len(panel.loc[
                (panel['pool_id'] == pid) &
                (panel['date'] == asof.normalize()),
                'tvl_usd'
            ]) > 0 else np.nan

            rows.append({
                'pool_id': pid,
                'target_date': target_date,
                'pred_global_apy': baseline_apy,
                'pred_tvl_t1': baseline_tvl,
                'cold_start_flag': True
            })
            continue

        # Main model branch
        feat_row = build_pool_feature_row(panel, pid, asof, group_col=group_col)
        if not feat_row:
            continue

        fr = pd.DataFrame([feat_row])

        # Encode pool_id if needed
        if pool_id_encoder is not None:
            if "pool_id_code" not in fr.columns:
                if str(pid) in pool_id_encoder.classes_:
                    fr["pool_id_code"] = pool_id_encoder.transform([str(pid)])[0]
                else:
                    fr["pool_id_code"] = -1

        # Encode instability_group if present
        if "instability_group" in fr.columns:
            instab_map = {"insufficient": -1, "low": 0, "medium": 1, "high": 2}
            fr["instability_group_code"] = fr["instability_group"].map(instab_map).astype("Int64")

        # TVL stacking at inference
        tvl_hat = np.nan
        if tvl_model is not None and tvl_feature_cols is not None:
            # Ensure TVL features exist
            missing_tvl = [c for c in tvl_feature_cols if c not in fr.columns]
            for m in missing_tvl:
                fr[m] = 0.0

            X_tvl = fr[tvl_feature_cols].fillna(0.0)
            tvl_hat = float(tvl_model.predict(X_tvl)[0])
            fr["tvl_hat_t1_oof"] = tvl_hat

        # Ensure needed APY features exist
        for c in apy_feature_cols:
            if c not in fr.columns:
                fr[c] = 0.0

        fr_pred = fr[apy_feature_cols].fillna(0.0)

        # Data cleaning: handle infinities and extreme values
        # Replace infinities with NaN, then fill with 0
        fr_pred = fr_pred.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Clip extreme values to prevent float32 overflow
        # RandomForest can handle large values but we'll be conservative
        for col in fr_pred.columns:
            if fr_pred[col].dtype in [np.float64, np.float32]:
                # Calculate reasonable bounds using percentiles
                if fr_pred[col].std() > 0:
                    mean_val = fr_pred[col].mean()
                    std_val = fr_pred[col].std()
                    # Clip to mean ± 10*std to handle outliers
                    lower_bound = mean_val - 10 * std_val
                    upper_bound = mean_val + 10 * std_val
                    fr_pred[col] = fr_pred[col].clip(lower_bound, upper_bound)
                else:
                    # If no variation, clip to reasonable range
                    fr_pred[col] = fr_pred[col].clip(-1e6, 1e6)

        # Convert to float32 compatible range
        fr_pred = fr_pred.astype(np.float32)

        # Predict APY
        yhat_apy = float(apy_model.predict(fr_pred)[0])

        rows.append({
            'pool_id': pid,
            'target_date': target_date,
            'pred_global_apy': yhat_apy,
            'pred_tvl_t1': tvl_hat,
            'cold_start_flag': False
        })

    return pd.DataFrame(rows)

def persist_global_forecasts(predictions_df: pd.DataFrame):
    """
    Persist global model forecasts to pool_daily_metrics table.
    Maintains same interface as original system but with global model predictions.
    """
    if predictions_df.empty:
        logger.warning("No predictions to persist")
        return

    engine = get_db_connection()
    
    with engine.connect() as conn:
        for _, row in predictions_df.iterrows():
            pool_id = row['pool_id']
            target_date = row['target_date']
            apy_forecast = float(row['pred_global_apy'])
            tvl_forecast = float(row['pred_tvl_t1']) if pd.notna(row['pred_tvl_t1']) else None

            # Check if record exists
            check_query = text("""
            SELECT COUNT(*) as count FROM pool_daily_metrics
            WHERE pool_id = :pool_id AND date = :target_date
            """)
            result = conn.execute(check_query, {
                "pool_id": pool_id, 
                "target_date": target_date.strftime('%Y-%m-%d')
            })
            exists = result.fetchone()[0] > 0

            if exists:
                # Update existing record
                update_query = text("""
                UPDATE pool_daily_metrics
                SET forecasted_apy = :apy_forecast, forecasted_tvl = :tvl_forecast
                WHERE pool_id = :pool_id AND date = :target_date
                """)
                conn.execute(update_query, {
                    "apy_forecast": apy_forecast,
                    "tvl_forecast": tvl_forecast,
                    "pool_id": pool_id,
                    "target_date": target_date.strftime('%Y-%m-%d')
                })
            else:
                # Insert new record
                insert_query = text("""
                INSERT INTO pool_daily_metrics (pool_id, date, forecasted_apy, forecasted_tvl)
                VALUES (:pool_id, :target_date, :apy_forecast, :tvl_forecast)
                """)
                conn.execute(insert_query, {
                    "pool_id": pool_id,
                    "target_date": target_date.strftime('%Y-%m-%d'),
                    "apy_forecast": apy_forecast,
                    "tvl_forecast": tvl_forecast
                })
        
        conn.commit()
    
    logger.info(f"Persisted {len(predictions_df)} forecasts to database")

def train_and_forecast_global(pool_ids: Optional[List[str]] = None,
                            train_days: int = 60,
                            forecast_ahead: int = 1,
                            use_tvl_stacking: bool = True,
                            n_trials: int = 30) -> Dict:
    """
    Enhanced main function to train global models and generate forecasts.
    Uses RandomForest with Optuna for APY and LightGBM for TVL.
    """
    if pool_ids is None:
        pool_ids = get_filtered_pool_ids()

    if not pool_ids:
        logger.warning("No pool IDs found for forecasting")
        return {}

    logger.info(f"Processing {len(pool_ids)} pools with enhanced global modeling approach")

    # Set up training window
    end_date = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=1)  # Yesterday
    start_date = end_date - pd.Timedelta(days=train_days)

    try:
        # Train enhanced global models
        apy_model, tvl_model, apy_features, tvl_features, pool_id_encoder = train_global_models(
            start_date, end_date, pool_ids,
            use_tvl_stacking=use_tvl_stacking,
            n_trials=n_trials
        )

        # Generate forecasts
        predictions_df = predict_global_lgbm(
            end_date, pool_ids, apy_model, apy_features,
            tvl_model=tvl_model, tvl_feature_cols=tvl_features,
            pool_id_encoder=pool_id_encoder
        )

        # Persist forecasts
        persist_global_forecasts(predictions_df)

        # Return summary statistics
        cold_start_count = predictions_df['cold_start_flag'].sum()
        total_count = len(predictions_df)

        result = {
            'total_pools': total_count,
            'cold_start_pools': int(cold_start_count),
            'model_pools': int(total_count - cold_start_count),
            'train_window_days': train_days,
            'forecast_ahead_days': forecast_ahead,
            'use_tvl_stacking': use_tvl_stacking,
            'apy_feature_count': len(apy_features),
            'tvl_feature_count': len(tvl_features) if tvl_features else 0,
            'optuna_trials': n_trials,
            'model_type': 'RandomForest + LightGBM with macro features'
        }

        logger.info(f"Enhanced global forecasting completed: {result}")
        return result

    except Exception as e:
        logger.error(f"Error in enhanced global forecasting: {e}")
        raise

def main():
    """
    Main entry point for enhanced global forecasting.
    Uses RandomForest with Optuna for APY and LightGBM for TVL with macroeconomic features.
    """
    logger.info("Starting enhanced global forecasting (RandomForest + LightGBM + Macro features)")

    try:
        result = train_and_forecast_global(
            pool_ids=None,  # Use all filtered pools
            train_days=60,
            forecast_ahead=1,
            use_tvl_stacking=True,
            n_trials=30  # Optuna trials for hyperparameter tuning
        )

        # Print comprehensive summary
        logger.info("\n" + "="*70)
        logger.info("📊 ENHANCED GLOBAL FORECASTING SUMMARY")
        logger.info("="*70)
        logger.info(f"Total pools processed: {result['total_pools']}")
        logger.info(f"✅ Model-based forecasts: {result['model_pools']}")
        logger.info(f"🟡 Cold-start baselines: {result['cold_start_pools']}")
        logger.info(f"📈 Success rate: {(result['model_pools']/result['total_pools']*100):.1f}%")
        logger.info(f"🔧 TVL stacking: {'Enabled' if result['use_tvl_stacking'] else 'Disabled'}")
        logger.info(f"🧠 APY features: {result['apy_feature_count']}")
        logger.info(f"💰 TVL features: {result['tvl_feature_count']}")
        logger.info(f"⚡ Optuna trials: {result['optuna_trials']}")
        logger.info(f"🤖 Model type: {result['model_type']}")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"Fatal error in enhanced global forecasting: {e}")
        raise

if __name__ == "__main__":
    main()