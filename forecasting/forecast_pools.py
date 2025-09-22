import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
import pandas as pd
from datetime import timedelta
from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection import bayesian_search_forecaster, TimeSeriesFold
from xgboost import XGBRegressor
from database.db_utils import get_db_connection
from forecasting.data_preprocessing import preprocess_data
from optuna.distributions import IntDistribution, FloatDistribution

logger = logging.getLogger(__name__)

def has_sufficient_data(df: pd.DataFrame) -> bool:
    """
    Checks if there is sufficient historical data for forecasting.
    Requires at least 14 days of data with minimum 10 non-null values
    for both APY and TVL, optimized for 30-day exogenous data window.
    """
    if len(df) < 14:
        return False
    
    # Check if we have at least 10 non-null values for better stability
    valid_apy = df['apy_7d'].notna().sum() >= 10
    valid_tvl = df['tvl_usd'].notna().sum() >= 10
    
    return valid_apy and valid_tvl


def persist_forecasts(pool_id: str, future_dates: pd.DatetimeIndex, apy_series: pd.Series, tvl_series: pd.Series):
    """
    Persist forecasted APY and TVL into pool_daily_metrics table.
    Reuses the same SQL update/insert pattern as the main training function.
    """
    engine = get_db_connection()
    from sqlalchemy import text

    # Make sure series are aligned with future_dates
    apy_series = pd.Series(apy_series, index=future_dates)
    tvl_series = pd.Series(tvl_series, index=future_dates)

    with engine.connect() as conn:
        for i, forecast_date in enumerate(future_dates):
            forecast_date_str = forecast_date.strftime('%Y-%m-%d')
            apy_forecast = float(apy_series.iloc[i])
            tvl_forecast = float(tvl_series.iloc[i])

            check_query = text("""
            SELECT COUNT(*) as count FROM pool_daily_metrics
            WHERE pool_id = :pool_id AND date = :forecast_date
            """)
            result = conn.execute(check_query, {"pool_id": pool_id, "forecast_date": forecast_date_str})
            exists = result.fetchone()[0] > 0

            if exists:
                update_query = text("""
                UPDATE pool_daily_metrics
                SET forecasted_apy = :apy_forecast, forecasted_tvl = :tvl_forecast
                WHERE pool_id = :pool_id AND date = :forecast_date
                """)
                conn.execute(update_query, {
                    "apy_forecast": apy_forecast,
                    "tvl_forecast": tvl_forecast,
                    "pool_id": pool_id,
                    "forecast_date": forecast_date_str
                })
            else:
                insert_query = text("""
                INSERT INTO pool_daily_metrics (pool_id, date, forecasted_apy, forecasted_tvl)
                VALUES (:pool_id, :forecast_date, :apy_forecast, :tvl_forecast)
                """)
                conn.execute(insert_query, {
                    "pool_id": pool_id,
                    "forecast_date": forecast_date_str,
                    "apy_forecast": apy_forecast,
                    "tvl_forecast": tvl_forecast
                })
        conn.commit()


def fallback_forecast_and_persist(pool_id: str, data: pd.DataFrame, steps: int = 1) -> dict:
    """
    Simple statistical fallback for pools with insufficient data.
    - APY: mean of last up-to-7 non-null values if >=3 samples, otherwise last non-null,
      otherwise global median across pools.
    - TVL: last non-null TVL if present, otherwise global median TVL, otherwise 0.
    Persists results into DB and returns the forecast dict.
    """
    steps_int = int(steps)
    today = pd.Timestamp.now(tz='UTC').normalize()
    future_dates = pd.date_range(start=today, periods=steps_int, freq='D')

    # APY fallback
    apy_nonnull = data['apy_7d'].dropna() if 'apy_7d' in data.columns else pd.Series(dtype=float)
    if len(apy_nonnull) >= 3:
        apy_val = float(apy_nonnull.tail(7).mean())
    elif len(apy_nonnull) >= 1:
        apy_val = float(apy_nonnull.iloc[-1])
    else:
        # Query global recent median APY
        engine = get_db_connection()
        try:
            with engine.connect() as conn:
                global_apy_df = pd.read_sql("""
                    SELECT rolling_apy_7d as apy_7d
                    FROM pool_daily_metrics
                    WHERE rolling_apy_7d IS NOT NULL
                    ORDER BY date DESC
                    LIMIT 1000
                """, conn)
            if not global_apy_df.empty:
                apy_val = float(global_apy_df['apy_7d'].median())
            else:
                apy_val = 0.0
        except Exception:
            apy_val = 0.0

    # TVL fallback
    tvl_val = 0.0
    if 'tvl_usd' in data.columns and data['tvl_usd'].dropna().any():
        tvl_val = float(data['tvl_usd'].dropna().iloc[-1])
    else:
        engine = get_db_connection()
        try:
            with engine.connect() as conn:
                global_tvl_df = pd.read_sql("""
                    SELECT actual_tvl as tvl_usd
                    FROM pool_daily_metrics
                    WHERE actual_tvl IS NOT NULL
                    ORDER BY date DESC
                    LIMIT 1000
                """, conn)
            if not global_tvl_df.empty:
                tvl_val = float(global_tvl_df['tvl_usd'].median())
            else:
                tvl_val = 0.0
        except Exception:
            tvl_val = 0.0

    # Create forecast series
    apy_forecast = pd.Series([apy_val] * steps_int, index=future_dates)
    tvl_forecast = pd.Series([tvl_val] * steps_int, index=future_dates)

    # Persist forecasts
    try:
        persist_forecasts(pool_id, future_dates, apy_forecast, tvl_forecast)
        logger.info(f"Fallback forecasts persisted for pool {pool_id}: apy={apy_val}, tvl={tvl_val}")
    except Exception as e:
        logger.error(f"Failed to persist fallback forecasts for pool {pool_id}: {e}")

    return {
        'pool_id': pool_id,
        'forecast_apy': apy_forecast.to_dict(),
        'forecast_tvl': tvl_forecast.to_dict(),
        'used_fallback': True
    }

def fetch_pool_data(pool_id: str) -> pd.DataFrame:
    """
    Fetches last 210 days (7 months) of historical APY, TVL, and exogenous data for a specific pool.
    Uses a sliding window of 210 days for comprehensive forecasting with lagged features.
    """
    engine = get_db_connection()
    with engine.connect() as conn:
        query = """
        SELECT
            date,
            rolling_apy_7d as apy_7d,
            actual_tvl as tvl_usd,
            eth_open,
            btc_open,
            gas_price_gwei
        FROM
            pool_daily_metrics
        WHERE
            pool_id = %s
            AND rolling_apy_7d IS NOT NULL
            AND rolling_apy_30d IS NOT NULL
            AND stddev_apy_7d IS NOT NULL
        ORDER BY
            date DESC
        LIMIT 210;
        """
        df = pd.read_sql(query, conn, params=(pool_id,), parse_dates=['date'], index_col='date')
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
        
        # Reverse to get chronological order (oldest to newest)
        df = df.sort_index()
        return df

def train_and_forecast_pool(pool_id: str, steps: int = 1) -> dict:
    """
    Trains a forecasting model for a specific pool's APY and TVL,
    generates forecasts, and persists the model.
    """
    logger.info(f"Processing pool: {pool_id}")

    # Fetch data - now includes exogenous variables from pool_daily_metrics
    data = fetch_pool_data(pool_id)
    logger.info(f"Pool {pool_id} raw fetched data:\n{data}\nShape: {data.shape}")

    # Check data sufficiency
    if not has_sufficient_data(data):
        logger.info(f"No sufficient data for pool {pool_id}. Using fallback statistical estimator.")
        return fallback_forecast_and_persist(pool_id, data, steps)

    # Ensure data is properly formatted
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    data.index = data.index.tz_localize('UTC') if data.index.tz is None else data.index.tz_convert('UTC')
    data.index = data.index.normalize()

    # Define exogenous columns for this specific forecasting task
    exogenous_cols = ['eth_open', 'btc_open', 'gas_price_gwei']
    
    # Check if we have sufficient exogenous data
    has_missing_exog = data[exogenous_cols].isna().any().any()
    if has_missing_exog:
        logger.info(f"Pool {pool_id} has missing exogenous data, filling with forward fill...")
        data[exogenous_cols] = data[exogenous_cols].ffill()

    has_missing_exog_after_ffill = data[exogenous_cols].isna().any().any()
    if has_missing_exog_after_ffill:
        logger.info(f"Pool {pool_id} still has missing exogenous data after forward fill, using backward fill...")
        data[exogenous_cols] = data[exogenous_cols].bfill()
    
    if data.shape[0] < 14:
        logger.info(f"Pool {pool_id} has less than 14 rows, using fallback statistical estimator.")
        return fallback_forecast_and_persist(pool_id, data, steps)

    # Apply preprocessing
    data_processed = preprocess_data(data, exogenous_cols=exogenous_cols)

    # Add additional lagged features for crypto prices (7-day and 30-day lags for longer-term patterns)
    from forecasting.data_preprocessing import create_lagged_features
    data_processed = create_lagged_features(data_processed, 'eth_open', [7, 30])
    data_processed = create_lagged_features(data_processed, 'btc_open', [7, 30])
    data_processed = create_lagged_features(data_processed, 'gas_price_gwei', [7, 30])

    # Use only the last 180 days (6 months) to ensure all lagged features have values
    data_processed = data_processed.tail(180)

    # Ensure the index is datetime and has a frequency
    if not isinstance(data_processed.index, pd.DatetimeIndex):
        data_processed.index = pd.to_datetime(data_processed.index)
    if data_processed.index.freq is None:
        # Attempt to infer frequency, e.g., 'D' for daily
        data_processed = data_processed.asfreq('D')
        data_processed = data_processed.ffill() # Fill missing dates if any

    logger.info(f"Pool {pool_id} processed dataset:\n{data_processed}\nShape: {data_processed.shape}")

    # Define forecasters for APY and TVL with fixed lags of 7 to align with experimental approach
    forecaster_apy = ForecasterRecursive(
        regressor=XGBRegressor(random_state=123),
        lags=7  # Fixed lags of 7 as per experimental script alignment
    )
    forecaster_tvl = ForecasterRecursive(
        regressor=XGBRegressor(random_state=123),
        lags=7  # Fixed lags of 7 as per experimental script alignment
    )

    # Hyperparameter tuning (example for APY, can be extended for TVL)
    logger.info(f"Starting hyperparameter tuning for APY for pool {pool_id}...")
    logger.info(f"Data processed length: {len(data_processed)}")

        # Check for NaN values in target and exogenous data
    logger.info(f"NaN in target (apy_7d): {data_processed['apy_7d'].isna().sum()}")
    logger.info(f"NaN in exogenous data: {data_processed[exogenous_cols].isna().sum()}")

    # Define additional exogenous features including lagged variables
    additional_features = ['day_of_week', 'day_of_year', 'month', 'year', 'week_of_year', 'quarter',
                           'eth_open_lag7', 'eth_open_lag30', 'btc_open_lag7', 'btc_open_lag30',
                           'gas_price_gwei_lag7', 'gas_price_gwei_lag30']

    # Filter out features that don't exist in the dataset
    available_additional_features = [feat for feat in additional_features if feat in data_processed.columns]
    logger.info(f"Available additional features: {available_additional_features}")

    # Combine shifted exogenous with additional features
    exog_features = [f'{col}_shifted' for col in exogenous_cols] + available_additional_features
    logger.info(f"Exogenous features for forecasting: {exog_features}")

    # Check for NaN in exogenous features
    exog_data = data_processed[exog_features]
    logger.info(f"NaN values in exogenous features: {exog_data.isna().sum().sum()}")

    # Drop rows with NaN values to ensure clean data for forecasting
    # Include TVL data in the combined dataset for cleaning
    data_columns = ['apy_7d']
    if 'tvl_usd' in data_processed.columns:
        data_columns.append('tvl_usd')
    
    combined_data = pd.concat([data_processed[data_columns], exog_data], axis=1)
    combined_data_clean = combined_data.dropna()
    logger.info(f"Data length after dropping NaN: {len(combined_data_clean)} (was {len(combined_data)})")
    logger.info(f"Columns in combined_data_clean: {combined_data_clean.columns.tolist()}")

    min_data_length_for_cv = 15  # Minimum required for basic forecasting
    if len(combined_data_clean) < min_data_length_for_cv:
        logger.info(f"After dropping NaN values, not enough data for forecasting. "
              f"Found {len(combined_data_clean)} records, need at least {min_data_length_for_cv}. Using fallback estimator for pool {pool_id}.")
        return fallback_forecast_and_persist(pool_id, data, steps)

    # Update data_processed and exog_data to clean versions
    # Use combined_data_clean directly since it already contains cleaned data
    data_processed_clean = combined_data_clean.copy()
    exog_data_clean = combined_data_clean[exog_features]

    # Define search space for Bayesian optimization using Optuna distributions
    def search_space_fn(trial):
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 5),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1)
        }

    # Adaptive validation setup optimized for 30-day exogenous data availability
    data_length = int(len(data_processed_clean))  # Ensure it's an integer
    
    # Ensure all variables are scalars
    min_lags = forecaster_apy.lags
    if hasattr(min_lags, '__len__') and not isinstance(min_lags, str):
        # If lags is an array, take the maximum
        min_lags = int(max(min_lags))
    else:
        min_lags = int(min_lags)
    
    min_train_size_required = min_lags + 1 # Must be > lags
    min_data_length_for_cv = min_train_size_required + int(steps) # At least one test sample
    
    print(f"Debug: data_length={data_length}, min_lags={min_lags}, min_train_size_required={min_train_size_required}, min_data_length_for_cv={min_data_length_for_cv}")

    if data_length < min_data_length_for_cv:
        logger.info(f"Not enough data points for cross-validation with lags={min_lags} and steps={steps}. "
              f"Found {data_length} records, need at least {min_data_length_for_cv}. Using fallback estimator for pool {pool_id}.")
        return fallback_forecast_and_persist(pool_id, data, steps)

    steps_int = int(steps)  # Ensure steps is an integer
    
    if data_length < 45:
        # For smaller datasets, use simple validation but ensure we have enough training data
        logger.info(f"Small dataset ({data_length} points), using simple train-test split")
        initial_train_size = max(min_train_size_required, data_length - steps_int)
        # Ensure initial_train_size is strictly less than data_length
        if initial_train_size >= data_length:
            initial_train_size = data_length - 1

        logger.info(f"Creating CV with initial_train_size: {initial_train_size}, data_length: {data_length}")
        cv = TimeSeriesFold(
            steps=steps_int,
            initial_train_size=initial_train_size,
            refit=False
        )
    else:
        # For larger datasets, use more sophisticated cross-validation with better coverage
        test_size = max(steps_int * 5, 14)  # Minimum test size of 14 days (2 weeks)
        n_splits = max(1, min(4, (data_length - 35) // test_size))  # Ensure at least 1 split
        initial_train_size = max(data_length - n_splits * test_size, 35)  # Ensure positive and reasonable size

        # Ensure initial_train_size is valid
        if initial_train_size >= data_length:
            initial_train_size = max(min_train_size_required, data_length - steps_int)
            n_splits = 1

        logger.info(f"Using {n_splits} splits with initial_train_size: {initial_train_size}, test_size: {test_size}, data_length: {data_length}")
        cv = TimeSeriesFold(
            steps=steps_int,
            initial_train_size=initial_train_size,
            refit=False
        )

    # Ensure data types are numeric and handle any potential type issues
    y_clean = data_processed_clean['apy_7d'].astype(float)
    exog_data_clean = data_processed_clean[exog_features].astype(float)
    
    # Ensure no infinite values using numpy
    import numpy as np
    
    if not np.isfinite(y_clean).all():
        logger.warning("Found infinite values in target variable, replacing with NaN and dropping")
        y_clean = y_clean.replace([float('inf'), float('-inf')], float('nan')).dropna()
        exog_data_clean = exog_data_clean.loc[y_clean.index]
    
    if not np.isfinite(exog_data_clean).all().all():
        logger.warning("Found infinite values in exogenous variables, replacing with NaN and dropping")
        finite_mask = np.isfinite(exog_data_clean).all(axis=1)
        y_clean = y_clean[finite_mask]
        exog_data_clean = exog_data_clean[finite_mask]
    
    logger.info(f"Final clean data length: {len(y_clean)}")
    
    # Additional validation to avoid any potential array comparison issues
    if len(y_clean) != len(exog_data_clean):
        logger.warning(f"Mismatch in data lengths: y_clean={len(y_clean)}, exog_data_clean={len(exog_data_clean)} - using fallback estimator")
        return fallback_forecast_and_persist(pool_id, data, steps)
    
    # Reset indices to ensure proper alignment
    y_clean = y_clean.reset_index(drop=True)
    exog_data_clean = exog_data_clean.reset_index(drop=True)
    
    # Ensure all data is finite
    logger.info(f"Y data check - finite: {np.isfinite(y_clean).all()}, any NaN: {y_clean.isna().any()}")
    logger.info(f"Exog data check - finite: {np.isfinite(exog_data_clean).all().all()}, any NaN: {exog_data_clean.isna().any().any()}")

    # Skip Bayesian optimization for now and use default parameters to test if that's the issue
    logger.info("Skipping Bayesian search and using default parameters to avoid array ambiguity...")
    default_params = {
        'n_estimators': 150,
        'max_depth': 4,
        'learning_rate': 0.05,
        'random_state': 123
    }
    forecaster_apy.regressor.set_params(**default_params)
    logger.info(f"Using default hyperparameters for APY: {default_params}")

    # Train forecasters with default parameters
    logger.info(f"Training forecasters for pool {pool_id}...")
    training_exog_cols = [f'{col}_shifted' for col in exogenous_cols] + ['day_of_week', 'day_of_year', 'month']
    training_exog_cols = [col for col in training_exog_cols if col in exog_data_clean.columns]
    
    # Prepare clean TVL data similarly to APY data
    tvl_data_available = 'tvl_usd' in data_processed_clean.columns
    if tvl_data_available:
        logger.info(f"TVL data available for pool {pool_id}")
        # Use the same cleaning approach as APY data - get TVL from combined_data_clean
        tvl_clean = combined_data_clean['tvl_usd'].astype(float) if 'tvl_usd' in combined_data_clean.columns else data_processed_clean['tvl_usd'].loc[combined_data_clean.index].astype(float)
        
        # Debug TVL data
        logger.info(f"TVL data before cleaning - shape: {tvl_clean.shape}, sample values: {tvl_clean.head().tolist()}")
        logger.info(f"TVL data stats - min: {tvl_clean.min()}, max: {tvl_clean.max()}, mean: {tvl_clean.mean()}")
        
        # Handle infinite values in TVL
        if not np.isfinite(tvl_clean).all():
            logger.warning("Found infinite values in TVL data, replacing with NaN and using available data")
            tvl_clean = tvl_clean.replace([float('inf'), float('-inf')], float('nan'))
            finite_mask = np.isfinite(tvl_clean)
            y_clean = y_clean[finite_mask]
            tvl_clean = tvl_clean[finite_mask]
            exog_data_clean = exog_data_clean[finite_mask]
            logger.info(f"TVL data after infinite value cleanup - shape: {tvl_clean.shape}")
        
        # Reset indices to ensure alignment
        tvl_clean = tvl_clean.reset_index(drop=True)
        logger.info(f"Final TVL data - shape: {tvl_clean.shape}, sample values: {tvl_clean.head().tolist()}")
    else:
        logger.info(f"No TVL data available for pool {pool_id}")
    
    # Get training exogenous data
    training_exog_data = exog_data_clean[training_exog_cols]

    forecaster_apy.fit(
        y=y_clean,
        exog=training_exog_data
    )
    
    if tvl_data_available:
        forecaster_tvl.fit(
            y=tvl_clean,
            exog=training_exog_data
        )

    # Always forecast for today, regardless of whether pool has data up to yesterday or today
    last_date = data_processed_clean.index[-1]
    today = pd.Timestamp.now(tz='UTC').normalize()
    
    # Always set forecast date to today
    forecast_start_date = today
    
    if last_date.date() >= today.date():
        logger.info(f"Pool {pool_id} has data up to {last_date.date()} (today or future), forecasting for today {forecast_start_date.date()}")
    else:
        logger.info(f"Pool {pool_id} has data up to {last_date.date()} (yesterday), forecasting for today {forecast_start_date.date()}")
    
    future_dates = pd.date_range(start=forecast_start_date, periods=steps_int, freq='D')

    # Prepare exogenous data for forecasting
    future_exog = pd.DataFrame(index=future_dates)
    for col in exogenous_cols:
        # Use the last known value from the original data since we reset indices
        future_exog[f'{col}_shifted'] = data_processed_clean[f'{col}_shifted'].iloc[-1] # Using last known value
    future_exog['day_of_week'] = future_exog.index.dayofweek
    future_exog['day_of_year'] = future_exog.index.dayofyear
    future_exog['month'] = future_exog.index.month

    # Ensure future_exog has the same columns as training data
    future_exog_filtered = future_exog[training_exog_cols]
    
    # Reset index to RangeIndex as expected by skforecast
    # The index should start one step ahead of the last window
    last_index = len(y_clean) - 1  # Last index of training data
    future_exog_filtered = future_exog_filtered.reset_index(drop=True)
    future_exog_filtered.index = range(last_index + 1, last_index + 1 + steps_int)
    
    forecast_apy = forecaster_apy.predict(steps=steps_int, exog=future_exog_filtered)
    
    if tvl_data_available:
        logger.info(f"Generating TVL forecast for pool {pool_id}")
        forecast_tvl = forecaster_tvl.predict(steps=steps_int, exog=future_exog_filtered)
        logger.info(f"TVL forecast generated: {forecast_tvl.tolist()}")
    else:
        # If no TVL data available, use a simple estimate based on the last known value
        logger.info(f"No TVL data available for forecasting, using fallback method for pool {pool_id}")
        last_tvl = data_processed_clean['tvl_usd'].iloc[-1] if 'tvl_usd' in data_processed_clean.columns else 0
        logger.info(f"Last known TVL value: {last_tvl}")
        forecast_tvl = pd.Series([last_tvl] * steps_int, index=future_dates)
        logger.info(f"TVL forecast (fallback): {forecast_tvl.tolist()}")

    # Update pool_daily_metrics with forecasted values
    engine = get_db_connection()
    with engine.connect() as conn:
        for i in range(steps_int):
            forecast_date = future_dates[i].strftime('%Y-%m-%d')
            apy_forecast = float(forecast_apy.iloc[i])
            tvl_forecast = float(forecast_tvl.iloc[i])

            # Use text() for proper SQL execution
            from sqlalchemy import text
            
            # First check if record exists
            check_query = text("""
            SELECT COUNT(*) as count FROM pool_daily_metrics
            WHERE pool_id = :pool_id AND date = :forecast_date
            """)
            result = conn.execute(check_query, {"pool_id": pool_id, "forecast_date": forecast_date})
            exists = result.fetchone()[0] > 0

            if exists:
                # Update existing record
                update_query = text("""
                UPDATE pool_daily_metrics
                SET forecasted_apy = :apy_forecast, forecasted_tvl = :tvl_forecast
                WHERE pool_id = :pool_id AND date = :forecast_date
                """)
                conn.execute(update_query, {
                    "apy_forecast": apy_forecast,
                    "tvl_forecast": tvl_forecast,
                    "pool_id": pool_id,
                    "forecast_date": forecast_date
                })
            else:
                # Insert new record
                insert_query = text("""
                INSERT INTO pool_daily_metrics (pool_id, date, forecasted_apy, forecasted_tvl)
                VALUES (:pool_id, :forecast_date, :apy_forecast, :tvl_forecast)
                """)
                conn.execute(insert_query, {
                    "pool_id": pool_id,
                    "forecast_date": forecast_date,
                    "apy_forecast": apy_forecast,
                    "tvl_forecast": tvl_forecast
                })
        
        conn.commit()
    
    logger.info(f"Forecasts for pool {pool_id} updated in pool_daily_metrics.")

    return {
        'pool_id': pool_id,
        'forecast_apy': forecast_apy.to_dict(),
        'forecast_tvl': forecast_tvl.to_dict()
    }

def get_filtered_pool_ids() -> list:
    """
    Fetches pool_ids from pool_daily_metrics that are not filtered out.
    """
    engine = get_db_connection()
    with engine.connect() as conn:
        query = """
        SELECT DISTINCT pool_id
        FROM pool_daily_metrics
        WHERE is_filtered_out = FALSE
        """
        df = pd.read_sql(query, conn)
        return df['pool_id'].tolist()

def main():
    # Use filtered pools by default (pools that passed final filtering including icebox)
    filtered_pool_ids = get_filtered_pool_ids()
    if not filtered_pool_ids:
        logger.info("No filtered pools found in the database to forecast.")
        return
    
    logger.info(f"Found {len(filtered_pool_ids)} filtered pools to forecast.")
    
    # Track statistics
    successful_forecasts = 0
    failed_forecasts = 0
    skipped_forecasts = 0
    fallback_forecasts = 0
    
    for pool_id in filtered_pool_ids:
        logger.info(f"Processing pool: {pool_id}")
        try:
            result = train_and_forecast_pool(pool_id, steps=1) # Forecast for 1 day ahead
            if isinstance(result, dict) and result:
                successful_forecasts += 1
                if result.get('used_fallback'):
                    fallback_forecasts += 1
            else:
                skipped_forecasts += 1
        except Exception as e:
            logger.error(f"Error forecasting for pool {pool_id}: {e}")
            failed_forecasts += 1
    
    # Print final summary
    logger.info("\n" + "="*60)
    logger.info("📊 FORECAST POOLS SUMMARY")
    logger.info("="*60)
    logger.info(f"Total pools processed: {len(filtered_pool_ids)}")
    logger.info(f"✅ Successful forecasts: {successful_forecasts}")
    logger.info(f"🟡 Fallback forecasts used: {fallback_forecasts}")
    logger.info(f"⏭️  Skipped (insufficient data): {skipped_forecasts}")
    logger.info(f"❌ Failed forecasts: {failed_forecasts}")
    logger.info(f"📈 Success rate: {(successful_forecasts/len(filtered_pool_ids)*100):.1f}%")
    logger.info("="*60)

if __name__ == "__main__":
    main()