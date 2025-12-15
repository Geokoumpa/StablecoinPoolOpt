import pandas as pd
from datetime import timedelta
import logging
from typing import List, Dict, Any
from database.repositories.gas_fee_repository import GasFeeRepository
from forecasting.data_preprocessing import preprocess_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

# Lazy imports for heavy ML libraries - imported inside functions to reduce cold start time

def fetch_gas_data_for_forecasting() -> pd.DataFrame:
    """
    Fetches historical daily gas fee data, ETH and BTC prices from the database
    and calculates rolling metrics.
    """
    repo = GasFeeRepository()
    
    rows = repo.get_data_for_forecasting()
    
    if not rows:
        logger.warning("No sufficient gas fee, ETH, or BTC data found in gas_fees_daily table.")
        return pd.DataFrame()
        
    # Convert manually to DataFrame to ensure correct column names
    columns = ['date', 'actual_avg_gas_gwei', 'actual_max_gas_gwei', 'eth_open', 'btc_open']
    df = pd.DataFrame(rows, columns=columns)
    
    # Set date index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    if df.empty:
         return df
    
    # Ensure timezone awareness
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
    
    # Calculate rolling metrics on the fly
    df = calculate_gas_rolling_metrics(df)
    
    return df

def calculate_gas_rolling_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rolling metrics for gas fee data on the fly.
    """
    if df.empty or len(df) < 7:
        logger.warning(f"Insufficient data for rolling calculations. Found {len(df)} records, need at least 7.")
        return df
    
    # Rolling averages
    # Convert to numeric if needed, though they should be floats from DB
    df['actual_avg_gas_gwei'] = pd.to_numeric(df['actual_avg_gas_gwei'], errors='coerce')
    
    df['rolling_avg_gas_7d'] = df['actual_avg_gas_gwei'].rolling(window=7, min_periods=1).mean()
    df['rolling_avg_gas_30d'] = df['actual_avg_gas_gwei'].rolling(window=30, min_periods=1).mean()
    
    # Gas price deltas (today vs yesterday)
    df['gas_delta_today_yesterday'] = df['actual_avg_gas_gwei'].diff()
    
    # Rolling standard deviations
    df['stddev_gas_7d'] = df['actual_avg_gas_gwei'].rolling(window=7, min_periods=1).std()
    df['stddev_gas_30d'] = df['actual_avg_gas_gwei'].rolling(window=30, min_periods=1).std()
    
    # Standard deviation deltas
    df['stddev_gas_7d_delta'] = df['stddev_gas_7d'].diff()
    df['stddev_gas_30d_delta'] = df['stddev_gas_30d'].diff()
    
    return df

def calculate_previous_day_actuals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates actuals for the previous day's gas fees from hourly data.
    """
    df_daily = df.resample('D').agg(
        actual_avg_gas_gwei=('gas_price_gwei', 'mean'),
        actual_max_gas_gwei=('estimated_gas_usd', 'max') # Assuming estimated_gas_usd can be used for max gas
    )
    return df_daily

def train_and_forecast_gas_fees() -> dict:
    """
    Trains a forecasting model for gas fees and generates single-step ahead forecast.
    """
    # Lazy imports for heavy ML libraries
    from skforecast.recursive import ForecasterRecursive
    from xgboost import XGBRegressor
    import numpy as np
    
    logger.info("Processing gas fee forecasts...")

    # Fetch daily data including ETH and BTC prices
    daily_merged = fetch_gas_data_for_forecasting()

    if daily_merged.empty:
        logger.warning("No sufficient data for gas fee forecasting. Skipping.")
        return {}

    # Print merged dataset for debugging
    logger.debug(f"Merged dataset shape: {daily_merged.shape}")

    # Define exogenous columns for gas fee forecasting
    exogenous_cols = ['eth_open', 'btc_open', 'rolling_avg_gas_7d', 'rolling_avg_gas_30d',
                     'gas_delta_today_yesterday', 'stddev_gas_7d', 'stddev_gas_30d']

    # Apply preprocessing to daily data first
    data_processed_daily = preprocess_data(daily_merged, exogenous_cols=exogenous_cols)

    # Add lagged features for exogenous variables (7-day lags to align with forecaster lags)
    from forecasting.data_preprocessing import create_lagged_features
    data_processed_daily = create_lagged_features(data_processed_daily, 'eth_open', [7])
    data_processed_daily = create_lagged_features(data_processed_daily, 'btc_open', [7])
    
    # Use only the last 180 days (6 months) to ensure all lagged features have values
    data_processed_daily = data_processed_daily.tail(180)

    # Ensure the index is datetime and has a frequency
    if not isinstance(data_processed_daily.index, pd.DatetimeIndex):
        data_processed_daily.index = pd.to_datetime(data_processed_daily.index)
    if data_processed_daily.index.freq is None:
        data_processed_daily = data_processed_daily.asfreq('D')
        data_processed_daily = data_processed_daily.ffill()

    # Check for sufficient data
    if len(data_processed_daily) < 14:
        logger.warning(f"Gas fee data has less than 14 rows, skipping.")
        return {}

    # Define additional exogenous features including lagged variables
    additional_features = ['day_of_week', 'day_of_year', 'month', 'year', 'week_of_year', 'quarter',
                           'eth_open_lag7', 'btc_open_lag7']

    # Filter out features that don't exist in the dataset
    available_additional_features = [feat for feat in additional_features if feat in data_processed_daily.columns]

    # Combine shifted exogenous with additional features
    exog_features = [f'{col}_shifted' for col in exogenous_cols] + available_additional_features

    # Check for NaN in exogenous features
    exog_data = data_processed_daily[exog_features]

    # Drop rows with NaN values to ensure clean data for forecasting
    # Start with average gas data only to ensure we can always forecast
    data_columns = ['actual_avg_gas_gwei']
    
    combined_data = pd.concat([data_processed_daily[data_columns], exog_data], axis=1)
    combined_data_clean = combined_data.dropna()
    
    min_data_length_for_cv = 15  # Minimum required for basic forecasting
    if len(combined_data_clean) < min_data_length_for_cv:
        logger.warning(f"After dropping NaN values, not enough data for forecasting. "
              f"Found {len(combined_data_clean)} records, need at least {min_data_length_for_cv}. Skipping gas fee forecasting.")
        return {}

    # Update data_processed_daily to clean version
    data_processed_clean = combined_data_clean.copy()
    exog_data_clean = combined_data_clean[exog_features]
    
    # Add max gas data if available, using the same clean indices
    if 'actual_max_gas_gwei' in data_processed_daily.columns:
        # Use reindex instead of loc to handle potential missing indices gracefully (though indices match combined_data_clean)
        max_gas_series = data_processed_daily['actual_max_gas_gwei'].reindex(combined_data_clean.index)
        data_processed_clean['actual_max_gas_gwei'] = max_gas_series

    # Define forecasters for average and max daily gas price with fixed lags of 7
    forecaster_gas_price = ForecasterRecursive(
        regressor=XGBRegressor(random_state=123),
        lags=7  # Fixed lags of 7 as per experimental script
    )
    forecaster_max_gas_price = ForecasterRecursive(
        regressor=XGBRegressor(random_state=123),
        lags=7  # Fixed lags of 7 as per experimental script
    )

    # Prepare clean data for both average and max gas fees
    y_avg_clean = data_processed_clean['actual_avg_gas_gwei'].astype(float)
    exog_data_clean = data_processed_clean[exog_features].astype(float)
    
    # Check if max gas data is available and has sufficient data points
    max_gas_column_exists = 'actual_max_gas_gwei' in data_processed_clean.columns
    if max_gas_column_exists:
        y_max_clean = data_processed_clean['actual_max_gas_gwei'].astype(float)
        # Check if we have sufficient non-NaN data points for training (need at least 15 for basic forecasting)
        valid_max_gas_count = y_max_clean.notna().sum()
        max_gas_data_available = valid_max_gas_count >= 15
    else:
        max_gas_data_available = False
    
    if not np.isfinite(y_avg_clean).all():
        logger.warning("Found infinite values in avg gas target variable, replacing with NaN and dropping")
        y_avg_clean = y_avg_clean.replace([float('inf'), float('-inf')], float('nan')).dropna()
        exog_data_clean = exog_data_clean.loc[y_avg_clean.index]
        if max_gas_data_available:
            y_max_clean = y_max_clean.loc[y_avg_clean.index]
    
    if max_gas_data_available and not np.isfinite(y_max_clean).all():
        logger.warning("Found infinite values in max gas target variable, replacing with NaN and using available data")
        y_max_clean = y_max_clean.replace([float('inf'), float('-inf')], float('nan'))
        finite_mask = np.isfinite(y_max_clean)
        y_avg_clean = y_avg_clean[finite_mask]
        y_max_clean = y_max_clean[finite_mask]
        exog_data_clean = exog_data_clean[finite_mask]
    
    if not np.isfinite(exog_data_clean).all().all():
        logger.warning("Found infinite values in exogenous variables, replacing with NaN and dropping")
        finite_mask = np.isfinite(exog_data_clean).all(axis=1)
        y_avg_clean = y_avg_clean[finite_mask]
        exog_data_clean = exog_data_clean[finite_mask]
        if max_gas_data_available:
            y_max_clean = y_max_clean[finite_mask]
    
    # Reset indices to ensure proper alignment
    y_avg_clean = y_avg_clean.reset_index(drop=True)
    exog_data_clean = exog_data_clean.reset_index(drop=True)
    if max_gas_data_available:
        y_max_clean = y_max_clean.reset_index(drop=True)
    
    # Hyperparameter tuning logic skipped for brevity/simplicity as in original
    default_params = {
        'n_estimators': 150,
        'max_depth': 4,
        'learning_rate': 0.05,
        'random_state': 123
    }
    forecaster_gas_price.regressor.set_params(**default_params)
    if max_gas_data_available:
        forecaster_max_gas_price.regressor.set_params(**default_params)
    logger.info(f"Using default hyperparameters for gas price: {default_params}")

    # Train forecasters with best parameters
    training_exog_cols = [f'{col}_shifted' for col in exogenous_cols] + ['day_of_week', 'day_of_year', 'month']
    training_exog_cols = [col for col in training_exog_cols if col in exog_data_clean.columns]
    
    # Get training exogenous data
    training_exog_data = exog_data_clean[training_exog_cols]

    # Train average gas price forecaster
    forecaster_gas_price.fit(
        y=y_avg_clean,
        exog=training_exog_data
    )
    
    # Train max gas price forecaster if data is available
    if max_gas_data_available:
        forecaster_max_gas_price.fit(
            y=y_max_clean,
            exog=training_exog_data
        )

    # Always forecast for today, regardless of whether we have data up to yesterday or today
    last_date = data_processed_clean.index[-1]
    today = pd.Timestamp.now(tz='UTC').normalize()
    
    # Always set forecast date to today
    forecast_start_date = today
    steps_int = 1
    
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
    last_index = len(y_avg_clean) - 1
    future_exog_filtered = future_exog_filtered.reset_index(drop=True)
    future_exog_filtered.index = range(last_index + 1, last_index + 1 + steps_int)
    
    # Generate forecasts
    forecast_avg_gas_price = forecaster_gas_price.predict(steps=steps_int, exog=future_exog_filtered)
    
    if max_gas_data_available:
        forecast_max_gas_price = forecaster_max_gas_price.predict(steps=steps_int, exog=future_exog_filtered)
    else:
        # Fallback
        if 'actual_max_gas_gwei' in daily_merged.columns:
            recent_max_gas_values = daily_merged['actual_max_gas_gwei'].dropna()
            
            if len(recent_max_gas_values) >= 7:
                recent_max_gas_array = recent_max_gas_values.tail(7).values
                mean_max_gas = np.mean(recent_max_gas_array)
                median_max_gas = np.median(recent_max_gas_array)
                std_max_gas = np.std(recent_max_gas_array)
                
                if len(recent_max_gas_array) >= 3:
                    recent_trend = np.mean(recent_max_gas_array[-3:]) - np.mean(recent_max_gas_array[:3])
                    forecast_max_gas_value = median_max_gas + (recent_trend * 0.3)
                else:
                    forecast_max_gas_value = median_max_gas
                
                forecast_max_gas_value = max(0.1, min(forecast_max_gas_value, mean_max_gas + 2 * std_max_gas))
            else:
                forecast_max_gas_value = forecast_avg_gas_price.iloc[0] * 1.5
        else:
            forecast_max_gas_value = forecast_avg_gas_price.iloc[0] * 1.5
        
        forecast_max_gas_price = pd.Series([forecast_max_gas_value] * steps_int, index=future_dates)

    # Update gas_fees_daily with forecasted values using Repository
    updates = []
    for i in range(steps_int):
        forecast_date = future_dates[i].date() # Use date object
        avg_gas_price_forecast = float(forecast_avg_gas_price.iloc[i])
        max_gas_price_forecast = float(forecast_max_gas_price.iloc[i])
        
        updates.append({
            'date': forecast_date,
            'forecasted_avg_gas_gwei': avg_gas_price_forecast,
            'forecasted_max_gas_gwei': max_gas_price_forecast
        })
        
    repo = GasFeeRepository()
    try:
        repo.upsert_forecasts(updates)
        logger.info(f"Gas fee forecasts (avg and max) updated/inserted for {len(updates)} days.")
    except Exception as e:
        logger.error(f"Error persisting gas fee forecasts: {e}")
        raise

    return {
        'forecast_avg_gas_price': forecast_avg_gas_price.to_dict(),
        'forecast_max_gas_price': forecast_max_gas_price.to_dict()
    }

def main():
    try:
        train_and_forecast_gas_fees() # Single-step ahead forecast for today
    except Exception as e:
        logger.error(f"Error during gas fee forecasting: {e}")

if __name__ == "__main__":
    main()
