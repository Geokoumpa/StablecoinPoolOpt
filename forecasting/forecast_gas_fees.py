import pandas as pd
import pickle
from datetime import timedelta
from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection import bayesian_search_forecaster, TimeSeriesFold
from xgboost import XGBRegressor
from database.db_utils import get_db_connection
from forecasting.data_preprocessing import preprocess_data
from google.cloud import storage
from config import GCS_MODEL_BUCKET_NAME, ENVIRONMENT

def fetch_gas_data_for_forecasting() -> pd.DataFrame:
    """
    Fetches historical daily gas fee data from the database and calculates rolling metrics.
    """
    engine = get_db_connection()
    with engine.connect() as conn:
        query = """
        SELECT
            date,
            actual_avg_gas_gwei,
            actual_max_gas_gwei
        FROM
            gas_fees_daily
        WHERE actual_avg_gas_gwei IS NOT NULL
        ORDER BY
            date;
        """
        df = pd.read_sql(query, conn, parse_dates=['date'], index_col='date')
        
        if df.empty:
            print("No gas fee data found in gas_fees_daily table.")
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
        print(f"Insufficient data for rolling calculations. Found {len(df)} records, need at least 7.")
        return df
    
    # Rolling averages
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

def fetch_eth_price_daily() -> pd.DataFrame:
    """
    Fetches daily ETH price data from the database for exogenous variables.
    """
    engine = get_db_connection()
    with engine.connect() as conn:
        query = """
        SELECT
            data_timestamp AS timestamp,
            (raw_json_data->>'close')::numeric AS eth_price_usd
        FROM
            raw_coinmarketcap_ohlcv
        WHERE
            symbol = 'ETH'
        ORDER BY
            data_timestamp;
        """
        df = pd.read_sql(query, conn, parse_dates=['timestamp'], index_col='timestamp')
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
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

def train_and_forecast_gas_fees(steps: int = 1) -> dict:
    """
    Trains a forecasting model for gas fees, generates forecasts, and persists the model.
    """
    print("Processing gas fee forecasts...")

    # Fetch daily data only
    daily_data = fetch_gas_data_for_forecasting()
    eth_price_data = fetch_eth_price_daily()

    if daily_data.empty or eth_price_data.empty:
        print("No sufficient data for gas fee forecasting. Skipping.")
        return {}

    # No reference to hourly data, only use daily data

    # Merge daily gas fee data with daily ETH price data
    # Ensure both daily_data and eth_price_data have UTC-aware DatetimeIndex
    if daily_data.index.tz is None:
        daily_data.index = daily_data.index.tz_localize('UTC')
    else:
        daily_data.index = daily_data.index.tz_convert('UTC')
    if eth_price_data.index.tz is None:
        eth_price_data.index = eth_price_data.index.tz_localize('UTC')
    else:
        eth_price_data.index = eth_price_data.index.tz_convert('UTC')
    daily_merged = pd.merge(daily_data, eth_price_data, left_index=True, right_index=True, how='inner')

    # Print merged dataset for debugging
    print(f"Merged dataset shape: {daily_merged.shape}")
    print(f"Merged dataset columns: {daily_merged.columns.tolist()}")
    print(f"Date range: {daily_merged.index.min()} to {daily_merged.index.max()}")
    print(f"Sample of merged data:")
    print(daily_merged.head(10))
    print(f"Data types:")
    print(daily_merged.dtypes)

    # Define exogenous columns for gas fee forecasting
    exogenous_cols = ['eth_price_usd', 'rolling_avg_gas_7d', 'rolling_avg_gas_30d',
                     'gas_delta_today_yesterday', 'stddev_gas_7d', 'stddev_gas_30d']

    # Apply preprocessing to daily data
    data_processed_daily = preprocess_data(daily_merged, exogenous_cols=exogenous_cols)

    # Ensure the index is datetime and has a frequency
    if not isinstance(data_processed_daily.index, pd.DatetimeIndex):
        data_processed_daily.index = pd.to_datetime(data_processed_daily.index)
    if data_processed_daily.index.freq is None:
        data_processed_daily = data_processed_daily.asfreq('D')
        data_processed_daily = data_processed_daily.ffill()

    # Forecast target and exogenous variables
    daily_forecast_target = data_processed_daily['actual_avg_gas_gwei'].dropna()
    daily_exog_for_forecast = data_processed_daily[exogenous_cols].dropna()

    # Align indices before creating forecaster
    common_index = daily_forecast_target.index.intersection(daily_exog_for_forecast.index)
    daily_forecast_target = daily_forecast_target.loc[common_index]
    daily_exog_for_forecast = daily_exog_for_forecast.loc[common_index]

    if daily_forecast_target.empty:
        print("No sufficient daily aggregated data for gas fee forecasting. Skipping.")
        return {}

    # Define forecaster for average daily gas price
    forecaster_gas_price = ForecasterRecursive(
        regressor=XGBRegressor(random_state=123),
        steps=steps,
        features_type='datetime'
    )

    # Hyperparameter tuning
    print("Starting hyperparameter tuning for gas price...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1]
    }
    
    n_splits = 3
    test_size = steps * 7 # Example: 7 days for test size
    folds = TimeSeriesFold(n_splits=n_splits, test_size=test_size)

    results_bayesian_gas = bayesian_search_forecaster(
        forecaster=forecaster_gas_price,
        y=daily_forecast_target,
        exog=daily_exog_for_forecast,
        param_distributions=param_grid,
        steps=steps,
        metric='mean_squared_error',
        n_iter=5,
        refit=True,
        initial_train_size=len(daily_forecast_target) - n_splits * test_size,
        fixed_params=None,
        return_best=True,
        n_jobs=-1,
        verbose=False,
        show_progress=True
    )
    print(f"Best hyperparameters for gas price: {forecaster_gas_price.regressor.get_params()}")

    # Train forecaster with best parameters
    print("Training forecaster for gas fees...")
    forecaster_gas_price.fit(
        y=daily_forecast_target,
        exog=daily_exog_for_forecast
    )

    # Generate forecasts for the current day
    last_date = daily_forecast_target.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=steps, freq='D')

    # Prepare future exogenous data for forecasting
    # This is a simplification: in a real scenario, you'd need actual future ETH prices
    # or a separate model to forecast them. Here, we use the last known shifted value.
    future_exog_daily = pd.DataFrame(index=future_dates)
    for col in exogenous_cols:
        future_exog_daily[f'{col}_shifted'] = daily_exog_for_forecast[f'{col}_shifted'].iloc[-1]
    future_exog_daily['day_of_week'] = future_exog_daily.index.dayofweek
    future_exog_daily['day_of_year'] = future_exog_daily.index.dayofyear
    future_exog_daily['month'] = future_exog_daily.index.month
    # For 'hour', if daily forecast, it might be 0 or average, depending on how you want to use it.
    # For simplicity, we'll just set it to 0 or drop it if not relevant for daily.
    future_exog_daily['hour'] = 0 # Placeholder for daily forecast

    forecast_gas_price = forecaster_gas_price.predict(exog=future_exog_daily)

    # Update gas_fees_daily with forecasted values
    engine = get_db_connection()
    with engine.connect() as conn:
        with conn.begin():
            # Update forecasted values
            for i in range(steps):
                forecast_date = future_dates[i].strftime('%Y-%m-%d')
                gas_price_forecast = float(forecast_gas_price.iloc[i])

                from sqlalchemy import text
                insert_query = text("""
                    INSERT INTO gas_fees_daily (date, forecasted_avg_gas_gwei)
                    VALUES (:date, :forecasted_avg_gas_gwei)
                    ON CONFLICT (date) DO UPDATE SET
                        forecasted_avg_gas_gwei = EXCLUDED.forecasted_avg_gas_gwei;
                """)
                conn.execute(insert_query, {
                    "date": forecast_date,
                    "forecasted_avg_gas_gwei": gas_price_forecast
                })
    
    print("Gas fee forecasts updated in gas_fees_daily.")

    # Model persistence
    model_blob_name_gas = 'forecaster_gas_price.pkl'
    
    if ENVIRONMENT == 'production':
        print(f"Saving model to GCS bucket: {GCS_MODEL_BUCKET_NAME}")
        client = storage.Client()
        bucket = client.bucket(GCS_MODEL_BUCKET_NAME)

        # Save gas price model
        blob_gas = bucket.blob(model_blob_name_gas)
        blob_gas.upload_from_string(pickle.dumps(forecaster_gas_price))
        print(f"Gas fee model saved to gs://{GCS_MODEL_BUCKET_NAME}/{model_blob_name_gas}")
        
        model_path = f"gs://{GCS_MODEL_BUCKET_NAME}/{model_blob_name_gas}"

    else:
        print("Running in development environment. Saving model locally.")
        # Ensure the 'models' directory exists
        import os
        os.makedirs('models', exist_ok=True)
        
        model_filename_gas = f'models/{model_blob_name_gas}'

        with open(model_filename_gas, 'wb') as f:
            pickle.dump(forecaster_gas_price, f)
        print(f"Gas fee model saved locally to {model_filename_gas}")

        model_path = model_filename_gas

    return {
        'forecast_gas_price': forecast_gas_price.to_dict(),
        'model_path': model_filename_gas
    }

if __name__ == "__main__":
    try:
        train_and_forecast_gas_fees(steps=1) # Forecast for 1 day ahead
    except Exception as e:
        print(f"Error during gas fee forecasting: {e}")
