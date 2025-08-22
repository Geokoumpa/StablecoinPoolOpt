import pandas as pd
import pickle
from datetime import timedelta
from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection import bayesian_search_forecaster, TimeSeriesFold
from xgboost import XGBRegressor
from database.db_utils import get_db_connection
from forecasting.data_preprocessing import preprocess_data
from google.cloud import storage
from optuna.distributions import IntDistribution, FloatDistribution
from config import GCS_MODEL_BUCKET_NAME, ENVIRONMENT

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

def fetch_pool_data(pool_id: str) -> pd.DataFrame:
    """
    Fetches historical APY, TVL, and exogenous data for a specific pool from the database.
    All exogenous variables (ETH price, gas fees) are pre-merged in pool_daily_metrics.
    """
    engine = get_db_connection()
    with engine.connect() as conn:
        query = """
        SELECT
            date,
            rolling_apy_7d as apy_7d,
            actual_tvl as tvl_usd,
            eth_price_usd,
            btc_price_usd,
            gas_price_gwei
        FROM
            pool_daily_metrics
        WHERE
            pool_id = %s
        ORDER BY
            date;
        """
        df = pd.read_sql(query, conn, params=(pool_id,), parse_dates=['date'], index_col='date')
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
        return df

def train_and_forecast_pool(pool_id: str, steps: int = 1) -> dict:
    """
    Trains a forecasting model for a specific pool's APY and TVL,
    generates forecasts, and persists the model.
    """
    print(f"Processing pool: {pool_id}")

    # Fetch data - now includes exogenous variables from pool_daily_metrics
    data = fetch_pool_data(pool_id)
    print(f"Pool {pool_id} raw fetched data:\n{data}\nShape: {data.shape}")

    # Check data sufficiency
    if not has_sufficient_data(data):
        print(f"No sufficient data for pool {pool_id}. Skipping.")
        return {}

    if len(data) < 14:
        print(f"Not enough data points for bayesian_search_forecaster (requires > window_size=14). Skipping pool {pool_id}.")
        return {}
    
    # Ensure data is properly formatted
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    data.index = data.index.tz_localize('UTC') if data.index.tz is None else data.index.tz_convert('UTC')
    data.index = data.index.normalize()

    # Define exogenous columns for this specific forecasting task
    exogenous_cols = ['eth_price_usd', 'btc_price_usd', 'gas_price_gwei']
    
    # Check if we have sufficient exogenous data
    if data[exogenous_cols].isna().any().any():
        print(f"Pool {pool_id} has missing exogenous data, filling with forward fill...")
        data[exogenous_cols] = data[exogenous_cols].ffill()
    
    if data[exogenous_cols].isna().any().any():
        print(f"Pool {pool_id} still has missing exogenous data after forward fill, using backward fill...")
        data[exogenous_cols] = data[exogenous_cols].bfill()
    
    if data.shape[0] < 14:
        print(f"Pool {pool_id} has less than 14 rows, skipping.")
        return {}

    # Apply preprocessing
    data_processed = preprocess_data(data, exogenous_cols=exogenous_cols)

    # Ensure the index is datetime and has a frequency
    if not isinstance(data_processed.index, pd.DatetimeIndex):
        data_processed.index = pd.to_datetime(data_processed.index)
    if data_processed.index.freq is None:
        # Attempt to infer frequency, e.g., 'D' for daily
        data_processed = data_processed.asfreq('D')
        data_processed = data_processed.ffill() # Fill missing dates if any

    print(f"Pool {pool_id} processed dataset:\n{data_processed}\nShape: {data_processed.shape}")

    # Define forecasters for APY and TVL with increased lags to capture weekly patterns
    forecaster_apy = ForecasterRecursive(
        regressor=XGBRegressor(random_state=123),
        lags=14  # Capture up to 2-week patterns
    )
    forecaster_tvl = ForecasterRecursive(
        regressor=XGBRegressor(random_state=123),
        lags=14  # Capture up to 2-week patterns
    )

    # Hyperparameter tuning (example for APY, can be extended for TVL)
    print(f"Starting hyperparameter tuning for APY for pool {pool_id}...")
    print(f"Data processed length: {len(data_processed)}")
    
    # Define search space for Bayesian optimization using Optuna distributions
    def search_space_fn(trial):
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 5),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1)
        }
    
    # Adaptive validation setup optimized for 30-day exogenous data availability
    data_length = len(data_processed)
    
    if data_length < 45:
        # For smaller datasets, use simple validation but ensure we have enough training data
        print(f"Small dataset ({data_length} points), using simple train-test split")
        initial_train_size = max(int(data_length * 0.75), 21)  # Use 75% for training, minimum 21 (3 weeks)
        cv = TimeSeriesFold(
            steps=steps,
            initial_train_size=initial_train_size,
            refit=False
        )
    else:
        # For larger datasets, use more sophisticated cross-validation with better coverage
        test_size = max(steps * 5, 14)  # Minimum test size of 14 days (2 weeks)
        n_splits = min(4, (data_length - 35) // test_size)  # More splits with 35-day minimum training
        initial_train_size = max(data_length - n_splits * test_size, 35)  # Ensure positive and reasonable size
        
        print(f"Using {n_splits} splits with initial_train_size: {initial_train_size}, test_size: {test_size}")
        cv = TimeSeriesFold(
            steps=steps,
            initial_train_size=initial_train_size,
            refit=False
        )

    # Bayesian search for APY
    results_bayesian_apy, best_trial = bayesian_search_forecaster(
        forecaster=forecaster_apy,
        y=data_processed['apy_7d'],
        exog=data_processed[[f'{col}_shifted' for col in exogenous_cols] + ['day_of_week', 'day_of_year', 'month']],
        cv=cv,
        search_space=search_space_fn,
        metric='mean_squared_error',
        n_trials=5, # Number of trials for Bayesian search
        return_best=True,
        n_jobs=-1,
        verbose=False,
        show_progress=True
    )
    print(f"Best hyperparameters for APY: {forecaster_apy.regressor.get_params()}")

    # Train forecasters with best parameters
    print(f"Training forecasters for pool {pool_id}...")
    forecaster_apy.fit(
        y=data_processed['apy_7d'],
        exog=data_processed[[f'{col}_shifted' for col in exogenous_cols] + ['day_of_week', 'day_of_year', 'month']]
    )
    forecaster_tvl.fit(
        y=data_processed['tvl_usd'],
        exog=data_processed[[f'{col}_shifted' for col in exogenous_cols] + ['day_of_week', 'day_of_year', 'month']]
    )

    # Generate forecasts for the current day
    last_date = data_processed.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=steps, freq='D')

    # Prepare exogenous data for forecasting
    future_exog = pd.DataFrame(index=future_dates)
    for col in exogenous_cols:
        future_exog[f'{col}_shifted'] = data_processed[f'{col}_shifted'].iloc[-1] # Using last known value
    future_exog['day_of_week'] = future_exog.index.dayofweek
    future_exog['day_of_year'] = future_exog.index.dayofyear
    future_exog['month'] = future_exog.index.month

    forecast_apy = forecaster_apy.predict(steps=steps, exog=future_exog)
    forecast_tvl = forecaster_tvl.predict(steps=steps, exog=future_exog)

    # Update pool_daily_metrics with forecasted values
    engine = get_db_connection()
    with engine.connect() as conn:
        for i in range(steps):
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
    
    print(f"Forecasts for pool {pool_id} updated in pool_daily_metrics.")

    # Model persistence
    model_blob_name_apy = f'forecaster_apy_{pool_id}.pkl'
    model_blob_name_tvl = f'forecaster_tvl_{pool_id}.pkl'

    if ENVIRONMENT == 'production':
        print(f"Saving models to GCS bucket: {GCS_MODEL_BUCKET_NAME}")
        client = storage.Client()
        bucket = client.bucket(GCS_MODEL_BUCKET_NAME)

        # Save APY model
        blob_apy = bucket.blob(model_blob_name_apy)
        blob_apy.upload_from_string(pickle.dumps(forecaster_apy))
        print(f"APY model for pool {pool_id} saved to gs://{GCS_MODEL_BUCKET_NAME}/{model_blob_name_apy}")

        # Save TVL model
        blob_tvl = bucket.blob(model_blob_name_tvl)
        blob_tvl.upload_from_string(pickle.dumps(forecaster_tvl))
        print(f"TVL model for pool {pool_id} saved to gs://{GCS_MODEL_BUCKET_NAME}/{model_blob_name_tvl}")
        
        model_apy_path = f"gs://{GCS_MODEL_BUCKET_NAME}/{model_blob_name_apy}"
        model_tvl_path = f"gs://{GCS_MODEL_BUCKET_NAME}/{model_blob_name_tvl}"
    else:
        print("Running in development environment. Saving models locally.")
        import os
        os.makedirs('models', exist_ok=True)
        
        model_filename_apy = f'models/{model_blob_name_apy}'
        model_filename_tvl = f'models/{model_blob_name_tvl}'

        with open(model_filename_apy, 'wb') as f:
            pickle.dump(forecaster_apy, f)
        with open(model_filename_tvl, 'wb') as f:
            pickle.dump(forecaster_tvl, f)
        print(f"Models for pool {pool_id} saved locally to {model_filename_apy} and {model_filename_tvl}")

        model_apy_path = model_filename_apy
        model_tvl_path = model_filename_tvl

    return {
        'pool_id': pool_id,
        'forecast_apy': forecast_apy.to_dict(),
        'forecast_tvl': forecast_tvl.to_dict(),
        'model_apy_path': model_apy_path,
        'model_tvl_path': model_tvl_path
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

if __name__ == "__main__":
    # Use filtered pools by default (pools that passed final filtering including icebox)
    filtered_pool_ids = get_filtered_pool_ids()
    if not filtered_pool_ids:
        print("No filtered pools found in the database to forecast.")
        exit()
    
    print(f"Found {len(filtered_pool_ids)} filtered pools to forecast.")
    
    # Track statistics
    successful_forecasts = 0
    failed_forecasts = 0
    skipped_forecasts = 0
    
    for pool_id in filtered_pool_ids:
        print(f"Processing pool: {pool_id}")
        try:
            result = train_and_forecast_pool(pool_id, steps=1) # Forecast for 1 day ahead
            if result:
                successful_forecasts += 1
            else:
                skipped_forecasts += 1
        except Exception as e:
            print(f"Error forecasting for pool {pool_id}: {e}")
            failed_forecasts += 1
    
    # Print final summary
    print("\n" + "="*60)
    print("📊 FORECAST POOLS SUMMARY")
    print("="*60)
    print(f"Total pools processed: {len(filtered_pool_ids)}")
    print(f"✅ Successful forecasts: {successful_forecasts}")
    print(f"⏭️  Skipped (insufficient data): {skipped_forecasts}")
    print(f"❌ Failed forecasts: {failed_forecasts}")
    print(f"📈 Success rate: {(successful_forecasts/len(filtered_pool_ids)*100):.1f}%")
    print("="*60)