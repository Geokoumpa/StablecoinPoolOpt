import logging
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def preprocess_data(df: pd.DataFrame, exogenous_cols: list = None) -> pd.DataFrame:
    """
    Performs data preprocessing and feature engineering for forecasting models.

    Args:
        df (pd.DataFrame): The input DataFrame containing raw time series data.
        exogenous_cols (list): A list of column names to be treated as exogenous variables.

    Returns:
        pd.DataFrame: The preprocessed DataFrame with engineered features.
    """
    df = df.copy()

    # 1. Date Conversion: Convert raw date strings into proper datetime objects.
    # Assuming a 'date' or 'timestamp' column exists. Adjust column name as needed.
    for col in ['date', 'timestamp', 'day', 'hour']: # Add common date/time column names
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df = df.set_index(col).sort_index()
            break
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a datetime index after date conversion.")

    # 2. Unit Conversion: Transform gas price values from Wei to Gwei.
    if 'gas_price_wei' in df.columns:
        df['gas_price_gwei'] = df['gas_price_wei'] / 1e9

    # 3. Derived Financial Metrics: Calculate estimated gas fees in USD and ETH.
    # Requires 'gas_used' and 'gas_price_gwei' and 'eth_price_usd' columns.
    if 'gas_used' in df.columns and 'gas_price_gwei' in df.columns and 'eth_price_usd' in df.columns:
        df['gas_fee_eth'] = df['gas_used'] * df['gas_price_gwei'] / 1e9 # Convert Gwei to ETH
        df['gas_fee_usd'] = df['gas_fee_eth'] * df['eth_price_usd']

    # 4. Lagged Features: Create new features by shifting existing time series data.
    # Example: Lagging Ethereum closing price.
    if 'eth_price_usd' in df.columns:
        df['eth_price_usd_lag1'] = df['eth_price_usd'].shift(1)
        df['eth_price_usd_lag7'] = df['eth_price_usd'].shift(7) # Example for weekly lag

    # 5. Time-Based Features: Extract cyclical components from the date.
    df['day_of_week'] = df.index.dayofweek
    df['day_of_year'] = df.index.dayofyear
    df['month'] = df.index.month
    df['hour'] = df.index.hour # If hourly data

    # 6. Exogenous Variable Shifting: Shift exogenous variables by one period to prevent data leakage.
    # This assumes exogenous variables are known for the forecast period.
    if exogenous_cols:
        for col in exogenous_cols:
            if col in df.columns:
                df[f'{col}_shifted'] = df[col].shift(1)
                # Drop the original exogenous column if only the shifted version is needed for forecasting
                # df = df.drop(columns=[col])

    # 7. Missing Value Handling: Removed aggressive df.dropna()
    # Specific dropna calls in forecasting scripts should handle this more precisely.

    return df

def create_lagged_features(df: pd.DataFrame, column: str, lags: list) -> pd.DataFrame:
    """
    Creates lagged features for a specified column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to create lagged features for.
        lags (list): A list of integers representing the number of periods to lag.

    Returns:
        pd.DataFrame: The DataFrame with added lagged features.
    """
    df_copy = df.copy()
    for lag in lags:
        df_copy[f'{column}_lag{lag}'] = df_copy[column].shift(lag)
    return df_copy




