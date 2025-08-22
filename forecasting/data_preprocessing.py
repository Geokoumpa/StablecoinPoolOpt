import pandas as pd
from datetime import datetime, timedelta

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
    # Requires 'gas_used' and 'eth_price_usd' columns.
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

    # 7. Missing Value Handling: Remove rows with missing values after all preprocessing.
    # This is crucial after creating lagged and shifted features.
    df = df.dropna()

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

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts time-based features from the DataFrame's datetime index.

    Args:
        df (pd.DataFrame): The input DataFrame with a datetime index.

    Returns:
        pd.DataFrame: The DataFrame with added time-based features.
    """
    df_copy = df.copy()
    df_copy['year'] = df_copy.index.year
    df_copy['month'] = df_copy.index.month
    df_copy['day'] = df_copy.index.day
    df_copy['day_of_week'] = df_copy.index.dayofweek
    df_copy['day_of_year'] = df_copy.index.dayofyear
    df_copy['week_of_year'] = df_copy.index.isocalendar().week.astype(int)
    df_copy['quarter'] = df_copy.index.quarter
    df_copy['is_month_start'] = df_copy.index.is_month_start.astype(int)
    df_copy['is_month_end'] = df_copy.index.is_month_end.astype(int)
    df_copy['is_quarter_start'] = df_copy.index.is_quarter_start.astype(int)
    df_copy['is_quarter_end'] = df_copy.index.is_quarter_end.astype(int)
    df_copy['is_year_start'] = df_copy.index.is_year_start.astype(int)
    df_copy['is_year_end'] = df_copy.index.is_year_end.astype(int)
    if 'hour' in df_copy.index.name or (df_copy.index.freq and 'H' in df_copy.index.freq):
        df_copy['hour'] = df_copy.index.hour
    return df_copy

def shift_exogenous_variables(df: pd.DataFrame, exogenous_cols: list) -> pd.DataFrame:
    """
    Shifts exogenous variables by one period to prevent data leakage.
    This assumes exogenous variables are known for the forecast period.

    Args:
        df (pd.DataFrame): The input DataFrame.
        exogenous_cols (list): A list of column names to be shifted.

    Returns:
        pd.DataFrame: The DataFrame with shifted exogenous variables.
    """
    df_copy = df.copy()
    for col in exogenous_cols:
        if col in df_copy.columns:
            df_copy[f'{col}_shifted'] = df_copy[col].shift(1)
            # Optionally, drop the original exogenous column if only the shifted version is needed
            # df_copy = df_copy.drop(columns=[col])
    return df_copy
