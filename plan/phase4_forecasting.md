# Phase 4: Forecasting Layer

This phase focuses on developing and implementing the forecasting models for pool APYs, TVLs, and gas fees, including data preprocessing, model training, and persistence.

## Detailed Tasks:

### 4.1 Implement Data Preprocessing & Feature Engineering
- [x] Create a Python module or script for data preprocessing and feature engineering, including:
    - Date Conversion: Convert raw date strings into proper datetime objects.
    - Unit Conversion: Transform gas price values from Wei to Gwei.
    - Derived Financial Metrics: Calculate estimated gas fees in USD and ETH.
    - Lagged Features: Create new features by shifting existing time series data (e.g., Ethereum closing price) by a specific number of periods.
    - Time-Based Features: Extract cyclical components from the date (e.g., day of the week).
    - Exogenous Variable Identification: Define which columns will be used as external predictors.
    - Exogenous Variable Shifting: Shift exogenous variables by one period to prevent data leakage.
    - Missing Value Handling: Remove rows with missing values after all preprocessing.

### 4.2 Implement `forecast_pools.py`
- [x] Create a Python script to:
    - Read historical actuals for APY and TVL from `pool_daily_metrics`.
    - Apply the data preprocessing and feature engineering steps.
    - Implement the rolling forecast with daily refitting using `skforecast.recursive.ForecasterRecursive` and `xgboost.XGBRegressor`.
    - Perform hyperparameter tuning using `skforecast.model_selection.bayesian_search_forecaster` and `skforecast.model_selection.TimeSeriesFold`.
    - Generate APY and TVL forecasts for selected pools for the current day.
    - Update `pool_daily_metrics` with the forecasted values.
    - Implement model persistence using `pickle` to save and reload trained forecaster objects.

### 4.3 Implement `forecast_gas_fees.py`
- [x] Create a Python script to:
    - Read historical actuals for gas fees from `gas_fees_hourly` and `gas_fees_daily`.
    - Apply the data preprocessing and feature engineering steps.
    - Calculate actuals for the previous day's gas fees.
    - Implement the rolling forecast with daily refitting for gas fees.
    - Generate gas fee forecasts for the current day.
    - Update `gas_fees_daily` with actuals and forecasted values.
    - Implement model persistence using `pickle`.

### 4.4 Integrate Forecasting Tools
- [x] Ensure proper integration of `skforecast`, `xgboost`, and `pickle` within the forecasting scripts.
- [x] Manage dependencies and environment for these libraries.