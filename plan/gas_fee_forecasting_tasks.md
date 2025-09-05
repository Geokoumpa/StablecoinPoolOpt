# Gas Fee Forecasting Improvements - Task List

## Project Overview
Update gas fee forecasting system based on experimental script analysis - remove model persistence, align modeling approach, and ensure eth_open data availability in data pipeline.

## Tasks

### 1. Remove model persistence in production
**Status:** TODO
**Assignee:** AI IDE Agent
**Priority:** 1

**Description:**
Remove all model saving functionality from forecast_gas_fees.py - no GCS or local model persistence needed for production

**Details:**
- Remove GCS bucket upload code
- Remove local model saving code
- Clean up model path handling
- Ensure no model artifacts are created in production

---

### 2. Remove backtesting functionality
**Status:** TODO
**Assignee:** AI IDE Agent
**Priority:** 2

**Description:**
Remove all backtesting and historical forecasting capabilities from the production system - focus only on forward-looking predictions

**Details:**
- Remove historical forecast generation
- Remove backtesting evaluation code
- Simplify forecasting to single-step ahead predictions
- Remove rolling forecast with refitting logic

---

### 3. Align modeling approach with experimental script
**Status:** TODO
**Assignee:** AI IDE Agent
**Priority:** 3

**Description:**
Update the XGBoost modeling approach to match the experimental script: use fixed lags of 7, implement Bayesian search with optuna (10 trials), and align hyperparameter search space with the experimental script's search_space function

**Details:**
- Change lags from default to fixed value of 7
- Replace current param_grid with optuna-based search_space function
- Increase n_trials from 5 to 10
- Align hyperparameter ranges with experimental script
- Implement proper optuna integration

---

### 4. Ensure eth_open data availability in data pipeline
**Status:** TODO
**Assignee:** AI IDE Agent
**Priority:** 4

**Description:**
Verify and ensure eth_open price data is available in the database schema and data ingestion pipeline. Check gas_fees_daily table schema and data ingestion scripts to confirm eth_open column exists and is populated. Update data pipeline if necessary to include eth_open data from CoinMarketCap or other sources.

**Details:**
- Check gas_fees_daily table schema for eth_open column
- Verify data ingestion scripts populate eth_open
- Update CoinMarketCap data fetching if needed
- Ensure eth_open data flows through the entire pipeline
- Validate data quality and completeness

---

### 5. Update feature engineering to use ETH open price
**Status:** TODO
**Assignee:** AI IDE Agent
**Priority:** 5

**Description:**
Replace lagged ETH close price features with ETH open price in feature engineering. Update data preprocessing to use eth_open instead of creating lag features from eth_close

**Details:**
- Modify data preprocessing to use eth_open directly
- Remove eth_close lag feature creation
- Update exogenous variable lists
- Ensure feature engineering aligns with experimental approach
- Test feature importance and model performance

---

## Implementation Notes

- All tasks should follow the Archon task-driven development workflow
- Tasks should be updated to "doing" when started, then "review" when completed
- Each task should include implementation notes and testing results
- Coordinate with data pipeline changes for eth_open availability
- Ensure backward compatibility where possible

## Dependencies

- Task 4 (eth_open data pipeline) should be completed before Task 5 (feature engineering)
- Tasks 1-3 can be implemented independently
- Model changes (Task 3) may require re-tuning after feature changes (Task 5)

## Success Criteria

- [ ] Model persistence completely removed from production
- [ ] Backtesting functionality eliminated
- [ ] Modeling approach matches experimental script
- [ ] eth_open data available in database
- [ ] Feature engineering uses eth_open instead of lagged eth_close
- [ ] Forecasting accuracy maintained or improved
- [ ] Production system remains stable