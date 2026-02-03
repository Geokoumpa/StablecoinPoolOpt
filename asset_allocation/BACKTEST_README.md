# Asset Allocation Optimization Backtesting

This directory contains tools for backtesting the asset allocation optimization algorithm over historical data.

## Overview

The `backtest_optimization.py` script simulates running the optimization algorithm over historical pool metrics data from the production database. It:

- Fetches pool metrics (forecasted APY and TVL) for each day in the specified period
- Simulates portfolio rebalancing starting with a specified AUM
- Tracks portfolio value, allocations, and transaction costs over time
- Captures and categorizes any optimization failures
- Generates comprehensive performance reports

## Usage

### Basic Usage

Run a backtest for January 2026 with default parameters (1M USDC starting AUM):

```bash
python asset_allocation/backtest_optimization.py
```

### Advanced Usage

Customize the backtest with command-line arguments:

```bash
python asset_allocation/backtest_optimization.py \
    --start-date 2026-01-01 \
    --end-date 2026-01-31 \
    --initial-aum 1000000 \
    --min-pools 4 \
    --max-alloc-pct 0.25 \
    --tvl-limit-pct 0.05 \
    --pool-tvl-limit 100000 \
    --optimization-horizon 30 \
    --output results/january_2026_backtest.json
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--start-date` | string | `2026-01-01` | Start date (YYYY-MM-DD format) |
| `--end-date` | string | `2026-01-31` | End date (YYYY-MM-DD format) |
| `--initial-aum` | float | `1,000,000` | Initial AUM in USD |
| `--min-pools` | int | `4` | Minimum number of pools required |
| `--max-alloc-pct` | float | `0.25` | Maximum allocation percentage per pool (e.g., 0.25 = 25%) |
| `--tvl-limit-pct` | float | `0.05` | TVL limit percentage per pool (e.g., 0.05 = 5% of pool TVL) |
| `--pool-tvl-limit` | float | `0` | Minimum pool TVL threshold in USD |
| `--optimization-horizon` | int | `30` | Optimization horizon in days |
| `--output` | string | `backtest_results.json` | Output file path for results |

## Output

The script generates two outputs:

### 1. Console Output

Real-time progress and a summary printed to console:

```
================================================================================
BACKTEST SUMMARY
================================================================================

Period:
  Start: 2026-01-01
  End: 2026-01-31
  Total Days: 31

AUM:
  Initial: $1,000,000.00
  Final: $1,015,234.56

Performance:
  Total Return: 1.52%
  Avg Projected APY: 8.45%
  Total Transaction Costs: $234.56
  Avg Num Pools: 5.2

Success Rate:
  Successful Days: 29
  Failed Days: 2
  Success Rate: 93.5%

Failures by Category:
  no_pools: 1
  solver_infeasible: 1

Failure Details:
  2026-01-15: No pools available for optimization
  2026-01-22: Solver failed. Eligible pools (TVL >= $0): 2, Required: 4
```

### 2. JSON Results File

Detailed results saved to JSON with the following structure:

```json
{
  "summary": {
    "period": {
      "start_date": "2026-01-01",
      "end_date": "2026-01-31",
      "total_days": 31
    },
    "initial_aum": 1000000.0,
    "final_aum": 1015234.56,
    "performance": {
      "total_return_pct": 1.52,
      "avg_projected_apy": 8.45,
      "total_transaction_costs": 234.56,
      "avg_num_pools": 5.2
    },
    "success_rate": {
      "successful_days": 29,
      "failed_days": 2,
      "success_rate_pct": 93.5
    }
  },
  "failures": {
    "by_category": {
      "no_pools": 1,
      "solver_infeasible": 1
    },
    "details": [
      {
        "date": "2026-01-15",
        "success": false,
        "error": "No pools available for optimization",
        "error_category": "no_pools"
      },
      {
        "date": "2026-01-22",
        "success": false,
        "error": "Solver failed. Eligible pools (TVL >= $0): 2, Required: 4",
        "error_category": "solver_infeasible",
        "eligible_pools": 2,
        "required_pools": 4,
        "pool_tvl_limit": 0
      }
    ]
  },
  "daily_results": [
    {
      "date": "2026-01-01",
      "metrics": {
        "total_aum": 1000456.78,
        "allocated_value": 950000.00,
        "wallet_value": 50456.78,
        "transaction_costs": 12.34,
        "projected_apy": 8.5,
        "num_pools": 5,
        "num_transactions": 12
      },
      "num_allocations": 5,
      "num_transactions": 12
    }
  ]
}
```

## Understanding Failures

The script categorizes optimization failures into several types:

### Error Categories

1. **`no_pools`**: No pools available for optimization on that date
   - Possible causes: No data in database, all pools filtered out, data quality issues

2. **`optimizer_init`**: Failed to initialize the optimizer
   - Possible causes: Invalid parameters, data format issues, missing required fields

3. **`solver_error`**: Solver threw an exception during solving
   - Possible causes: Numerical issues, constraint conflicts, OR-Tools internal errors

4. **`solver_infeasible`**: Solver could not find a feasible solution
   - Possible causes: Min pools requirement > eligible pools, TVL constraints too strict
   - Details include `eligible_pools` vs `required_pools` count

5. **`result_extraction`**: Solver succeeded but result extraction failed
   - Possible causes: Data structure issues, unexpected solver output format

### Investigating Failures

When failures occur:

1. Check the error category to understand the type of failure
2. Review the failure details in the JSON output
3. Check `backtest_optimization.log` for detailed error messages and stack traces
4. Verify pool metrics data exists for that date in the database:
   ```sql
   SELECT COUNT(*)
   FROM pool_daily_metrics
   WHERE date = '2026-01-15'
     AND is_filtered_out = FALSE;
   ```

5. If `solver_infeasible`:
   - Try reducing `--min-pools`
   - Try reducing `--pool-tvl-limit`
   - Try increasing `--tvl-limit-pct`

## Database Requirements

The backtest requires the following data in the database:

### Pool Metrics (`pool_daily_metrics` table)
- Forecasted APY (`forecasted_apy`)
- Forecasted TVL (`forecasted_tvl`)
- Filtering status (`is_filtered_out`, `filter_reason`)

### Pools (`pools` table)
- Pool metadata (symbol, chain, protocol)
- Underlying tokens (`underlying_tokens`)
- Active status (`is_active`)

### Token Prices (`raw_coinmarketcap_ohlcv` table)
- Latest token prices for stablecoins and ETH
- Note: Currently uses latest prices; historical price support can be added

### Gas Data
- Currently uses default gas fee estimates
- Historical gas data support can be added

## Examples

### Example 1: Quick Test (3 Days)

```bash
python asset_allocation/backtest_optimization.py \
    --start-date 2026-01-01 \
    --end-date 2026-01-03 \
    --output results/quick_test.json
```

### Example 2: Conservative Strategy

```bash
python asset_allocation/backtest_optimization.py \
    --start-date 2026-01-01 \
    --end-date 2026-01-31 \
    --min-pools 6 \
    --max-alloc-pct 0.15 \
    --tvl-limit-pct 0.03 \
    --pool-tvl-limit 500000 \
    --output results/conservative_january.json
```

### Example 3: Large Portfolio Test

```bash
python asset_allocation/backtest_optimization.py \
    --start-date 2026-01-01 \
    --end-date 2026-01-31 \
    --initial-aum 10000000 \
    --min-pools 8 \
    --max-alloc-pct 0.20 \
    --output results/large_portfolio.json
```

## Logging

Detailed logs are written to `backtest_optimization.log` in the current directory, including:

- Daily optimization progress
- Pool and token data loading
- Solver status and objective values
- Detailed error messages with stack traces
- State updates after each successful optimization

## Troubleshooting

### Issue: "No pools available for optimization"
**Solution**: Verify pool metrics data exists for the date range:
```sql
SELECT date, COUNT(*), AVG(forecasted_apy)
FROM pool_daily_metrics
WHERE date >= '2026-01-01' AND date <= '2026-01-31'
  AND is_filtered_out = FALSE
GROUP BY date;
```

### Issue: High failure rate
**Solutions**:
- Check if `min_pools` is too high for the available pools
- Try reducing `pool_tvl_limit` to include more pools
- Review failure categories in the output
- Check logs for specific error messages

### Issue: Database connection errors
**Solution**: Ensure Cloud SQL proxy is running and database credentials are configured in `.env` file

### Issue: Import errors
**Solution**: Ensure you're running from the project root directory:
```bash
cd /path/to/StablecoinPoolOpt
python asset_allocation/backtest_optimization.py
```

## Future Enhancements

Potential improvements to the backtesting script:

1. **Historical Price Data**: Use actual historical token prices instead of latest prices
2. **Historical Gas Data**: Use actual historical gas fees
3. **Realized Yield Comparison**: Compare forecasted APY vs actual realized yield
4. **Rolling Returns**: Calculate rolling 7-day, 30-day returns
5. **Risk Metrics**: Add volatility, max drawdown, Sharpe ratio calculations
6. **Visualization**: Generate charts for portfolio value, allocations over time
7. **Parallel Execution**: Run multiple backtests with different parameter combinations
8. **Transaction Simulation**: More realistic transaction execution with slippage

## Contact

For questions or issues with the backtesting script, please check the logs first, then reach out to the development team.
