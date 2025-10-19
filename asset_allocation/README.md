# Asset Allocation Optimization

## Overview

This module implements a **transaction-aware optimization algorithm** for allocating assets to stablecoin liquidity pools. It uses convex optimization (CVXPY) with the GUROBI solver to maximize daily yield while accounting for gas fees, conversion costs, and various portfolio constraints.

## Key Features

### 1. Transaction-Level Modeling
- Models actual blockchain transactions: withdrawals, conversions, and allocations
- Accounts for gas fees per transaction
- Tracks conversion fees between tokens
- Generates executable transaction sequences

### 2. Multi-Token Pool Support
- Handles pools with single tokens (e.g., "USDC") and multiple tokens (e.g., "DAI-USDC-USDT")
- Enforces even distribution requirements for multi-token pools (50-50 for pairs, 33-33-33 for triplets)
- Automatically converts tokens to achieve required pool balance

### 3. Rebalancing Logic
- Considers existing allocations to pools
- Optimally withdraws and reallocates based on yield opportunities
- Balances rebalancing benefits against transaction costs

### 4. Comprehensive Constraint System
- Token balance conservation across all operations
- Maximum allocation per pool (configurable percentage of AUM)
- Multi-token pool even distribution
- Withdrawal limits (can't withdraw more than currently allocated)
- Total allocation cannot exceed total AUM

## Architecture

### Core Components

```
optimize_allocations.py
├── Data Loading Functions
│   ├── fetch_pool_data()          # Approved pools with forecasted APY
│   ├── fetch_token_prices()        # Latest token prices from OHLCV data
│   ├── fetch_gas_fee_data()        # Gas fees and ETH price
│   ├── fetch_current_balances()    # Cold wallet + allocated positions
│   └── fetch_allocation_parameters() # Configuration parameters
│
├── Helper Functions
│   ├── parse_pool_tokens()         # Extract tokens from pool symbol
│   ├── calculate_aum()             # Calculate total AUM in USD
│   └── build_token_universe()      # Build complete token set
│
├── AllocationOptimizer Class
│   ├── __init__()                  # Initialize optimizer with data
│   ├── build_model()               # Construct CVXPY optimization model
│   ├── solve()                     # Solve optimization problem
│   └── extract_results()           # Extract allocations and transactions
│
└── Main Orchestration
    ├── optimize_allocations()      # Main entry point
    └── store_results()             # Persist results to database
```

### Decision Variables

The optimizer uses the following decision variables:

1. **`alloc[i,j]`**: Amount allocated to pool `i` of token `j` (target state)
2. **`withdraw[i,j]`**: Amount withdrawn from pool `i` of token `j`
3. **`convert[i,j]`**: Amount of token `i` converted to token `j`
4. **`final_cold_wallet[j]`**: Final cold wallet balance for token `j`

### Objective Function

**Maximize**: Daily Net Yield

```
Daily Yield (USD) = Σ (allocation[i,j] × daily_apy[i,j] × price[j])
  where daily_apy = forecasted_apy / 365

Transaction Costs:
  - Gas Costs = Σ(transactions) × gas_fee_usd
  - Conversion Costs = Σ(conversions) × amount × conversion_rate × price

Net Objective = Daily Yield - Gas Costs - Conversion Costs
```

### Constraints

1. **Token Balance Conservation** (for each token j):
   ```
   initial_cold_wallet[j] + Σ withdrawals[i,j] + Σ conversions_in[:, j] = 
   final_cold_wallet[j] + Σ allocations[i,j] + Σ conversions_out[j, :]
   ```

2. **Withdrawal Limits** (for each pool i, token j):
   ```
   withdraw[i,j] ≤ current_allocation[i,j]
   ```

3. **No Self-Conversion** (for each token j):
   ```
   convert[j,j] = 0
   ```

4. **Multi-Token Pool Even Distribution** (for pools with N tokens):
   ```
   alloc[i,j1] × price[j1] = alloc[i,j2] × price[j2] = ... = alloc[i,jN] × price[jN]
   ```

5. **Maximum Allocation Per Pool** (for each pool i):
   ```
   Σ (alloc[i,j] × price[j]) ≤ max_alloc_percentage × total_aum
   ```

6. **Total Allocation Limit**:
   ```
   Σ (alloc[i,j] × price[j]) ≤ total_aum
   ```

## Data Flow

### Input Data Sources

| Data | Source Table | Query Filter | Purpose |
|------|--------------|--------------|---------|
| Pools | `pool_daily_metrics` + `pools` | `date = CURRENT_DATE AND is_filtered_out = FALSE` | Approved pools with forecasted APY |
| Token Prices | `raw_coinmarketcap_ohlcv` | Latest closing price per token | USD conversion and valuation |
| Gas Fees | `gas_fees_daily` | `date = CURRENT_DATE` | Transaction cost calculation |
| Current Balances | `daily_balances` | `date = CURRENT_DATE - 1` | Cold wallet + allocated positions |
| Parameters | `allocation_parameters` | Latest record | Configuration settings |

### Output Data

The optimizer produces two key outputs:

#### 1. Final Allocations
```python
{
    'pool_id': str,
    'pool_symbol': str,
    'token': str,
    'amount': float,
    'amount_usd': float
}
```

#### 2. Transaction Sequence
Ordered list of transactions to execute:

```python
{
    'seq': int,                    # Execution order
    'type': str,                   # WITHDRAWAL, CONVERSION, or ALLOCATION
    'from_location': str,          # pool_id or 'cold_wallet'
    'to_location': str,            # pool_id or 'cold_wallet'
    'token': str,                  # Token symbol
    'amount': float,               # Token amount
    'amount_usd': float,           # USD value
    'gas_cost_usd': float,         # Gas cost (if applicable)
    'conversion_fee_usd': float    # Conversion fee (if applicable)
}
```

### Database Storage

Results are persisted to:

1. **`allocation_parameters`**: Snapshot of parameters used
2. **`asset_allocations`**: Transaction sequence with run_id

## Configuration Parameters

Key parameters from `allocation_parameters` table:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_alloc_percentage` | Maximum allocation per pool as % of AUM | 0.20 (20%) |
| `conversion_rate` | Fee rate for token conversions | 0.0004 (0.04%) |
| `min_transaction_value` | Minimum transaction value in USD | 50.0 |

## Usage

### Running the Optimizer

```python
from asset_allocation.optimize_allocations import optimize_allocations

# Run optimization
optimize_allocations()
```

### Running Tests

```bash
# Set Python path and run tests
cd /home/geokoumpa/workspace/StablecoinPoolOpt
PYTHONPATH=$PWD python asset_allocation/test_optimization.py
```

### Integration with Pipeline

Add to `main_pipeline.py`:

```python
from asset_allocation.optimize_allocations import optimize_allocations

# In main pipeline
def run_allocation_optimization():
    logger.info("Running asset allocation optimization...")
    optimize_allocations()
```

## Example Scenarios

### Scenario 1: Fresh Allocation (No Existing Positions)

**Input:**
- Cold Wallet: USDC: 50,000, USDT: 30,000
- Pools: 4 approved pools with APYs ranging 4-6%
- Current Allocations: None

**Process:**
1. Optimizer evaluates all pools
2. Allocates to highest-yield pools within constraints
3. Converts tokens as needed
4. Generates transaction sequence

**Output:**
- Allocations to top 3-4 pools
- Conversion transactions (e.g., USDT → USDC)
- Allocation transactions to pools

### Scenario 2: Rebalancing with Existing Positions

**Input:**
- Cold Wallet: USDC: 10,000
- Current Allocations: Pool A (USDC: 15,000), Pool B (USDT: 10,000)
- New opportunity: Pool C with higher APY

**Process:**
1. Evaluates rebalancing Pool B → Pool C
2. Calculates gas + conversion costs
3. Only rebalances if net yield improvement exceeds costs

**Output:**
- Withdrawal from Pool B
- Conversion USDT → USDC (if needed)
- Allocation to Pool C

### Scenario 3: Multi-Token Pool Allocation

**Input:**
- Pool: "DAI-USDC-USDT" (3-token pool, 6% APY)
- Cold Wallet: USDC: 30,000

**Process:**
1. Determines need for DAI and USDT
2. Plans conversions: USDC → DAI, USDC → USDT
3. Ensures even 33-33-33 distribution

**Output:**
- Conversion: USDC → DAI (10,000)
- Conversion: USDC → USDT (10,000)
- Allocation: DAI (10,000), USDC (10,000), USDT (10,000) to pool

## Technical Notes

### Solver Requirements

- **Primary Solver**: GUROBI (commercial, requires license)
  - Best performance for MILP/convex problems
  - Handles large-scale optimizations efficiently

- **Fallback Solver**: ECOS (open-source)
  - Used if GUROBI unavailable
  - Suitable for smaller problems

### Performance Considerations

- **Scalability**: Tested with up to 50 pools, 20 tokens
- **Solve Time**: Typically 1-5 seconds for standard problems
- **Memory**: ~100MB for typical problem sizes

### Limitations & Future Enhancements

**Current Limitations:**
1. Gas fee modeled as fixed per transaction (not per gas unit)
2. Discrete transaction counting approximated with continuous variables
3. Does not model slippage or market impact

**Planned Enhancements:**
1. Add yield reinvestment logic
2. Implement minimum holding periods
3. Add risk-adjusted optimization (Sharpe ratio)
4. Support for additional chains beyond Ethereum
5. Real-time rebalancing triggers

## Troubleshooting

### Common Issues

**Issue**: Optimization fails with "infeasible" status
- **Cause**: Constraints too restrictive
- **Solution**: Check if `max_alloc_percentage` is too low, or pool tokens unavailable

**Issue**: No allocations generated
- **Cause**: Transaction costs exceed potential yield
- **Solution**: Verify gas fees are reasonable, check if pools have sufficient APY

**Issue**: GUROBI not available
- **Cause**: License not configured
- **Solution**: Model falls back to ECOS; consider installing GUROBI for better performance

## References

- [CVXPY Documentation](https://www.cvxpy.org/)
- [GUROBI Solver](https://www.gurobi.com/)
- Convex Optimization: Boyd & Vandenberghe

## Contact

For questions or issues, please contact the development team or create an issue in the project repository.