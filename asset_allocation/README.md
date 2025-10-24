# Asset Allocation Optimization

## Overview

This module implements a **transaction-aware optimization algorithm** for allocating assets to stablecoin liquidity pools. It uses convex optimization (CVXPY) with multiple solver options to maximize daily yield while accounting for gas fees, conversion costs, and various portfolio constraints.

## Key Features

### 1. Transaction-Level Modeling
- Models actual blockchain transactions: withdrawals, conversions, and allocations
- Accounts for gas fees per transaction type
- Tracks conversion fees between tokens
- Generates executable transaction sequences

### 2. Multi-Token Pool Support
- Handles pools with single tokens (e.g., "USDC") and multiple tokens (e.g., "DAI-USDC-USDT")
- Enforces even distribution requirements for multi-token pools (50-50 for pairs, 33-33-33 for triplets)
- Automatically converts tokens to achieve required pool balance
- Supports token normalization through mapping

### 3. Rebalancing Logic
- Considers existing allocations to pools
- Optimally withdraws and reallocates based on yield opportunities
- Balances rebalancing benefits against transaction costs

### 4. Comprehensive Constraint System
- Token balance conservation across all operations
- Maximum allocation per pool (configurable percentage of AUM)
- TVL limit constraint (percentage of pool's forecasted TVL)
- Multi-token pool even distribution
- Withdrawal limits (can't withdraw more than currently allocated)
- Total allocation cannot exceed total AUM

### 5. Data Quality Assessment
- Generates comprehensive data quality reports before optimization
- Checks for model feasibility and critical data issues
- Provides warnings for potential optimization problems

## Architecture

### Core Components

```
optimize_allocations.py
├── Data Loading Functions
│   ├── fetch_pool_data()          # Approved pools with forecasted APY
│   ├── fetch_token_prices()        # Latest token prices from OHLCV data
│   ├── fetch_gas_fee_data()        # Gas fees and ETH price
│   ├── fetch_current_balances()    # Warm wallet + allocated positions
│   └── fetch_allocation_parameters() # Configuration parameters
│
├── Helper Functions
│   ├── parse_pool_tokens()         # Extract tokens from pool symbol
│   ├── parse_pool_tokens_with_mapping() # Extract tokens with normalization
│   ├── calculate_aum()             # Calculate total AUM in USD
│   ├── build_token_universe()      # Build complete token set
│   ├── calculate_gas_fee_usd()     # Calculate gas fee in USD
│   └── calculate_transaction_gas_fees() # Calculate fees by transaction type
│
├── AllocationOptimizer Class
│   ├── __init__()                  # Initialize optimizer with data
│   ├── build_model()               # Construct CVXPY optimization model
│   ├── solve()                     # Solve optimization problem
│   ├── extract_results()           # Extract allocations and transactions
│   └── format_results()            # Format results for output
│
├── Result Persistence
│   ├── delete_todays_allocations() # Clear existing allocations
│   └── store_results()             # Persist results to database
│
└── Main Orchestration
    ├── optimize_allocations()      # Main entry point
    └── Data quality assessment     # Pre-optimization validation
```

### Decision Variables

The optimizer uses the following decision variables:

1. **`alloc[i,j]`**: Amount allocated to pool `i` of token `j` (target state)
2. **`withdraw[i,j]`**: Amount withdrawn from pool `i` of token `j`
3. **`convert[i,j]`**: Amount of token `i` converted to token `j`
4. **`final_warm_wallet[j]`**: Final warm wallet balance for token `j`
5. **`needs_conversion[i,j]`**: Binary variable indicating if conversion needed for allocation
6. **`has_allocation[i]`**: Binary variable indicating if pool has any allocation
7. **`is_withdrawal[i,j]`**: Binary variable indicating if withdrawal occurs
8. **`is_conversion[i,j]`**: Binary variable indicating if conversion occurs

### Objective Function

**Maximize**: Net Yield Improvement

```
Net Yield Improvement = New Yield - Yield Lost from Withdrawals

New Yield = Σ (allocation[i,j] × daily_apy[i,j] × price[j])
Yield Lost = Σ (withdrawal[i,j] × daily_apy[i,j] × price[j])

Transaction Costs:
  - Withdrawal Gas Costs = Σ(is_withdrawal[i,j] × withdrawal_gas_fee)
  - Allocation Gas Costs = Σ(needs_conversion[i,j] × allocation_gas_fee)
  - Conversion Costs = Σ(convert[i,j] × amount × conversion_rate × price)
  - Conversion Gas Costs = Σ(is_conversion[i,j] × conversion_gas_fee)

Net Objective = Net Yield Improvement - Total Transaction Costs
```

### Constraints

1. **Token Balance Conservation** (for each token j):
   ```
   initial_warm_wallet[j] + Σ withdrawals[i,j] + Σ conversions_in[:, j] = 
   final_warm_wallet[j] + Σ allocations[i,j] + Σ conversions_out[j, :]
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

6. **TVL Limit Constraint** (for each pool i):
   ```
   Σ (alloc[i,j] × price[j]) ≤ tvl_limit_percentage × pool_forecasted_tvl
   ```

7. **AUM Conservation Constraint**:
   ```
   Total Allocated + Transaction Costs + Final Warm Wallet ≤ Total AUM
   ```

## Data Flow

### Input Data Sources

| Data | Source Table | Query Filter | Purpose |
|------|--------------|--------------|---------|
| Pools | `pool_daily_metrics` + `pools` | `date = CURRENT_DATE AND is_filtered_out = FALSE` | Approved pools with forecasted APY |
| Token Prices | `raw_coinmarketcap_ohlcv` | Latest closing price per token | USD conversion and valuation |
| Gas Fees | `raw_coinmarketcap_ohlcv` | Latest ETH price + configured fees | Transaction cost calculation |
| Current Balances | `daily_balances` | `date = CURRENT_DATE` | Warm wallet + allocated positions |
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
    'amount_usd': float,
    'needs_conversion': bool
}
```

#### 2. Transaction Sequence
Ordered list of transactions to execute:

```python
{
    'seq': int,                    # Execution order
    'type': str,                   # WITHDRAWAL, CONVERSION, or ALLOCATION
    'from_location': str,          # pool_id or 'warm_wallet'
    'to_location': str,            # pool_id or 'warm_wallet'
    'token': str,                  # Token symbol (for non-conversions)
    'from_token': str,             # Source token (for conversions)
    'to_token': str,               # Target token (for conversions)
    'amount': float,               # Token amount
    'amount_usd': float,           # USD value
    'gas_cost_usd': float,         # Gas cost
    'conversion_cost_usd': float,  # Conversion fee (if applicable)
    'total_cost_usd': float,       # Total transaction cost
    'needs_conversion': bool        # Whether conversion needed
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
| `tvl_limit_percentage` | Maximum allocation as % of pool TVL | 0.05 (5%) |

## Gas Fee Calculation

The system uses a sophisticated gas fee model with different fees for different transaction types:

### Gas Fee Components
- **ETH Price**: Fetched from CoinMarketCap OHLCV data
- **Base Fee Transfer**: 10.0 Gwei (for transfers/deposits)
- **Base Fee Swap**: 30.0 Gwei (for token swaps)
- **Priority Fee**: 10.0 Gwei
- **Minimum Gas Units**: 21,000

### Gas Fee Formula
```
Gas Fee (USD) = Gas Units × (Base Fee + Priority Fee) × 1e-9 × ETH Price
```

### Transaction Type Fees
- **Pool Allocation/Withdrawal**: Uses transfer base fee
- **Token Conversion/Swap**: Uses swap base fee
- **General Transfers**: Uses transfer base fee
- **Deposits**: Uses transfer base fee

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
- Warm Wallet: USDC: 50,000, USDT: 30,000
- Pools: 4 approved pools with APYs ranging 4-6%
- Current Allocations: None

**Process:**
1. Data quality assessment validates input data
2. Optimizer evaluates all pools
3. Allocates to highest-yield pools within constraints
4. Converts tokens as needed
5. Generates transaction sequence

**Output:**
- Allocations to top 3-4 pools
- Conversion transactions (e.g., USDT → USDC)
- Allocation transactions to pools

### Scenario 2: Rebalancing with Existing Positions

**Input:**
- Warm Wallet: USDC: 10,000
- Current Allocations: Pool A (USDC: 15,000), Pool B (USDT: 10,000)
- New opportunity: Pool C with higher APY

**Process:**
1. Data quality assessment checks feasibility
2. Evaluates rebalancing Pool B → Pool C
3. Calculates gas + conversion costs
4. Only rebalances if net yield improvement exceeds costs

**Output:**
- Withdrawal from Pool B
- Conversion USDT → USDC (if needed)
- Allocation to Pool C

### Scenario 3: Multi-Token Pool Allocation

**Input:**
- Pool: "DAI-USDC-USDT" (3-token pool, 6% APY)
- Warm Wallet: USDC: 30,000

**Process:**
1. Determines need for DAI and USDT
2. Plans conversions: USDC → DAI, USDC → USDT
3. Ensures even 33-33-33 distribution
4. Applies TVL limit constraints

**Output:**
- Conversion: USDC → DAI (10,000)
- Conversion: USDC → USDT (10,000)
- Allocation: DAI (10,000), USDC (10,000), USDT (10,000) to pool

## Technical Notes

### Solver Requirements

- **Primary Solvers**: HIGHS, CBC, SCIPY (tried in order)
  - HIGHS: Best performance for MILP/convex problems
  - CBC: Reliable open-source fallback
  - SCIPY: Additional fallback option

- **Fallback Solver**: ECOS (used if primary solvers fail)
  - Suitable for smaller problems
  - Less efficient for large-scale optimizations

### Performance Considerations

- **Scalability**: Tested with up to 50 pools, 20 tokens
- **Solve Time**: Typically 1-5 seconds for standard problems
- **Memory**: ~100MB for typical problem sizes

### Data Quality Assessment

Before optimization, the system generates a comprehensive data quality report that includes:

- **Overall Quality Score**: 0-100 scale assessment
- **Abnormal Values Detection**: Identifies outliers and data issues
- **Model Feasibility Check**: Validates if optimization is possible
- **Critical Issues Warning**: Highlights problems that might prevent optimization

### Limitations & Future Enhancements

**Current Limitations:**
1. Gas fee modeled as fixed per transaction type (not per gas unit)
2. Discrete transaction counting approximated with continuous variables
3. Does not model slippage or market impact
4. Token price volatility not considered in optimization

**Planned Enhancements:**
1. Add yield reinvestment logic
2. Implement minimum holding periods
3. Add risk-adjusted optimization (Sharpe ratio)
4. Support for additional chains beyond Ethereum
5. Real-time rebalancing triggers
6. Dynamic gas fee forecasting
7. Slippage modeling for large transactions

## Troubleshooting

### Common Issues

**Issue**: Optimization fails with "infeasible" status
- **Cause**: Constraints too restrictive
- **Solution**: Check if `max_alloc_percentage` or `tvl_limit_percentage` is too low, or pool tokens unavailable

**Issue**: No allocations generated
- **Cause**: Transaction costs exceed potential yield
- **Solution**: Verify gas fees are reasonable, check if pools have sufficient APY

**Issue**: Data quality assessment fails
- **Cause**: Missing or invalid input data
- **Solution**: Check database connections, verify data freshness, ensure required tables exist

**Issue**: Solver not available
- **Cause**: Required solver not installed
- **Solution**: Install additional solvers or use fallback options

## References

- [CVXPY Documentation](https://www.cvxpy.org/)
- [HIGHS Solver](https://highs.dev/)
- [CBC Solver](https://github.com/coin-or/Cbc)
- Convex Optimization: Boyd & Vandenberghe

## Contact

For questions or issues, please contact the development team or create an issue in the project repository.