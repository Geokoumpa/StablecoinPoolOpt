# Optimization Specification

This document describes the mathematical model for the Asset Allocation Optimization, based on the existing `cvxpy` implementation.

## Sets and Indices

*   $P$: Set of candidate pools, indexed by $i = 1, \dots, N_{pools}$.
*   $T$: Set of all unique tokens (from pools, warm wallet, and current allocations), indexed by $j = 1, \dots, N_{tokens}$.

## Parameters

*   $ForecastedAPY_{i}$: Forecasted annual percentage yield for pool $i$.
*   $Price_{j}$: Current USD price of token $j$.
*   $CurrentAlloc_{i,j}$: Current amount of token $j$ allocated to pool $i$.
*   $WarmWallet_{j}$: Current amount of token $j$ in the warm wallet.
*   $PoolTokens_{i}$: Set of tokens supported by pool $i$.
*   $PoolForecastedTVL_{i}$: Forecasted TVL for pool $i$.

### Cost Parameters
*   $Gas_{alloc}$: Gas cost for allocation (deposit) transaction ($).
*   $Gas_{withdraw}$: Gas cost for withdrawal transaction ($).
*   $Gas_{convert}$: Gas cost for token swap transaction ($).
*   $Rate_{convert}$: Conversion fee rate (e.g., 0.0004 or 0.04%).

### Constraint Parameters
*   $MaxAlloc\%$: Maximum percentage of total AUM allowed in a single pool (e.g., 25%).
*   $TVLLimit\%$: Maximum percentage of a pool's TVL that our position can occupy (e.g., 5%).
*   $MinPools$: Minimum number of pools to invest in.
*   $PoolTVLMinLimit$: Absolute minimum TVL for a pool to be considered.

## Decision Variables

### Continuous Variables (Non-negative)
*   $x_{i,j}$: Final allocation of token $j$ in pool $i$.
*   $w_{i,j}$: Amount of token $j$ withdrawn from pool $i$.
*   $d_{i,j}$: Amount of token $j$ deposited into pool $i$.
*   $c_{j,k}$: Amount of token $j$ converted to token $k$.
*   $y_{j}$: Final amount of token $j$ in the warm wallet.

### Binary Variables
*   $h_{i}$: 1 if pool $i$ has any allocation in the final state, 0 otherwise.
*   $has\_w_{i,j}$: 1 if any withdrawal occurs from pool $i$ for token $j$.
*   $has\_d_{i,j}$: 1 if any deposit occurs to pool $i$ for token $j$.
*   $is\_conv_{j,k}$: 1 if any conversion occurs from token $j$ to token $k$.

## Objective Function

Maximize **Net Annualized Yield**:

$$
\text{Maximize } \quad (Yield_{annual} - Cost_{total})
$$

Where:
$$
Yield_{daily} = \sum_{i \in P} \sum_{j \in T} x_{i,j} \cdot \frac{ForecastedAPY_{i}}{100 \cdot 365} \cdot Price_{j}
$$
$$
Yield_{annual} = Yield_{daily} \cdot 365
$$

$$
Cost_{total} = \sum_{i,j} (has\_w_{i,j} \cdot Gas_{withdraw}) + \sum_{i,j} (has\_d_{i,j} \cdot Gas_{alloc}) + \sum_{j,k} (is\_conv_{j,k} \cdot Gas_{convert}) + \sum_{j,k} (c_{j,k} \cdot Price_{j} \cdot Rate_{convert})
$$

## Constraints

### 1. Flow Conservation (Pools)
For each pool $i$ and token $j$:
$$
x_{i,j} = CurrentAlloc_{i,j} + d_{i,j} - w_{i,j}
$$

### 2. Flow Conservation (Warm Wallet)
For each token $j$:
$$
WarmWallet_{j} + \sum_{i} w_{i,j} + \sum_{k} c_{k,j} = y_{j} + \sum_{i} d_{i,j} + \sum_{k} c_{j,k}
$$
*(Inflow from initial wallet + withdrawals + received from conversions = Outflow to final wallet + deposits + sent to conversions)*

### 3. Withdrawal Limits
$$
w_{i,j} \le CurrentAlloc_{i,j}
$$

### 4. No Self-Conversion
$$
c_{j,j} = 0
$$

### 5. Equal Token Distribution (Multi-token Pools)
For pools with multiple tokens, the value allocated to each token must be equal in the final state:
$$
x_{i, j1} \cdot Price_{j1} = x_{i, j2} \cdot Price_{j2} \quad \forall j1, j2 \in PoolTokens_{i}
$$

### 6. Pool Allocation Limits
Total USD value in pool $i$: $V_i = \sum_{j} x_{i,j} \cdot Price_{j}$

*   **Max Allocation per Pool:**
    $$
    V_i \le TotalAUM \cdot MaxAlloc\%
    $$
*   **Pool TVL Limit:**
    $$
    V_i \le PoolForecastedTVL_{i} \cdot TVLLimit\%
    $$
*   **Absolute TVL Floor:**
    If $PoolForecastedTVL_{i} < PoolTVLMinLimit$, then $V_i = 0$.

### 7. AUM Conservation (Budget)
Total assets must remain constant (minus transaction costs):
$$
\sum_{i,j} (x_{i,j} \cdot Price_{j}) + \sum_{j} (y_{j} \cdot Price_{j}) + Cost_{total} \le TotalAUM
$$

### 8. Binary Constraints (Fixed Charge logic)
Using a large constant $M$:

*   **Pool Usage:** $V_i \le M \cdot h_{i}$ and $V_i \ge \epsilon \cdot h_{i}$
*   **Withdrawals:** $w_{i,j} \le M \cdot has\_w_{i,j}$
*   **Deposits:** $d_{i,j} \le M \cdot has\_d_{i,j}$
*   **Conversions:** $c_{j,k} \le M \cdot is\_conv_{j,k}$ and $c_{j,k} \ge \epsilon \cdot is\_conv_{j,k}$

### 9. Minimum Pools Constraint
$$
\sum_{i} h_{i} \ge MinPools
$$

### 10. Non-negativity
All continuous variables $\ge 0$.
