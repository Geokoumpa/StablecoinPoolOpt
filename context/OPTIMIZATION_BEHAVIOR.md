# Optimization Algorithm Behavior Documentation

## Overview

The `AllocationOptimizer` in `optimize_allocations.py` uses a Mixed Integer Linear Programming (MILP) approach to maximize portfolio yield while minimizing transaction costs. This document explains some non-obvious behaviors that may appear suboptimal but are actually globally optimal.

## Key Insight: Interconnected Token Flows

**The optimizer considers all token flows simultaneously.** When analyzing a single allocation (e.g., "why didn't the optimizer put more USDT in HWHLP?"), you cannot evaluate it in isolation because:

1. Token conversions are shared resources
2. Changing one allocation affects the feasibility of others
3. The 25% max allocation per pool constraint creates competition

### Example Case Study (December 2025)

**Observation:** The optimizer allocated only $456.67 USDT to HWHLP (10.26% APY) and converted $742.33 USDT to USDS for a lower-APY pool (4.59% APY).

**Initial Analysis (Incorrect):**
- HWHLP could have received up to $1,052 (25% of AUM)
- Putting all $1,199 USDT into HWHLP would yield ~$123/year
- Converting to USDS and allocating there yields only ~$34/year
- This appears to be a ~$46/year suboptimal decision

**Why This Analysis is Wrong:**

The token flows are interconnected:

```
Current Allocations:
- Pool c743eac6: 1199 USDT (to be withdrawn)
- Pool 9b9e6e11: 979 USDF (to be withdrawn)
- Pool 09348d64: 1003 USDC (to be withdrawn)
- Pool a948bde2: 1014 USDe (to be kept)
- Pool 7501ef09: 16.7 USDS (to be kept)
```

**Original Solution ($420.70/year net):**
| Pool | Token | Amount | APY | Yield/Year |
|------|-------|--------|-----|------------|
| HWHLP | USDT | $456.67 | 10.26% | $46.85 |
| USDS pool | USDS | $742.33 | 4.59% | $34.08 |
| CRVUSD | crvUSD | $974.55 | 9.84% | $95.90 |
| NBASIS | USDC | $1,002.56 | 12.34% | $123.70 |
| JRUSDE | USDe | $1,013.34 | 13.26% | $134.34 |
| USDS hold | USDS | $16.71 | 4.17% | $0.69 |
| **Transaction costs** | | | | -$14.38 |
| **NET** | | | | **$420.70** |

**Forced Solution (HWHLP ≥$1,000, resulting in $412.94/year net):**
| Pool | Token | Amount | APY | Yield/Year |
|------|-------|--------|-----|------------|
| HWHLP | USDT | $1,051.22 | 10.26% | $107.86 |
| BBQUSDT | USDT | $147.65 | 5.08% | $6.73 |
| SUSDF | USDF | $140.25 | 9.42% | $13.21 |
| USDS pool | USDS | $826.48 | 4.59% | $37.95 |
| NBASIS | USDC | $1,002.56 | 12.34% | $123.70 |
| JRUSDE | USDe | $1,013.34 | 13.26% | $134.34 |
| USDS hold | USDS | $16.47 | 4.17% | $0.69 |
| **Transaction costs** | | | | -$11.54 |
| **NET** | | | | **$412.94** |

**Key Difference:** In the forced solution, **CRVUSD is not available** because:
1. More USDT goes to HWHLP, so less USDT is converted to USDS
2. This creates pressure to use USDF for something else
3. USDF goes to SUSDF (9.42%) instead of being converted to crvUSD for CRVUSD pool (9.84%)
4. The remaining USDT that doesn't fit in HWHLP (25% cap) goes to BBQUSDT (only 5.08%)

**Result:** The "obvious" optimization (more to HWHLP) actually **loses $7.76/year** compared to the optimizer's solution.

## Why Intuition Fails

Human intuition tends to:
1. **Evaluate allocations in isolation** - "This pool has higher APY, put more there"
2. **Ignore token conversion constraints** - Converting tokens costs gas and has rate fees
3. **Miss opportunity costs** - Using tokens for one pool means they can't be used elsewhere

The optimizer:
1. **Considers all allocations simultaneously**
2. **Accounts for all transaction costs**
3. **Respects all constraints globally**

## Solver Configuration

The optimizer uses HiGHS solver with tight MIP gap tolerance:

```python
solver_options = {
    'mip_rel_gap': 0.0001,  # 0.01% relative gap
    'time_limit': 300.0,     # 5 minute limit
}
```

This ensures solutions are within 0.01% of theoretical optimum.

## Validation Method

To verify the optimizer is working correctly:

1. Note the reported objective value
2. Force a constraint that seems "obviously better"
3. Solve again and compare objective values
4. If the forced solution is worse, the optimizer was correct

```python
# Example: Force HWHLP to receive more USDT
new_constraint = optimizer.final_alloc[hwhlp_idx, usdt_idx] >= 1000
problem_with_constraint = cp.Problem(problem.objective, problem.constraints + [new_constraint])
problem_with_constraint.solve(solver=cp.HIGHS, verbose=False)
# If objective is worse, original solution was optimal
```

## Constraints That Affect Allocation Decisions

1. **Max allocation per pool (25% of AUM)** - Prevents concentration
2. **TVL limit (5% of pool TVL)** - Limits exposure to small pools
3. **Token balance conservation** - Tokenswithdrawal = deposit + conversion
4. **Min pools (4)** - Ensures diversification
5. **Binary transaction costs** - Gas is per-transaction, not per-amount

## Conclusion

When an allocation appears suboptimal, it's usually because:
- The optimizer is preserving access to better opportunities elsewhere
- Token flows are interconnected in non-obvious ways
- The "better" alternative would violate some constraint

**Trust the optimizer** unless you can demonstrate a feasible solution with higher objective value.

---
*Document created: 2025-12-17*
*Based on investigation of HWHLP vs USDS allocation behavior*
