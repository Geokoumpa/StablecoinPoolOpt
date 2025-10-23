The goal is to build an optimization algorithm that runs daily and allocates the available assets under management to stablecoin pools.

# Input 

The algorithm should take the following information as input:

* The list of stablecoin pools with their forecasted APY, pool_id and symbol from which the list of token can be extracted, i.e. {pool_id: "aa70268e-4b52-42bf-a116-608b370f9501", symbol: "USDC", forecasted_apy: "3.8889" }
* The list of allocations already done in previous iterations
* The list of balances of various tokens currently in the cold wallet, unallocated and available for investment
* The OHLCV data of the tokens used, i.e. {"symbol": "USDC", "low": 0.9997491162438754, "high": 1.0006105082622172, "open": 0.99991808336297, "close": 1.0002087128202493, "volume": 4779945985.14, "timestamp": "2025-01-04T23:59:59.999Z", "market_cap": 45554906948.15}
* The forecasted gas fee for transactions on the ethereum chain

# General rules

* Each pool can contain 1 or more tokens, this can be derived by its symbol, tokens will be split by "-", i.e. "DAI-USDC-USDT" contains 3 tokens, "GTUSDC" contains 1.
* Pools that contain more than 1 tokens have an even distribution (i.e. 50% - 50% for pairs) which must be followed when allocating assets.
* Conversions from and to various tokens cost a 0.0004 conversion rate
* All transactions (allocations to / withdrawals from pools, conversions etc.) cost a gas fee
* Yield reinvestment will initially be excluded from the model logic and will be handled manually
* The objective is to maximize the overall yield, taking into account conversion fees and gas fees
* Due the fact that one of the previous pipeline steps is to filter out pools not meeting a number  of criteria, it is possible that a pool becomes unapproved  while assets are already allocated to it.
* When allocating to a pool, the formula is:

    * Cost = (amount * conversion_rate) + 2 * gas_fee  (when the token needs to be converted)
    * Cost = (amount * conversion_rate) + gas_fee  (when we already have sufficient amount of that token on the cold wallet)

* Withdrawing cost formula is Cost = amount * conversion_rate + gas_fee (because we don't convert until it is needed during next allocation)

# Output

The model should generate, print and persist:

1) A final allocation of assets, both to pools and their respective tokens, as well as tokens to be unallocated in the cold wallet
2) An ordered list of all the transactions (transfers and conversions), with specific amounts that need to be followed in order to end up with the suggested allocation. Since we don't have the pool's on chain addresses, use pool_ids and 'cold_wallet' to indicate accordingly

# Tech

* Use convex optimization, cvxpy library and GUROBI solver. Suggest, if based on the nature of the optimization problem, a different modelling approach (MILP, CP or other), or python libary, would be a better match.
* Build the model taking into account the actual transactions that will be taking place and accounting for gas fees anc converstions, instead of just using balances of tokens and only considering final allocations
* Consider checking the database/schema folder to understand how to load data required as input, as well as the rest of the data pipeline scripts to understand its logic. Use db_utils.py for database connection
    * Use `pool_daily_metrics` filtering with the latest (current) date, is_filtered_out = false  for the forecasted APY and `pools` for the general pool data. The `pool_daily_metrics` is_filtered_out is more reliable than `pools` currently_filtered_out
    * Use `raw_coinmarketcap_ohlcv` for the tokens' OHLCV data
    * Use `gas_fees_daily` to get forecasted_avg_gas_gwei and forecasted_max_gas_gwei
    * current allocations data management is currently work in progress, so suggest a data structure and mock a data sample for now
    * Final allocations data management (the model's output) will also be determined by this development. Suggest a structure


