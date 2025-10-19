Our pipeline in @main_pipeline.py currently has the manage_ledger step after the optimization step. We need to make a few changes.
The ledger must be updated before the optimization script, so that we know the assets that we have available for allocation, either already allocated or free. This change of order must also be reflected on the production pipeline described in @workflow and @workflow. 

Based on the information below, feel free to suggest changes in the daily_ledger table schema, or suggest a new table properly named.

The logic we must follow with the ledger is the following:
From the account_transactions table, we need to track the transactions that come from our cold wallet address, which will be provided as a new env var. These transactions will normally be in USDC, and will be considered new capital for investment.
From the account_transactions table, we need to track the transactions to one of the pools that are filtered during the prefiltering stage. We want to find transactions towards pools, that have no opposite transactions after them. This will give us the assets that are currently allocated in a pool (amount + token)
Because the pipeline runs each day, and a pool previously allowed, may be filtered out in the next day, we want to keep a boolean currently_filtered_out column in the pools table. This will turn to true only for the pools that are filtered out during the pre-filtering stage, not the ones in the final filtering, as that stage is more vollatile and we don't want to possibly exclude invested capital.

The end goal is for the optimization script to know:

wallet balance for each token currently unallocated
assets currently allocated in pools