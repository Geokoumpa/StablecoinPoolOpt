from ortools.sat.python import cp_model
import pandas as pd
import numpy as np
import json
from datetime import datetime, date
from uuid import uuid4

from database.db_utils import get_db_connection

# Re-using data fetching functions from optimize_allocations.py
from asset_allocation.optimize_allocations import (
    fetch_pool_daily_metrics,
    fetch_daily_ledger,
    fetch_allocation_parameters,
    fetch_dynamic_lists,
    fetch_previous_allocation,
    should_force_reallocation,
    calculate_net_yield,
    store_allocation_parameters_snapshot,
    store_asset_allocations
)

def optimize_allocations_milp():
    """
    Orchestrates the asset allocation optimization process using a MILP model (CP-SAT).
    """
    print("Starting MILP asset allocation optimization...")
    conn = None
    try:
        conn = get_db_connection()
        
        # 1. Read forecasted APY and TVL for selected pools
        pool_metrics = fetch_pool_daily_metrics(conn)
        print(f"Fetched {len(pool_metrics)} pool daily metrics.")

        # 2. Read current account balances and liquidity portfolio
        current_ledger = fetch_daily_ledger(conn)
        print(f"Fetched {len(current_ledger)} current ledger entries.")

        # 3. Read configurable parameters from allocation_parameters
        alloc_params = fetch_allocation_parameters(conn)
        print(f"Fetched allocation parameters: {alloc_params}")

        # Prepare data for optimization
        if pool_metrics.empty:
            print("No pools available for optimization after filtering. Exiting.")
            return

        forecasted_apy = pool_metrics['forecasted_apy'].values
        forecasted_tvl = pool_metrics['forecasted_tvl'].values
        
        total_usd_value = current_ledger['end_of_day_balance'].sum() if not current_ledger.empty else 100000 # Example default

        num_pools = len(pool_metrics)

        model = cp_model.CpModel()

        # Define variables
        # x[i] = allocation percentage for pool i (continuous variable, scaled to integer for CP-SAT)
        # We'll scale by a large factor to maintain precision, e.g., 1,000,000
        scaling_factor = 1_000_000
        x_scaled = [model.NewIntVar(0, scaling_factor, f'x_scaled_{i}') for i in range(num_pools)]
        
        # z[i] = binary variable, 1 if pool i is selected, 0 otherwise
        z = [model.NewBoolVar(f'z_{i}') for i in range(num_pools)]

        # Constraints
        # 1. Sum of Weights Equals 1 (Full Allocation)
        model.Add(sum(x_scaled) == scaling_factor)

        # 2. Allocation Amounts Less Than or Equal to Per-Pool TVL Limit (`tvl_limit_percentage`)
        # x_scaled[i] / scaling_factor * total_usd_value <= forecasted_tvl[i] * tvl_limit_percentage
        # x_scaled[i] * total_usd_value <= forecasted_tvl[i] * tvl_limit_percentage * scaling_factor
        tvl_limit_percentage = alloc_params.get('tvl_limit_percentage', 1.0)
        for i in range(num_pools):
            model.Add(x_scaled[i] * int(total_usd_value) <= int(forecasted_tvl[i] * tvl_limit_percentage * scaling_factor))

        # 3. Binary Variable Linkage (Pool Selection)
        # If x_scaled[i] > 0, then z[i] must be 1.
        # If z[i] = 0, then x_scaled[i] must be 0.
        # This can be done by: x_scaled[i] <= M * z[i] and x_scaled[i] >= epsilon * z[i]
        # For CP-SAT, we can link directly:
        for i in range(num_pools):
            # If z[i] is 0, x_scaled[i] must be 0.
            model.Add(x_scaled[i] == 0).OnlyEnforceIf(z[i].Not())
            # If x_scaled[i] > 0, z[i] must be 1.
            # This is implicitly handled by the objective and other constraints if x_scaled[i] is non-negative
            # and we want to maximize yield. A more explicit way:
            model.Add(x_scaled[i] >= 1).OnlyEnforceIf(z[i]) # If z[i] is 1, x_scaled[i] must be at least 1 (smallest unit)

        # 4. Minimum Number of Pools (`min_pools`)
        min_pools = alloc_params.get('min_pools', 4)
        if alloc_params.get('profit_optimization', False):
            min_pools = alloc_params.get('min_pools', 1) # Relaxed for profit optimization
        model.Add(sum(z) >= min_pools)

        # 5. Maximum Allocation Percentage (`max_alloc_percentage`, conditional on `profit_optimization`)
        if not alloc_params.get('profit_optimization', False):
            max_alloc_percentage = alloc_params.get('max_alloc_percentage')
            if max_alloc_percentage is not None:
                for i in range(num_pools):
                    model.Add(x_scaled[i] <= int(max_alloc_percentage * scaling_factor))

        # 6. Position Limits (`position_max_pct_total_assets`, `position_max_pct_pool_tvl`)
        position_max_pct_total_assets = alloc_params.get('position_max_pct_total_assets')
        if position_max_pct_total_assets is not None:
            for i in range(num_pools):
                model.Add(x_scaled[i] * int(total_usd_value) <= int(position_max_pct_total_assets * total_usd_value * scaling_factor))

        position_max_pct_pool_tvl = alloc_params.get('position_max_pct_pool_tvl')
        if position_max_pct_pool_tvl is not None:
            for i in range(num_pools):
                model.Add(x_scaled[i] * int(total_usd_value) <= int(position_max_pct_pool_tvl * forecasted_tvl[i] * scaling_factor))

        # Group Allocation Limits (Removed as pool_group column is not available)
        # This section will be re-evaluated if pool grouping logic is re-introduced with a different schema.

        # Objective Function: Maximize (daily_yield * total_usd - total_gas_fees * total_usd - total_conversion_penalty * total_usd)
        # Convert APY to daily yield and scale for integer arithmetic
        daily_apys_scaled = [int(apy * scaling_factor / 365) for apy in forecasted_apy] # APY is annual, convert to daily
        
        # Simplified gas fees and conversion penalties, scaled
        gas_fee_rate = alloc_params.get('gas_fee_rate', 0.001)
        conversion_rate = alloc_params.get('conversion_rate', 0.0005)

        # Total yield from allocations
        total_yield_scaled = sum(x_scaled[i] * daily_apys_scaled[i] for i in range(num_pools))

        # Total costs (scaled)
        # Assuming gas and conversion fees are applied to the total allocated amount (total_usd_value)
        # This needs to be refined if fees are per-pool or per-transaction.
        total_gas_fees_scaled = int(gas_fee_rate * scaling_factor) * scaling_factor # scaled_rate * total_scaled_amount
        total_conversion_penalty_scaled = int(conversion_rate * scaling_factor) * scaling_factor

        # Objective: Maximize (total_yield_scaled - total_gas_fees_scaled - total_conversion_penalty_scaled)
        # Note: The total_usd_value factor is implicitly handled by the scaling and the objective structure.
        # If total_usd_value is large, it might cause overflow with integer scaling.
        # A better approach for objective might be to maximize net yield percentage.
        
        # For now, let's maximize the scaled net yield
        # The objective function in CP-SAT must be linear.
        # Maximize sum(x_scaled[i] * daily_apys_scaled[i]) - (total_gas_fees_scaled + total_conversion_penalty_scaled)
        # This is still a simplification. The original objective was:
        # Maximize (daily_yield * total_usd - total_gas_fees * total_usd - total_conversion_penalty * total_usd)
        # Let's try to represent this as accurately as possible with integer variables.
        
        # Net yield in USD terms, scaled
        # daily_yield_usd_scaled = sum(x_scaled[i] * daily_apys_scaled[i] * total_usd_value / scaling_factor for i in range(num_pools))
        # This would require floating point, which CP-SAT doesn't directly support in objective.
        # We need to maximize the scaled daily yield, and subtract scaled costs.
        
        # Let's redefine the objective to maximize the scaled daily yield, and handle costs separately if they are fixed.
        # If costs are proportional to allocation, they can be part of the coefficients.
        
        # For simplicity, let's maximize the sum of (scaled_apy - scaled_fees_per_unit) * x_scaled
        # This requires a more detailed breakdown of fees per pool.
        
        # Reverting to a simpler objective for initial MILP implementation: Maximize total scaled APY
        # The original objective was: Maximize (daily_yield * total_usd - total_gas_fees * total_usd - total_conversion_penalty * total_usd)
        # Let's try to approximate this by maximizing the sum of (forecasted_apy - effective_cost_rate) * x
        
        # For CP-SAT, the objective must be a sum of integer variables.
        # Let's define the objective as maximizing the total scaled daily yield.
        # The costs (gas, conversion) are tricky with binary variables if they are fixed per transaction.
        # If they are proportional to allocation, they can be incorporated into the APY.
        
        # For now, let's maximize the total scaled APY, and acknowledge that fees need more complex modeling.
        # This is a temporary simplification to get the MILP structure working.
        
        # Objective: Maximize sum(x_scaled[i] * daily_apys_scaled[i])
        # This is equivalent to maximizing the total daily yield percentage.
        model.Maximize(sum(x_scaled[i] * daily_apys_scaled[i] for i in range(num_pools)))


        # Solve the model
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        print(f"Optimization status: {solver.StatusName(status)}")
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print(f"Optimal value: {solver.ObjectiveValue()}")
            
            allocations_data = []
            for i in range(num_pools):
                allocated_percentage = solver.Value(x_scaled[i]) / scaling_factor
                if allocated_percentage > 1e-6: # Filter out very small allocations
                    allocations_data.append({
                        'pool_id': pool_metrics['pool_id'].iloc[i],
                        'allocated_percentage': allocated_percentage,
                        'allocated_usd': allocated_percentage * total_usd_value,
                        'forecasted_apy': forecasted_apy[i] # Include APY for net yield calculation
                    })
            allocations = pd.DataFrame(allocations_data)
            print("Optimal Allocations (MILP):")
            print(allocations)
        else:
            print("No optimal or feasible solution found.")
            allocations = pd.DataFrame() # Empty DataFrame if no solution

        # The rest of the logic (reallocation decision, storing snapshots/results)
        # can largely remain the same, using the 'allocations' DataFrame generated here.
        
        # Calculate net forecasted yield for the new optimal allocation
        net_forecasted_yield_new_allocation = 0.0
        if not allocations.empty:
            net_forecasted_yield_new_allocation = calculate_net_yield(
                allocations, total_usd_value,
                alloc_params.get('conversion_rate', 0.0004),
                alloc_params.get('gas_fee_rate', 0.001)
            )
            print(f"Net forecasted yield for new allocation: {net_forecasted_yield_new_allocation:.4f} USD")

        # Calculate net yield from existing allocation
        previous_allocation = fetch_previous_allocation(conn)
        print(f"Fetched previous allocation: {len(previous_allocation)} entries.")
        net_forecasted_yield_previous_allocation = 0.0
        if not previous_allocation.empty:
            net_forecasted_yield_previous_allocation = calculate_net_yield(
                previous_allocation, total_usd_value,
                alloc_params.get('conversion_rate', 0.0004),
                alloc_params.get('gas_fee_rate', 0.001)
            )
            print(f"Net forecasted yield for previous allocation: {net_forecasted_yield_previous_allocation:.4f} USD")

        # Force reallocation if any token, pool, or protocol is no longer approved/available
        dynamic_lists = fetch_dynamic_lists(conn)
        force_reallocate = should_force_reallocation(previous_allocation, dynamic_lists)

        final_allocations = pd.DataFrame()
        
        if force_reallocate or (net_forecasted_yield_new_allocation > net_forecasted_yield_previous_allocation and (status == cp_model.OPTIMAL or status == cp_model.FEASIBLE)):
            print("Decision: Full reallocation based on higher forecasted yield or forced reallocation.")
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                final_allocations = allocations
            else:
                print("No optimal/feasible solution found for full reallocation, but forced reallocation is active. Fallback strategy needed.")
                final_allocations = previous_allocation # For now, keep previous if no new optimal
        else:
            print("Decision: Reallocate only yesterday's yield.")
            yesterday_yield_row = current_ledger[current_ledger['token_symbol'] == 'TOTAL_YIELD']
            yesterday_realized_yield = yesterday_yield_row['end_of_day_balance'].iloc[0] if not yesterday_yield_row.empty else 0.0
            
            if yesterday_realized_yield > 0:
                print(f"Optimizing for yesterday's realized yield: {yesterday_realized_yield} USD")
                
                # Re-run optimization for only yesterday's yield using CP-SAT
                yield_model = cp_model.CpModel()
                yield_x_scaled = [yield_model.NewIntVar(0, scaling_factor, f'yield_x_scaled_{i}') for i in range(num_pools)]
                yield_z = [yield_model.NewBoolVar(f'yield_z_{i}') for i in range(num_pools)]

                yield_model.Add(sum(yield_x_scaled) == scaling_factor)
                for i in range(num_pools):
                    yield_model.Add(yield_x_scaled[i] * int(yesterday_realized_yield) <= int(forecasted_tvl[i] * tvl_limit_percentage * scaling_factor))
                    yield_model.Add(yield_x_scaled[i] == 0).OnlyEnforceIf(yield_z[i].Not())
                    yield_model.Add(yield_x_scaled[i] >= 1).OnlyEnforceIf(yield_z[i])
                
                yield_min_pools = alloc_params.get('min_pools', 4)
                if alloc_params.get('profit_optimization', False):
                    yield_min_pools = alloc_params.get('min_pools', 1)
                yield_model.Add(sum(yield_z) >= yield_min_pools)

                if not alloc_params.get('profit_optimization', False):
                    max_alloc_percentage = alloc_params.get('max_alloc_percentage')
                    if max_alloc_percentage is not None:
                        for i in range(num_pools):
                            yield_model.Add(yield_x_scaled[i] <= int(max_alloc_percentage * scaling_factor))

                yield_position_max_pct_total_assets = alloc_params.get('position_max_pct_total_assets')
                if yield_position_max_pct_total_assets is not None:
                    for i in range(num_pools):
                        yield_model.Add(yield_x_scaled[i] * int(yesterday_realized_yield) <= int(yield_position_max_pct_total_assets * yesterday_realized_yield * scaling_factor))

                yield_position_max_pct_pool_tvl = alloc_params.get('position_max_pct_pool_tvl')
                if yield_position_max_pct_pool_tvl is not None:
                    for i in range(num_pools):
                        yield_model.Add(yield_x_scaled[i] * int(yesterday_realized_yield) <= int(yield_position_max_pct_pool_tvl * forecasted_tvl[i] * scaling_factor))

                yield_model.Maximize(sum(yield_x_scaled[i] * daily_apys_scaled[i] for i in range(num_pools)))

                yield_solver = cp_model.CpSolver()
                yield_status = yield_solver.Solve(yield_model)

                if yield_status == cp_model.OPTIMAL or yield_status == cp_model.FEASIBLE:
                    print(f"Optimal yield reallocation value: {yield_solver.ObjectiveValue()}")
                    yield_allocations_data = []
                    for i in range(num_pools):
                        allocated_percentage = yield_solver.Value(yield_x_scaled[i]) / scaling_factor
                        if allocated_percentage > 1e-6:
                            yield_allocations_data.append({
                                'pool_id': pool_metrics['pool_id'].iloc[i],
                                'allocated_percentage': allocated_percentage,
                                'allocated_usd': allocated_percentage * yesterday_realized_yield,
                                'forecasted_apy': forecasted_apy[i]
                            })
                    yield_allocations = pd.DataFrame(yield_allocations_data)
                    print("Optimal Yield Allocations (MILP):")
                    print(yield_allocations)
                    
                    # This part needs careful handling: merge yield_allocations with previous_allocation
                    # For now, if yield optimization is successful, we consider it the final allocation.
                    # In a real system, you'd add this yield allocation to the existing portfolio.
                    final_allocations = yield_allocations 
                else:
                    print("No optimal/feasible solution found for yield reallocation. Keeping previous allocation.")
                    final_allocations = previous_allocation
            else:
                print("No yield generated yesterday or yield is zero. Keeping previous allocation.")
                final_allocations = previous_allocation

        # Store optimization parameters (snapshots)
        run_id = store_allocation_parameters_snapshot(conn, alloc_params, dynamic_lists)
        
        # Store allocation results
        if not final_allocations.empty:
            store_asset_allocations(conn, run_id, final_allocations)
        else:
            print("No final allocations to store.")

    except Exception as e:
        print(f"An error occurred during MILP optimization: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

    print("MILP Asset allocation optimization completed.")

if __name__ == "__main__":
    optimize_allocations_milp()