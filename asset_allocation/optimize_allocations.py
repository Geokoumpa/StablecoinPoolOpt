import cvxpy as cp
import pandas as pd
import json
from datetime import datetime, date, timezone
from uuid import uuid4

from database.db_utils import get_db_connection

def fetch_pool_daily_metrics(conn):
    """Fetches forecasted APY and TVL for selected pools."""
    query = """
    SELECT
        pdm.pool_id,
        p.symbol,
        p.chain,
        p.protocol,
        pdm.forecasted_apy,
        pdm.forecasted_tvl,
        pdm.is_filtered_out,
        pdm.filter_reason,
        pdm.rolling_apy_7d,
        pdm.rolling_apy_30d,
        pdm.apy_delta_today_yesterday,
        pdm.stddev_apy_7d,
        pdm.stddev_apy_30d,
        pdm.stddev_apy_7d_delta,
        pdm.stddev_apy_30d_delta
    FROM pool_daily_metrics pdm
    JOIN pools p ON pdm.pool_id = p.pool_id
    WHERE pdm.date = CURRENT_DATE AND pdm.is_filtered_out = FALSE;
    """
    return pd.read_sql(query, conn)

def fetch_daily_ledger(conn):
    """Fetches current account balances and liquidity portfolio from the previous day."""
    query = """
    SELECT
        token_symbol,
        end_of_day_balance
    FROM daily_ledger
    WHERE date = CURRENT_DATE - INTERVAL '1 day';
    """
    return pd.read_sql(query, conn)

def fetch_allocation_parameters(conn):
    """Fetches configurable parameters for allocation."""
    query = """
    SELECT *
    FROM allocation_parameters
    ORDER BY timestamp DESC
    LIMIT 1;
    """
    params = pd.read_sql(query, conn)
    if not params.empty:
        return params.iloc[0].to_dict()
    return {}

def fetch_dynamic_lists(conn):
    """Fetches dynamic lists for snapshotting."""
    approved_tokens = pd.read_sql("SELECT token_symbol FROM approved_tokens;", conn)['token_symbol'].tolist()
    blacklisted_tokens = pd.read_sql("SELECT token_symbol FROM blacklisted_tokens;", conn)['token_symbol'].tolist()
    approved_protocols = pd.read_sql("SELECT protocol_name FROM approved_protocols;", conn)['protocol_name'].tolist()
    icebox_tokens = pd.read_sql("SELECT token_symbol FROM icebox_tokens;", conn)['token_symbol'].tolist()
    return {
        "approved_tokens": approved_tokens,
        "blacklisted_tokens": blacklisted_tokens,
        "approved_protocols": approved_protocols,
        "icebox_tokens": icebox_tokens,
    }

def fetch_previous_allocation(conn):
    """Fetches the most recent asset allocation."""
    query = """
    SELECT
        aa.pool_id,
        aa.allocated_amount_usd,
        aa.allocation_percentage,
        pdm.forecasted_apy -- Fetch forecasted APY for the previous allocation's pools
    FROM asset_allocations aa
    JOIN (
        SELECT run_id, MAX(timestamp) as max_timestamp
        FROM allocation_parameters
        GROUP BY run_id
        ORDER BY max_timestamp DESC
        LIMIT 1
    ) latest_run ON aa.run_id = latest_run.run_id
    LEFT JOIN pool_daily_metrics pdm ON aa.pool_id = pdm.pool_id AND pdm.date = CURRENT_DATE
    ;
    """
    return pd.read_sql(query, conn)

def should_force_reallocation(previous_allocation, dynamic_lists):
    """
    Checks if a full reallocation should be forced due to unapproved/unavailable assets.
    This is a simplified check. A more robust implementation would check against
    the actual state of pools (e.g., if a pool is no longer active or has drastically changed).
    """
    if previous_allocation.empty:
        return False

    approved_tokens = set(dynamic_lists.get("approved_tokens", []))
    blacklisted_tokens = set(dynamic_lists.get("blacklisted_tokens", []))
    approved_protocols = set(dynamic_lists.get("approved_protocols", []))
    icebox_tokens = set(dynamic_lists.get("icebox_tokens", []))

    # Check if any previously allocated pool's token is now blacklisted or in icebox
    # This requires joining with the 'pools' table to get token information
    # For now, we'll assume pool_id implies token. This needs refinement.
    # A more accurate check would involve fetching pool details (token, protocol)
    # and comparing against the dynamic lists.

    # Placeholder for a more detailed check:
    # For example, if a pool's underlying token is blacklisted:
    # for index, row in previous_allocation.iterrows():
    #     pool_id = row['pool_id']
    #     # Fetch token/protocol for pool_id from 'pools' table
    #     # if token in blacklisted_tokens or protocol not in approved_protocols:
    #     #     return True

    # For demonstration, let's assume if any pool in previous_allocation has a null forecasted_apy,
    # it means it's no longer available or relevant for today's forecast.
    if previous_allocation['forecasted_apy'].isnull().any():
        logger.info("Previous allocation contains pools with no current forecasted APY, forcing reallocation.")
        return True

    return False

def calculate_net_yield(allocations_df, total_usd_value, conversion_rate, gas_fee_rate):
    """
    Calculates the net forecasted yield for a given allocation.
    Simplified: assumes gas fees and conversion penalties are proportional to allocated amount.
    """
    if allocations_df.empty:
        return 0.0

    # Ensure 'forecasted_apy' is available in the DataFrame
    if 'forecasted_apy' not in allocations_df.columns:
        logger.warning("'forecasted_apy' not found in allocations_df. Cannot calculate net yield.")
        return 0.0

    daily_yield = (allocations_df['allocated_percentage'] * allocations_df['forecasted_apy']).sum()
    
    # Assuming gas fees and conversion penalties are applied to the total allocated amount
    total_gas_fees_usd = gas_fee_rate * total_usd_value
    total_conversion_penalty_usd = conversion_rate * total_usd_value

    net_yield_usd = (daily_yield * total_usd_value) - total_gas_fees_usd - total_conversion_penalty_usd
    return net_yield_usd

def store_allocation_parameters_snapshot(conn, alloc_params, dynamic_lists):
    """Stores a snapshot of allocation parameters and dynamic lists."""
    run_id = uuid4()
    cursor = conn.cursor()
    try:
        # Merge dynamic lists into alloc_params for storage
        alloc_params_to_store = alloc_params.copy()
        alloc_params_to_store['approved_tokens_snapshot'] = json.dumps(dynamic_lists.get('approved_tokens', []))
        alloc_params_to_store['blacklisted_tokens_snapshot'] = json.dumps(dynamic_lists.get('blacklisted_tokens', []))
        alloc_params_to_store['approved_protocols_snapshot'] = json.dumps(dynamic_lists.get('approved_protocols', []))
        alloc_params_to_store['icebox_tokens_snapshot'] = json.dumps(dynamic_lists.get('icebox_tokens', []))

        # Construct the INSERT query dynamically based on available keys in alloc_params_to_store
        columns = ["run_id", "timestamp"]
        values = [str(run_id), datetime.now()]
        
        # Add other parameters from alloc_params_to_store, ensuring they exist in the schema
        # This is a simplified approach; a more robust solution would validate against schema
        schema_columns = [
            'tvl_limit_percentage', 'max_alloc_percentage', 'conversion_rate', 'min_pools',
            'profit_optimization', 'approved_tokens_snapshot', 'blacklisted_tokens_snapshot',
            'approved_protocols_snapshot', 'icebox_tokens_snapshot', 'token_marketcap_limit',
            'pool_tvl_limit', 'pool_apy_limit', 'pool_pair_tvl_ratio_min', 'pool_pair_tvl_ratio_max',
            'group1_max_pct', 'group2_max_pct', 'group3_max_pct', 'position_max_pct_total_assets',
            'position_max_pct_pool_tvl', 'group1_apy_delta_max', 'group1_7d_stddev_max',
            'group1_30d_stddev_max', 'group2_apy_delta_max', 'group2_7d_stddev_max',
            'group2_30d_stddev_max', 'group3_apy_delta_min', 'group3_7d_stddev_min',
            'group3_30d_stddev_min', 'other_dynamic_limits'
        ]

        for col in schema_columns:
            if col in alloc_params_to_store:
                columns.append(col)
                val = alloc_params_to_store[col]
                # Handle boolean values for PostgreSQL
                if isinstance(val, bool):
                    values.append('TRUE' if val else 'FALSE')
                elif isinstance(val, dict) or isinstance(val, list):
                    values.append(json.dumps(val))
                else:
                    values.append(val)

        placeholders = ', '.join(['%s'] * len(columns))
        insert_query = f"INSERT INTO allocation_parameters ({', '.join(columns)}) VALUES ({placeholders});"
        
        cursor.execute(insert_query, values)
        conn.commit()
        logger.info(f"Allocation parameters snapshot stored with run_id: {run_id}")
        return run_id
    except Exception as e:
        conn.rollback()
        logger.error(f"Error storing allocation parameters snapshot: {e}")
        raise

def store_asset_allocations(conn, run_id, allocations_df):
    """Stores the asset allocation results with deduplication."""
    cursor = conn.cursor()
    try:
        from datetime import datetime
        current_date = datetime.now(timezone.utc).date()
        
        # Check if we already have allocations for today's run
        cursor.execute("""
            SELECT COUNT(*) FROM asset_allocations aa
            JOIN allocation_parameters ap ON aa.run_id = ap.run_id
            WHERE DATE(ap.timestamp) = %s;
        """, (current_date,))
        existing_count = cursor.fetchone()[0]
        
        if existing_count > 0:
            logger.info(f"Asset allocations already exist for today ({current_date}). Checking if update is needed...")
            
            # Delete existing allocations for today before inserting new ones
            cursor.execute("""
                DELETE FROM asset_allocations
                WHERE run_id IN (
                    SELECT run_id FROM allocation_parameters
                    WHERE DATE(timestamp) = %s
                );
            """, (current_date,))
            logger.info(f"Removed {cursor.rowcount} existing allocation records for today")
        
        # Insert new allocations
        for index, row in allocations_df.iterrows():
            insert_query = """
            INSERT INTO asset_allocations (run_id, pool_id, allocated_amount_usd, allocation_percentage)
            VALUES (%s, %s, %s, %s);
            """
            cursor.execute(insert_query, (
                str(run_id),
                row['pool_id'],
                row['allocated_usd'],
                row['allocated_percentage']
            ))
        
        conn.commit()
        logger.info(f"Asset allocations stored for run_id: {run_id} (total: {len(allocations_df)} allocations)")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Error storing asset allocations: {e}")
        raise

def optimize_allocations():
    """
    Orchestrates the asset allocation optimization process.
    """
    logger.info("Starting asset allocation optimization...")
    conn = None
    try:
        conn = get_db_connection()
        
        # 1. Read forecasted APY and TVL for selected pools
        pool_metrics = fetch_pool_daily_metrics(conn)
        logger.info(f"Fetched {len(pool_metrics)} pool daily metrics.")

        # 2. Read current account balances and liquidity portfolio
        current_ledger = fetch_daily_ledger(conn)
        logger.info(f"Fetched {len(current_ledger)} current ledger entries.")

        # 3. Read configurable parameters from allocation_parameters
        alloc_params = fetch_allocation_parameters(conn)
        logger.info(f"Fetched allocation parameters: {alloc_params}")

        # Prepare data for optimization
        if pool_metrics.empty:
            logger.info("No pools available for optimization after filtering. Exiting.")
            return

        # Convert relevant columns to numpy arrays for cvxpy
        forecasted_apy = pool_metrics['forecasted_apy'].fillna(0).values
        forecasted_tvl = pool_metrics['forecasted_tvl'].fillna(0).values
        
        # Assuming total_usd_value is the total value of assets to allocate
        # For now, let's assume a fixed value or derive from current_ledger
        # This needs to be refined based on how total_usd is calculated from daily_ledger
        total_usd_value = current_ledger['end_of_day_balance'].sum() if not current_ledger.empty else 100000 # Example default

        num_pools = len(pool_metrics)
        x = cp.Variable(num_pools, nonneg=True) # Allocation weights for each pool

        # Define objective function
        # Maximize (daily_yield * total_usd - total_gas_fees * total_usd - total_conversion_penalty * total_usd)
        # For simplicity, let's assume gas fees and conversion penalties are proportional to allocation
        # These will need to be more accurately modeled later.
        daily_yield = cp.sum(cp.multiply(x, forecasted_apy)) # Simplified daily yield
        
        # Placeholder for gas fees and conversion penalties
        # These would ideally be functions of the allocation x
        total_gas_fees = 0.001 * cp.sum(x) # Example: 0.1% of allocated amount
        total_conversion_penalty = 0.0005 * cp.sum(x) # Example: 0.05% of allocated amount

        objective = cp.Maximize(daily_yield * total_usd_value - total_gas_fees * total_usd_value - total_conversion_penalty * total_usd_value)

        # Define constraints
        constraints = [
            cp.sum(x) == 1, # Sum of Weights Equals 1 (Full Allocation)
            # Allocation Amounts Less Than or Equal to Per-Pool TVL Limit (`tvl_limit_percentage`)
            x * total_usd_value <= forecasted_tvl * alloc_params.get('tvl_limit_percentage', 1.0),
            # Non-Negative Weights (already handled by cp.Variable(nonneg=True))
        ]

        # Conditional constraints based on profit_optimization flag
        if alloc_params.get('profit_optimization', False):
            max_alloc_percentage = alloc_params.get('max_alloc_percentage')
            if max_alloc_percentage is not None:
                constraints.append(x <= max_alloc_percentage) # Maximum Allocation Percentage

        # Position Limits
        position_max_pct_total_assets = alloc_params.get('position_max_pct_total_assets')
        if position_max_pct_total_assets is not None:
            constraints.append(x * total_usd_value <= position_max_pct_total_assets * total_usd_value)

        position_max_pct_pool_tvl = alloc_params.get('position_max_pct_pool_tvl')
        if position_max_pct_pool_tvl is not None:
            constraints.append(x * total_usd_value <= position_max_pct_pool_tvl * forecasted_tvl)

        # Group Allocation Limits (Removed as pool_group column is not available)
        # This section will be re-evaluated if pool grouping logic is re-introduced with a different schema.

        # Create and solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS) # Specify a solver, ECOS is a good general-purpose one

        logger.info(f"Optimization status: {problem.status}")
        if problem.status in ["optimal", "optimal_near"]:
            logger.info(f"Optimal value: {problem.value}")
            allocations = pd.DataFrame({
                'pool_id': pool_metrics['pool_id'],
                'allocated_percentage': x.value,
                'allocated_usd': x.value * total_usd_value
            })
            allocations = allocations[allocations['allocated_percentage'] > 1e-6] # Filter out very small allocations
            logger.info("Optimal Allocations:")
            logger.info(allocations)
        else:
            logger.info("No optimal solution found.")
            logger.info(f"Problem status: {problem.status}")
            logger.info(f"Problem value: {problem.value}")

        # Fetch previous allocation for comparison
        previous_allocation = fetch_previous_allocation(conn)
        logger.info(f"Fetched previous allocation: {len(previous_allocation)} entries.")

        # Calculate net forecasted yield for the new optimal allocation
        net_forecasted_yield_new_allocation = 0.0
        if problem.status in ["optimal", "optimal_near"] and not allocations.empty:
            net_forecasted_yield_new_allocation = calculate_net_yield(
                allocations, total_usd_value,
                alloc_params.get('conversion_rate', 0.0004), # Default conversion rate
                alloc_params.get('gas_fee_rate', 0.001) # Default gas fee rate
            )
            logger.info(f"Net forecasted yield for new allocation: {net_forecasted_yield_new_allocation:.4f} USD")

        # Calculate net yield from existing allocation
        net_forecasted_yield_previous_allocation = 0.0
        if not previous_allocation.empty:
            # Need to join previous_allocation with pool_metrics to get forecasted_apy for calculation
            # Assuming previous_allocation already has 'forecasted_apy' from fetch_previous_allocation
            net_forecasted_yield_previous_allocation = calculate_net_yield(
                previous_allocation, total_usd_value, # Use total_usd_value for previous allocation context
                alloc_params.get('conversion_rate', 0.0004),
                alloc_params.get('gas_fee_rate', 0.001)
            )
            logger.info(f"Net forecasted yield for previous allocation: {net_forecasted_yield_previous_allocation:.4f} USD")

        # 7. Force reallocation if any token, pool, or protocol is no longer approved/available
        dynamic_lists = fetch_dynamic_lists(conn)
        logger.info(f"Dynamic lists fetched: {dynamic_lists}")
        force_reallocate = should_force_reallocation(previous_allocation, dynamic_lists)

        final_allocations = pd.DataFrame()
        run_id = None

        if force_reallocate or (net_forecasted_yield_new_allocation > net_forecasted_yield_previous_allocation and problem.status in ["optimal", "optimal_near"]):
            logger.info("Decision: Full reallocation based on higher forecasted yield or forced reallocation.")
            if problem.status in ["optimal", "optimal_near"]:
                final_allocations = allocations
            else:
                logger.info("No optimal solution found for full reallocation, but forced reallocation is active. Fallback strategy needed.")
                # Fallback: e.g., keep current allocation or move to safe assets
                final_allocations = previous_allocation # For now, keep previous if no new optimal
        else:
            logger.info("Decision: Reallocate only yesterday's yield.")
            # Fetch yesterday's realized yield from daily_ledger
            yesterday_yield_row = current_ledger[current_ledger['token_symbol'] == 'TOTAL_YIELD'] # Assuming a 'TOTAL_YIELD' entry
            yesterday_realized_yield = yesterday_yield_row['end_of_day_balance'].iloc[0] if not yesterday_yield_row.empty else 0.0
            
            if yesterday_realized_yield > 0:
                logger.info(f"Optimizing for yesterday's realized yield: {yesterday_realized_yield} USD")
                # Re-run optimization for only yesterday's yield
                # This requires a new optimization problem instance with total_usd_value = yesterday_realized_yield
                # and potentially different constraints (e.g., min_pools=1 if profit_optimization is True)
                
                # For simplicity, re-using the existing optimization setup but scaling total_usd_value
                # In a real scenario, you might want to re-evaluate constraints for this smaller allocation
                
                # Create a new problem instance for yield reallocation
                yield_x = cp.Variable(num_pools, nonneg=True)
                yield_daily_yield = cp.sum(cp.multiply(yield_x, forecasted_apy))
                yield_total_gas_fees = 0.001 * cp.sum(yield_x)
                yield_total_conversion_penalty = 0.0005 * cp.sum(yield_x)
                
                yield_objective = cp.Maximize(yield_daily_yield * yesterday_realized_yield - yield_total_gas_fees * yesterday_realized_yield - yield_total_conversion_penalty * yesterday_realized_yield)
                
                yield_constraints = [
                    cp.sum(yield_x) == 1,
                    yield_x * yesterday_realized_yield <= forecasted_tvl * alloc_params.get('tvl_limit_percentage', 1.0),
                ]
                
                if alloc_params.get('profit_optimization', False):
                    max_alloc_percentage = alloc_params.get('max_alloc_percentage')
                    if max_alloc_percentage is not None:
                        yield_constraints.append(yield_x <= max_alloc_percentage)
                
                yield_problem = cp.Problem(yield_objective, yield_constraints)
                yield_problem.solve(solver=cp.ECOS)

                if yield_problem.status in ["optimal", "optimal_near"]:
                    logger.info(f"Optimal yield reallocation value: {yield_problem.value}")
                    yield_allocations = pd.DataFrame({
                        'pool_id': pool_metrics['pool_id'],
                        'allocated_percentage': yield_x.value,
                        'allocated_usd': yield_x.value * yesterday_realized_yield
                    })
                    yield_allocations = yield_allocations[yield_allocations['allocated_percentage'] > 1e-6]
                    logger.info("Optimal Yield Allocations:")
                    logger.info(yield_allocations)
                    final_allocations = yield_allocations # This would be added to existing, not replace
                    # This part needs careful handling: merge yield_allocations with previous_allocation
                    # For now, just showing the yield allocation.
                else:
                    logger.info("No optimal solution found for yield reallocation. Keeping previous allocation.")
                    final_allocations = previous_allocation # Keep previous if yield reallocation fails
            else:
                logger.info("No yield generated yesterday or yield is zero. Keeping previous allocation.")
                final_allocations = previous_allocation

        # 8. Store optimization parameters (snapshots)
        run_id = store_allocation_parameters_snapshot(conn, alloc_params, dynamic_lists)
        
        # 9. Store allocation results
        if not final_allocations.empty:
            store_asset_allocations(conn, run_id, final_allocations)
        else:
            logger.info("No final allocations to store.")

        # Print comprehensive summary
        logger.info("\n" + "="*70)
        logger.info("💰 ASSET ALLOCATION OPTIMIZATION SUMMARY")
        logger.info("="*70)
        logger.info(f"📊 Input pools analyzed: {len(pool_metrics)}")
        logger.info(f"📈 Current ledger entries: {len(current_ledger)}")
        logger.info(f"💵 Total USD value: ${total_usd_value:,.2f}")
        logger.info(f"🔄 Previous allocations: {len(previous_allocation)}")
        if not final_allocations.empty:
            logger.info(f"✅ Final allocations created: {len(final_allocations)}")
            logger.info(f"💎 Total allocated: ${final_allocations['allocated_usd'].sum():,.2f}")
            logger.info(f"📊 Allocation spread: {len(final_allocations)} pools")
            if 'allocated_percentage' in final_allocations.columns:
                max_allocation = final_allocations['allocated_percentage'].max()
                min_allocation = final_allocations['allocated_percentage'].min()
                logger.info(f"📈 Max pool allocation: {max_allocation:.2%}")
                logger.info(f"📉 Min pool allocation: {min_allocation:.2%}")
        else:
            logger.info("❌ No final allocations created")
        if run_id:
            logger.info(f"🆔 Run ID: {run_id}")
        logger.info(f"🔁 Forced reallocation: {'Yes' if force_reallocate else 'No'}")
        if net_forecasted_yield_new_allocation > 0:
            logger.info(f"📈 New allocation net yield: ${net_forecasted_yield_new_allocation:.4f}")
        if net_forecasted_yield_previous_allocation > 0:
            logger.info(f"📊 Previous allocation net yield: ${net_forecasted_yield_previous_allocation:.4f}")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"An error occurred during optimization: {e}")
        logger.error("\n" + "="*70)
        logger.error("❌ ASSET ALLOCATION OPTIMIZATION FAILED")
        logger.error("="*70)
        logger.error(f"Error: {str(e)}")
        logger.error("="*70)
        if conn:
            conn.rollback() # Rollback any changes if an error occurs
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed.")

if __name__ == "__main__":
    optimize_allocations()