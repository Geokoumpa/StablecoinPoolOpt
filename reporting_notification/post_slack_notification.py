import os
import json
import requests
import sys
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Optional

# Configure logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


class SlackNotifier:
    def __init__(self):
        self.webhook_url = config.SLACK_WEBHOOK_URL

    def send_notification(self, message, title="Daily Allocation Report"):
        """
        Sends a formatted message to Slack via webhook.
        """
        if not self.webhook_url:
            logger.warning("Slack webhook URL not configured. Skipping notification.")
            return

        payload = {
            "attachments": [
                {
                    "fallback": title,
                    "color": "#36a64f",
                    "pretext": title,
                    "text": message,
                    "ts": os.path.getmtime(__file__) # Timestamp of the file modification
                }
            ]
        }

        try:
            response = requests.post(
                self.webhook_url, data=json.dumps(payload),
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            logger.info("Slack notification sent successfully!")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending Slack notification: {e}")


def fetch_optimization_results():
    """
    Fetches the latest optimization results from the database.
    
    Returns:
        Dictionary with optimization results or None if not found
    """
    try:
        from database.db_utils import get_db_connection
        import pandas as pd
        
        engine = get_db_connection()
        if not engine:
            logger.error("Failed to connect to database")
            return None
        
        # Get the latest optimization run
        query = """
        SELECT run_id, timestamp, max_alloc_percentage, conversion_rate, 
               projected_apy, transaction_costs, transaction_sequence
        FROM allocation_parameters
        ORDER BY timestamp DESC
        LIMIT 1
        """
        params_df = pd.read_sql(query, engine)
        
        if params_df.empty:
            logger.warning("No optimization runs found")
            return None
        
        run_id = params_df['run_id'].iloc[0]
        timestamp = params_df['timestamp'].iloc[0]
        projected_apy = params_df['projected_apy'].iloc[0] if not pd.isna(params_df['projected_apy'].iloc[0]) else 0.0
        transaction_costs = params_df['transaction_costs'].iloc[0] if not pd.isna(params_df['transaction_costs'].iloc[0]) else 0.0

        # Robustly extract transaction_sequence avoiding ambiguous truth-value of arrays
        transaction_sequence_raw = params_df['transaction_sequence'].iloc[0]
        transaction_sequence = []
        try:
            # None or explicit NaN -> treat as empty
            if transaction_sequence_raw is None or (isinstance(transaction_sequence_raw, float) and pd.isna(transaction_sequence_raw)):
                transaction_sequence = []
            elif isinstance(transaction_sequence_raw, str):
                # keep string as-is (will be parsed later)
                transaction_sequence = transaction_sequence_raw
            else:
                # If array-like (numpy array, pandas Series, list), convert to plain list
                if hasattr(transaction_sequence_raw, '__iter__') and not isinstance(transaction_sequence_raw, (str, bytes)):
                    try:
                        transaction_sequence = list(transaction_sequence_raw)
                    except Exception:
                        transaction_sequence = transaction_sequence_raw
                else:
                    transaction_sequence = transaction_sequence_raw
        except Exception as e:
            logger.warning(f"Error reading transaction_sequence from DB: {e}")
            transaction_sequence = []
        
        # Get the transactions/allocations for this run
        transactions_query = """
        SELECT step_number, operation, from_asset, to_asset, amount, pool_id
        FROM asset_allocations
        WHERE run_id = %s
        ORDER BY step_number
        """
        transactions_df = pd.read_sql(transactions_query, engine, params=(run_id,))
        
        # Get pool information with today's metrics
        if not transactions_df.empty and transactions_df['pool_id'].notna().any():
            pool_ids = transactions_df['pool_id'].dropna().unique()
            pool_ids_str = "', '".join([str(pid) for pid in pool_ids])
            pools_query = f"""
            SELECT p.pool_id, p.symbol, pdm.forecasted_apy, pdm.forecasted_tvl
            FROM pools p
            LEFT JOIN pool_daily_metrics pdm ON p.pool_id = pdm.pool_id 
                AND pdm.date = CURRENT_DATE
            WHERE p.pool_id IN ('{pool_ids_str}')
            """
            pools_df = pd.read_sql(pools_query, engine)
        else:
            pools_df = pd.DataFrame(columns=['pool_id', 'symbol', 'forecasted_apy', 'forecasted_tvl'])
        
        engine.dispose()
        
        # Process allocation transactions to get final allocations
        allocation_transactions = transactions_df[transactions_df['operation'] == 'ALLOCATION']
        
        # Use explicit empty-check to avoid ambiguous truth-value on pandas objects
        if not allocation_transactions.empty:
            # Merge with pool information
            allocations_with_pools = allocation_transactions.merge(
                pools_df, on='pool_id', how='left'
            )
            
            # Calculate USD amounts (assuming amount is in USD for allocations)
            allocations_with_pools['amount_usd'] = allocations_with_pools['amount']
            
            # Group by pool (sum all tokens for each pool)
            pool_summary = allocations_with_pools.groupby(['pool_id', 'symbol', 'forecasted_apy', 'forecasted_tvl']).agg({
                'amount_usd': 'sum'
            }).reset_index()
            
            # Create a mapping of pool_id to total pool allocation
            pool_total_mapping = pool_summary.set_index('pool_id')['amount_usd'].to_dict()
            pool_summary = pool_summary.sort_values('amount_usd', ascending=False)
            
            # Group by token
            token_summary = allocations_with_pools.groupby('to_asset').agg({
                'amount_usd': 'sum'
            }).reset_index()
            token_summary = token_summary.sort_values('amount_usd', ascending=False)
            token_summary = token_summary.rename(columns={'to_asset': 'token'})
            
            total_allocated = allocations_with_pools['amount_usd'].sum()
        else:
            # Ensure consistent return shapes when there are no allocations
            pool_summary = pd.DataFrame(columns=['pool_id', 'symbol', 'forecasted_apy', 'forecasted_tvl', 'amount_usd'])
            token_summary = pd.DataFrame(columns=['token', 'amount_usd'])
            total_allocated = 0.0
            pool_total_mapping = {}
        
        # Calculate transaction costs
        total_transaction_count = len(transactions_df)
        
        # Parse transaction sequence if available
        parsed_transaction_sequence = []
        # Avoid ambiguous truth-value checks on array-like objects (numpy / pandas)
        if transaction_sequence is not None and not (hasattr(transaction_sequence, '__len__') and len(transaction_sequence) == 0):
            try:
                if isinstance(transaction_sequence, str):
                    parsed_transaction_sequence = json.loads(transaction_sequence)
                else:
                    # Ensure array-like types become plain Python lists for downstream checks
                    try:
                        parsed_transaction_sequence = list(transaction_sequence)
                    except Exception:
                        parsed_transaction_sequence = transaction_sequence
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Error parsing transaction sequence: {e}")
                parsed_transaction_sequence = []
        
        # Compute transaction cost breakdowns from parsed transaction sequence where possible
        total_gas_cost = 0.0
        total_conversion_cost = 0.0
        # parsed_transaction_sequence can be a list of dicts, a string, or other types.
        if isinstance(parsed_transaction_sequence, list):
            for txn in parsed_transaction_sequence:
                try:
                    total_gas_cost += float(txn.get('gas_cost_usd', 0) or 0)
                except Exception:
                    # ignore malformed values
                    pass
                try:
                    total_conversion_cost += float(txn.get('conversion_cost_usd', 0) or 0)
                except Exception:
                    pass
        else:
            # If it's a string JSON, try to parse it and aggregate
            if isinstance(parsed_transaction_sequence, str):
                try:
                    seq_parsed = json.loads(parsed_transaction_sequence)
                    if isinstance(seq_parsed, list):
                        for txn in seq_parsed:
                            try:
                                total_gas_cost += float(txn.get('gas_cost_usd', 0) or 0)
                            except Exception:
                                pass
                            try:
                                total_conversion_cost += float(txn.get('conversion_cost_usd', 0) or 0)
                            except Exception:
                                pass
                except Exception:
                    # leave totals at 0 if parsing fails
                    pass
        
        return {
            'run_id': run_id,
            'timestamp': timestamp,
            'total_allocated': total_allocated,
            'total_gas_cost': total_gas_cost,
            'total_conversion_cost': total_conversion_cost,
            'total_transaction_cost': transaction_costs if transaction_costs is not None else (total_gas_cost + total_conversion_cost),
            'projected_apy': projected_apy,  # New field
            'transaction_sequence': parsed_transaction_sequence,  # New field
            'pool_count': len(pool_summary),
            'token_count': len(token_summary),
            'transaction_count': total_transaction_count,
            'top_pools': pool_summary.head(5).to_dict('records'),
            'token_allocation': token_summary.to_dict('records'),
            'allocations_df': allocations_with_pools if not allocation_transactions.empty else pd.DataFrame(),
            'transactions_df': transactions_df,
            'pool_total_mapping': pool_total_mapping if not allocation_transactions.empty else {}
        
        }
    except Exception as e:
        # Log full traceback to help identify where the ambiguous-array error originates
        logger.exception("Error fetching optimization results")
        return None


def calculate_yield_metrics(results: Dict) -> Dict:
    """
    Calculate yield metrics from optimization results.
    
    Args:
        results: Optimization results dictionary
        
    Returns:
        Dictionary with yield metrics
    """
    try:
        from database.db_utils import get_db_connection
        import pandas as pd
        
        engine = get_db_connection()
        if not engine:
            return {}
        
        # Check if we have allocation data
        if results['allocations_df'].empty:
            return {}
        
        # Get pool APYs
        pool_ids = results['allocations_df']['pool_id'].dropna().unique()
        if len(pool_ids) == 0:
            return {}
        
        pool_ids_str = "', '".join([str(pid) for pid in pool_ids])
        apy_query = f"""
        SELECT pdm.pool_id, pdm.forecasted_apy, p.symbol
        FROM pool_daily_metrics pdm
        JOIN pools p ON pdm.pool_id = p.pool_id
        WHERE pdm.pool_id IN ('{pool_ids_str}')
          AND pdm.date = CURRENT_DATE
        """
        apy_df = pd.read_sql(apy_query, engine)
        engine.dispose()
        
        if apy_df.empty:
            return {}
        
        # Check if forecasted_apy column exists in apy_df
        if 'forecasted_apy' not in apy_df.columns:
            logger.warning("forecasted_apy column not found in APY data")
            return {}
        
        # Instead of relying on a merge that may produce unexpected column names,
        # build a mapping and map forecasted_apy onto allocations to ensure the
        # column is always present.
        try:
            apy_map = apy_df.set_index('pool_id')['forecasted_apy'].to_dict()
            allocations_with_apy = results['allocations_df'].copy()
            allocations_with_apy['forecasted_apy'] = allocations_with_apy['pool_id'].map(apy_map)
        except Exception as map_error:
            logger.error(f"Error mapping APY values onto allocations: {map_error}")
            return {}
        
        # Fill missing APY values with 0
        allocations_with_apy['forecasted_apy'] = allocations_with_apy['forecasted_apy'].fillna(0)
        
        # Calculate daily and annual yield
        allocations_with_apy['daily_yield'] = (
            allocations_with_apy['amount_usd'] * allocations_with_apy['forecasted_apy'] / 100 / 365
        )
        allocations_with_apy['annual_yield'] = (
            allocations_with_apy['amount_usd'] * allocations_with_apy['forecasted_apy'] / 100
        )
        
        try:
            total_daily_yield = allocations_with_apy['daily_yield'].sum()
            total_annual_yield = allocations_with_apy['annual_yield'].sum()
            weighted_avg_apy = (allocations_with_apy['amount_usd'] * allocations_with_apy['forecasted_apy']).sum() / allocations_with_apy['amount_usd'].sum()
            
            return {
                'total_daily_yield': total_daily_yield,
                'total_annual_yield': total_annual_yield,
                'weighted_avg_apy': weighted_avg_apy,
                'net_daily_yield': total_daily_yield - (results['total_transaction_cost'] / 365),  # Subtract daily transaction cost
                'net_annual_yield': total_annual_yield - results['total_transaction_cost']
            }
        except Exception as calc_error:
            logger.error(f"Error in yield calculations: {calc_error}")
            return {}
        
    except Exception as e:
        logger.error(f"Error calculating yield metrics: {e}")
        return {}


def format_slack_message(results: Dict, yield_metrics: Dict) -> str:
    """
    Format optimization results into a Slack message.
    
    Args:
        results: Optimization results dictionary
        yield_metrics: Yield metrics dictionary
        
    Returns:
        Formatted Slack message string
    """
    from datetime import datetime
    from database.db_utils import get_db_connection
    import pandas as pd
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    # Mapping of pool_id -> total pool allocation provided by fetch_optimization_results (if available)
    pool_total_mapping = results.get('pool_total_mapping', {})
    
    # Fetch total AUM from daily_balances to calculate correct percentages
    # The constraint is based on total AUM, not just allocated amount
    total_aum = results['total_allocated']  # Default to allocated if can't get AUM
    try:
        engine = get_db_connection()
        if engine:
            aum_query = """
            SELECT SUM(allocated_balance + unallocated_balance) as total_aum
            FROM daily_balances
            WHERE date = CURRENT_DATE
            """
            aum_df = pd.read_sql(aum_query, engine)
            if not aum_df.empty and not pd.isna(aum_df['total_aum'].iloc[0]):
                total_aum = float(aum_df['total_aum'].iloc[0])
            engine.dispose()
    except Exception as e:
        logger.warning(f"Could not fetch total AUM, using allocated amount: {e}")
    
    message = f"""
*📊 Daily Allocation Report - {date_str}*

*💰 Portfolio Summary:*
• Total AUM: ${total_aum:,.2f}
• Total Allocated: ${results['total_allocated']:,.2f}
• Number of Pools: {results['pool_count']}
• Number of Tokens: {results['token_count']}
• Transactions: {results['transaction_count']}

*💵 Yield Metrics:*
"""
    
    if yield_metrics:
        message += f"""• Weighted Avg APY: {yield_metrics.get('weighted_avg_apy', 0):.2f}%
• Daily Yield: ${yield_metrics.get('total_daily_yield', 0):.2f}
• Annual Yield: ${yield_metrics.get('total_annual_yield', 0):,.2f}
• Net Daily Yield (after fees): ${yield_metrics.get('net_daily_yield', 0):.2f}
• Net Annual Yield (after fees): ${yield_metrics.get('net_annual_yield', 0):,.2f}
"""
    
    message += f"""
*💸 Transaction Costs:*
• Gas Fees: ${results['total_gas_cost']:.4f}
• Conversion Fees: ${results['total_conversion_cost']:.4f}
• Total Transaction Cost: ${results['total_transaction_cost']:.4f}

*📈 Optimization Metrics:*
• Projected APY: {results.get('projected_apy', 0):.2f}%

*🏆 Top Pool Allocations:*
"""
    
    for pool in results['top_pools']:
        # Get total pool allocation (sum of all tokens in this pool)
        pool_id = pool['pool_id']
        pool_total_amount = pool_total_mapping.get(pool_id, pool['amount_usd'])
        # Calculate percentage based on total AUM (same as optimizer constraint)
        percentage = (pool_total_amount / total_aum) * 100 if total_aum > 0 else 0
        apy = pool.get('forecasted_apy', 0)
        tvl = pool.get('forecasted_tvl', 0)
        tvl_str = f"${tvl:,.0f}" if tvl and tvl > 0 else "N/A"
        apy_str = f"{apy:.2f}%" if apy and apy > 0 else "N/A"
        
        # Display the total pool allocation amount (not individual token amount)
        message += f"• {pool['symbol']} (ID: {pool['pool_id']}): ${pool_total_amount:,.2f} ({percentage:.1f}% of AUM) | APY: {apy_str} | TVL: {tvl_str}\n"
    
    message += "\n*💎 Token Allocation:*\n"
    for token in results['token_allocation']:
        percentage = (token['amount_usd'] / results['total_allocated']) * 100 if results['total_allocated'] > 0 else 0
        message += f"• {token['token']}: ${token['amount_usd']:,.2f} ({percentage:.1f}%)\n"
    
    # Add transaction details if available
    transaction_sequence = results.get('transaction_sequence', [])
    if transaction_sequence:
        message += "\n*🔄 Transaction Sequence:*\n"
        for txn in transaction_sequence:
            txn_type = txn.get('type', '')
            seq = txn.get('seq', 0)
            amount_usd = txn.get('amount_usd', 0)
            gas_cost = txn.get('gas_cost_usd', 0)
            conv_cost = txn.get('conversion_cost_usd', 0)
            total_cost = txn.get('total_cost_usd', 0)
            
            if txn_type == 'ALLOCATION':
                token = txn.get('token', '')
                pool_id = txn.get('to_location', '')
                pool_info = f" (Pool ID: {pool_id})" if pool_id else ""
                message += f"• Step {seq}: ALLOCATE ${amount_usd:,.2f} {token}{pool_info} | Gas: ${gas_cost:.4f} | Total: ${total_cost:.4f}\n"
            elif txn_type == 'WITHDRAWAL':
                token = txn.get('token', '')
                pool_id = txn.get('from_location', '')
                pool_info = f" (Pool ID: {pool_id})" if pool_id else ""
                message += f"• Step {seq}: WITHDRAW ${amount_usd:,.2f} {token}{pool_info} | Gas: ${gas_cost:.4f} | Total: ${total_cost:.4f}\n"
            elif txn_type == 'CONVERSION':
                from_token = txn.get('from_token', '')
                to_token = txn.get('to_token', '')
                message += f"• Step {seq}: CONVERT ${amount_usd:,.2f} {from_token} → {to_token} | Gas: ${gas_cost:.4f} | Conv: ${conv_cost:.4f} | Total: ${total_cost:.4f}\n"
    elif not results['transactions_df'].empty:
        # Fallback to old method if transaction_sequence is not available
        # Use explicit empty-check to avoid ambiguous truth-value on pandas objects
        message += "\n*🔄 Transaction Sequence:*\n"
        for _, txn in results['transactions_df'].iterrows():
            if txn['operation'] == 'ALLOCATION':
                pool_info = ""
                if not pd.isna(txn['pool_id']):
                    pool_row = results['allocations_df'][results['allocations_df']['pool_id'] == txn['pool_id']]
                    if not pool_row.empty:
                        pool_symbol = pool_row['symbol'].iloc[0]
                        pool_apy = pool_row['forecasted_apy'].iloc[0]
                        pool_info = f" ({pool_symbol}, APY: {pool_apy:.2f}%)" if pool_apy else f" ({pool_symbol})"
                
                pool_id_info = f" (Pool ID: {txn['pool_id']})" if not pd.isna(txn['pool_id']) else ""
                message += f"• Step {txn['step_number']}: ALLOCATE ${txn['amount']:,.2f} {txn['to_asset']}{pool_info}{pool_id_info}\n"
            elif txn['operation'] == 'WITHDRAWAL':
                message += f"• Step {txn['step_number']}: WITHDRAW ${txn['amount']:,.2f} {txn['from_asset']}\n"
            elif txn['operation'] == 'CONVERSION':
                message += f"• Step {txn['step_number']}: CONVERT ${txn['amount']:,.2f} {txn['from_asset']} → {txn['to_asset']}\n"
    
    message += f"\n*🔍 Run ID: {results['run_id']}*"
    message += f"\n*✅ Status: Optimization completed successfully*"
    
    return message


def main():
    """
    Main function to fetch optimization results and send Slack notification.
    """
    logger.info("Starting Slack notification process...")
    
    # Fetch optimization results
    results = fetch_optimization_results()
    if not results:
        logger.error("No optimization results found. Cannot send notification.")
        return
    
    # Debug: Log pool total mapping
    logger.info(f"Pool total mapping: {results.get('pool_total_mapping', {})}")
    logger.info(f"Top pools data: {results.get('top_pools', [])}")
    
    # Calculate yield metrics
    yield_metrics = calculate_yield_metrics(results)
    
    # Format message
    message = format_slack_message(results, yield_metrics)
    
    # Print the message that will be sent to Slack
    logger.info("\n" + "="*80)
    logger.info("SLACK MESSAGE PREVIEW:")
    logger.info("="*80)
    print(message)
    logger.info("="*80)
    
    # Send notification
    notifier = SlackNotifier()
    notifier.send_notification(message, "Daily Allocation Report")
    
    logger.info("Slack notification process completed.")


if __name__ == "__main__":
    main()