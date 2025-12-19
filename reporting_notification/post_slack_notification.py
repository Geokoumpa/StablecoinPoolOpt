import os
import json
import requests
import sys
import logging
import pandas as pd
from datetime import datetime, date
from typing import Dict

from database.repositories.parameter_repository import ParameterRepository
from database.repositories.allocation_repository import AllocationRepository
from database.repositories.pool_metrics_repository import PoolMetricsRepository
from database.repositories.daily_balance_repository import DailyBalanceRepository
import config

# Configure logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

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
    Reconstructs the final portfolio state by applying transactions to the initial state.
    
    Returns:
        Dictionary with optimization results or None if not found
    """
    try:
        params_repo = ParameterRepository()
        alloc_repo = AllocationRepository()
        pool_repo = PoolMetricsRepository()
        balance_repo = DailyBalanceRepository()
        
        # Get latest parameters
        params = params_repo.get_latest_parameters()
        if not params:
            logger.warning("No optimization runs found")
            return None
            
        run_id = params.run_id
        timestamp = params.timestamp
        projected_apy = params.projected_apy or 0.0
        transaction_costs = params.transaction_costs or 0.0
        transaction_sequence_raw = params.transaction_sequence
        
        # Parse transaction_sequence
        transaction_sequence = []
        try:
            if transaction_sequence_raw:
                if isinstance(transaction_sequence_raw, str):
                    transaction_sequence = transaction_sequence_raw # Keep as string/json to be parsed later
                elif hasattr(transaction_sequence_raw, '__iter__'):
                    transaction_sequence = list(transaction_sequence_raw)
        except Exception as e:
            logger.warning(f"Error reading transaction_sequence: {e}")

        # Get allocations for this run (The execution log)
        allocations_objs = alloc_repo.get_allocations_by_run_id(run_id)
        
        # Convert to DataFrame of Transactions
        transactions_data = []
        for a in allocations_objs:
            transactions_data.append({
                'step_number': a.step_number,
                'operation': a.operation,
                'from_asset': a.from_asset,
                'to_asset': a.to_asset,
                'amount': float(a.amount) if a.amount is not None else 0.0,
                'pool_id': a.pool_id
            })
            
        transactions_df = pd.DataFrame(transactions_data)
        if transactions_df.empty:
            transactions_df = pd.DataFrame(columns=['step_number', 'operation', 'from_asset', 'to_asset', 'amount', 'pool_id'])
        
        # --- RECONSTRUCT FINAL PORTFOLIO STATE ---
        from config import MAIN_ASSET_HOLDING_ADDRESS
        
        # Fetch token prices for USD conversion
        from database.repositories.raw_data_repository import RawDataRepository
        price_repo = RawDataRepository()
        
        # Log configuration for debugging
        logger.info(f"MAIN_ASSET_HOLDING_ADDRESS: {MAIN_ASSET_HOLDING_ADDRESS}")
        
        tokens_to_fetch = set()
        if MAIN_ASSET_HOLDING_ADDRESS:
             start_balances = balance_repo.get_current_balances(MAIN_ASSET_HOLDING_ADDRESS, date.today())
             logger.info(f"Fetched {len(start_balances)} initial balance entries from DB")
             for bal in start_balances:
                 tokens_to_fetch.add(bal.token_symbol)
        else:
             logger.warning("MAIN_ASSET_HOLDING_ADDRESS is not configured. Starting portfolio state reconstruction from empty.")
        
        if not transactions_df.empty:
             tokens_to_fetch.update(transactions_df['to_asset'].dropna().unique())
             tokens_to_fetch.update(transactions_df['from_asset'].dropna().unique())
             
        token_prices = price_repo.get_latest_prices(list(tokens_to_fetch))
        
        # Parse JSON sequence for better fidelity (db table might miss pool_id on withdraw)
        parsed_sequence = []
        if isinstance(transaction_sequence, str):
            try:
                parsed_sequence = json.loads(transaction_sequence)
            except: pass
        elif isinstance(transaction_sequence, list):
            parsed_sequence = transaction_sequence

        # Build Portfolio State (Pool -> {token: amount})
        portfolio_state = {} 
        if MAIN_ASSET_HOLDING_ADDRESS:
             start_balances = balance_repo.get_current_balances(MAIN_ASSET_HOLDING_ADDRESS, date.today())
             for bal in start_balances:
                 if bal.allocated_balance and bal.allocated_balance > 0 and bal.pool_id:
                     pid = bal.pool_id
                     if pid not in portfolio_state: portfolio_state[pid] = {}
                     t = bal.token_symbol
                     portfolio_state[pid][t] = portfolio_state[pid].get(t, 0.0) + float(bal.allocated_balance)
        
        # Replay transactions
        for txn in parsed_sequence:
            t_type = txn.get('type')
            amt = float(txn.get('amount', 0))
            token = txn.get('token')
            
            if t_type == 'WITHDRAWAL':
                pid = txn.get('from_location')
                if pid and pid in portfolio_state and token in portfolio_state[pid]:
                    portfolio_state[pid][token] = max(0.0, portfolio_state[pid][token] - amt)
                    if portfolio_state[pid][token] <= 1e-9:
                        del portfolio_state[pid][token]
                    if not portfolio_state[pid]:
                        del portfolio_state[pid]
                        
            elif t_type == 'ALLOCATION':
                pid = txn.get('to_location')
                if pid:
                    if pid not in portfolio_state: portfolio_state[pid] = {}
                    portfolio_state[pid][token] = portfolio_state[pid].get(token, 0.0) + amt
                    
            elif t_type == 'HOLD':
                pid = txn.get('to_location') or txn.get('from_location')
                if pid:
                    if pid not in portfolio_state: portfolio_state[pid] = {}
                    # Only add if not already present to avoid double counting if daily_balances worked
                    # or update to the specific HOLD amount if we want to trust the optimizer's view
                    if token not in portfolio_state[pid]:
                        portfolio_state[pid][token] = amt
                    else:
                        # If already there, we trust the HOLD amount as the definitive state for that token/pool
                        portfolio_state[pid][token] = amt

        # Flatten for reporting
        final_allocations_list = []
        total_allocated_usd = 0.0
        
        for pid, tokens in portfolio_state.items():
            for t, amt in tokens.items():
                price = token_prices.get(t, 1.0)
                amt_usd = amt * price
                final_allocations_list.append({
                    'pool_id': pid,
                    'token': t,
                    'amount': amt,
                    'amount_usd': amt_usd
                })
                total_allocated_usd += amt_usd
        
        final_allocations_df = pd.DataFrame(final_allocations_list)
        if final_allocations_df.empty:
            final_allocations_df = pd.DataFrame(columns=['pool_id', 'token', 'amount', 'amount_usd'])

        # Get Metadata
        all_pool_ids = set(final_allocations_df['pool_id'].unique().tolist())
        
        # Add pool IDs from transactions (e.g. withdrawals from pools not in final allocation)
        for txn in parsed_sequence:
            pid = None
            t_type = txn.get('type')
            if t_type == 'ALLOCATION' or t_type == 'HOLD':
                pid = txn.get('to_location')
            elif t_type == 'WITHDRAWAL':
                pid = txn.get('from_location')
            
            if pid and pid != 'warm_wallet':
                all_pool_ids.add(pid)
        
        all_pool_ids = list(all_pool_ids)
        
        pools_data = []
        if all_pool_ids:
            pool_rows = pool_repo.get_pool_metrics_batch(all_pool_ids, date.today())
            for row in pool_rows:
                pools_data.append({
                    'pool_id': row[0],
                    'symbol': row[1],
                    'forecasted_apy': float(row[2]) if row[2] is not None else 0.0,
                    'forecasted_tvl': float(row[3]) if row[3] is not None else 0.0,
                    'pool_address': row[4]
                })
        pools_meta_df = pd.DataFrame(pools_data)
        if pools_meta_df.empty:
            pools_meta_df = pd.DataFrame(columns=['pool_id', 'symbol', 'forecasted_apy', 'forecasted_tvl', 'pool_address'])
            
        allocations_with_pools = final_allocations_df.merge(pools_meta_df, on='pool_id', how='left')
        
        # Build Address Map
        pool_address_map = {}
        if not pools_meta_df.empty:
             # Ensure pool_address is not None
             idx = pools_meta_df['pool_address'].notna()
             pool_address_map = pools_meta_df[idx].set_index('pool_id')['pool_address'].to_dict()

        # Build Summaries
        if not allocations_with_pools.empty:
            pool_summary = allocations_with_pools.groupby(['pool_id', 'symbol', 'forecasted_apy', 'forecasted_tvl']).agg({
                'amount_usd': 'sum'
            }).reset_index()
            # Restore pool_address somehow? It's lost in groupby unless included.
            # Easier to map back using pool_address_map in formatting.
            pool_total_mapping = pool_summary.set_index('pool_id')['amount_usd'].to_dict()
            pool_summary = pool_summary.sort_values('amount_usd', ascending=False)
            
            token_summary = allocations_with_pools.groupby('token').agg({
                'amount_usd': 'sum'
            }).reset_index()
            token_summary = token_summary.sort_values('amount_usd', ascending=False)
        else:
            pool_summary = pd.DataFrame(columns=['pool_id', 'symbol', 'forecasted_apy', 'forecasted_tvl', 'amount_usd'])
            token_summary = pd.DataFrame(columns=['token', 'amount_usd'])
            pool_total_mapping = {}

        # Calculate costs
        total_gas_cost = 0.0
        total_conversion_cost = 0.0
        for txn in parsed_sequence:
            if isinstance(txn, dict):
                 total_gas_cost += float(txn.get('gas_cost_usd', 0) or 0)
                 total_conversion_cost += float(txn.get('conversion_cost_usd', 0) or 0)

        return {
            'run_id': run_id,
            'timestamp': timestamp,
            'total_allocated': total_allocated_usd,
            'total_gas_cost': total_gas_cost,
            'total_conversion_cost': total_conversion_cost,
            'total_transaction_cost': float(transaction_costs),
            'projected_apy': float(projected_apy),
            'transaction_sequence': parsed_sequence,
            'pool_count': len(pool_summary),
            'token_count': len(token_summary),
            'transaction_count': len(transactions_df),
            'top_pools': pool_summary.head(5).to_dict('records'),
            'token_allocation': token_summary.to_dict('records'),
            'allocations_df': allocations_with_pools,
            'transactions_df': transactions_df,
            'pool_total_mapping': pool_total_mapping,
            'pool_address_map': pool_address_map
        }

    except Exception as e:
        logger.exception("Error fetching optimization results")
        return None


def calculate_yield_metrics(results: Dict) -> Dict:
    """Calculate yield metrics from results."""
    try:
        if results['allocations_df'].empty:
            return {}
            
        pool_ids = results['allocations_df']['pool_id'].dropna().unique().tolist()
        if not pool_ids:
            return {}
            
        repo = PoolMetricsRepository()
        pool_rows = repo.get_pool_metrics_batch(pool_ids, date.today())
        
        # Create map pool_id -> forecasted_apy
        apy_map = {}
        for row in pool_rows:
            # row: pool_id, symbol, apy, tvl
            apy_map[row[0]] = float(row[2]) if row[2] is not None else 0.0
            
        allocations_with_apy = results['allocations_df'].copy()
        allocations_with_apy['forecasted_apy'] = allocations_with_apy['pool_id'].map(apy_map).fillna(0)
        
        allocations_with_apy['daily_yield'] = (
            allocations_with_apy['amount_usd'] * allocations_with_apy['forecasted_apy'] / 100 / 365
        )
        allocations_with_apy['annual_yield'] = (
            allocations_with_apy['amount_usd'] * allocations_with_apy['forecasted_apy'] / 100
        )
        
        total_daily_yield = allocations_with_apy['daily_yield'].sum()
        total_annual_yield = allocations_with_apy['annual_yield'].sum()
        
        total_usd = allocations_with_apy['amount_usd'].sum()
        weighted_avg_apy = 0.0
        if total_usd > 0:
             weighted_avg_apy = (allocations_with_apy['amount_usd'] * allocations_with_apy['forecasted_apy']).sum() / total_usd

        return {
            'total_daily_yield': total_daily_yield,
            'total_annual_yield': total_annual_yield,
            'weighted_avg_apy': weighted_avg_apy,
            'net_daily_yield': total_daily_yield - (results['total_transaction_cost'] / 365),
            'net_annual_yield': total_annual_yield - results['total_transaction_cost']
        }
        
    except Exception as e:
        logger.error(f"Error calculating yield metrics: {e}")
        return {}


def format_slack_message(results: Dict, yield_metrics: Dict) -> str:
    """Format results into Slack message."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    pool_total_mapping = results.get('pool_total_mapping', {})
    pool_address_map = results.get('pool_address_map', {})
    
    # Fetch total AUM
    total_aum = results['total_allocated']
    try:
        repo = DailyBalanceRepository()
        db_aum = repo.get_total_aum(date.today())
        if db_aum > 0:
            total_aum = db_aum
    except Exception as e:
        logger.warning(f"Could not fetch total AUM: {e}")

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
        pool_id = pool['pool_id']
        pool_total_amount = pool_total_mapping.get(pool_id, pool['amount_usd'])
        percentage = (pool_total_amount / total_aum) * 100 if total_aum > 0 else 0
        apy = pool.get('forecasted_apy', 0)
        tvl = pool.get('forecasted_tvl', 0)
        tvl_str = f"${tvl:,.0f}" if tvl > 0 else "N/A"
        apy_str = f"{apy:.2f}%" if apy > 0 else "N/A"
        
        # Format pool ID and Address
        addr = pool_address_map.get(pool_id, "N/A")
        if addr != "N/A":
             id_str = f"ID: {pool_id} | Addr: {addr}"
        else:
             id_str = f"ID: {pool_id}"
             
        message += f"• {pool['symbol']} ({id_str}): ${pool_total_amount:,.2f} ({percentage:.1f}% of AUM) | APY: {apy_str} | TVL: {tvl_str}\n"

    message += "\n*💎 Token Allocation:*\n"
    for token in results['token_allocation']:
        percentage = (token['amount_usd'] / results['total_allocated']) * 100 if results['total_allocated'] > 0 else 0
        message += f"• {token['token']}: ${token['amount_usd']:,.2f} ({percentage:.1f}%)\n"

    # Transaction Sequence
    parsed_sequence = results.get('transaction_sequence', [])
    if parsed_sequence:
        message += "\n*🔄 Transaction Sequence:*\n"
        for txn in parsed_sequence:
             t_type = txn.get('type')
             seq = txn.get('seq')
             amt = txn.get('amount_usd', 0)
             total = txn.get('total_cost_usd', 0)
             token = txn.get('token')
             
             # Resolve pool address for context
             pool_id_ctx = txn.get('to_location') if t_type in ['ALLOCATION', 'HOLD'] else txn.get('from_location')
             addr_info = ""
             if pool_id_ctx:
                  addr = pool_address_map.get(pool_id_ctx)
                  if addr:
                      addr_info = f" (Pool: {pool_id_ctx} | {addr})"
                  else:
                      addr_info = f" (Pool: {pool_id_ctx})"
             
             if t_type == 'HOLD':
                 # Step 0 or similar
                 message += f"• Step {seq}: HOLD ${amt:,.2f} {token}{addr_info}\n"
             elif t_type == 'ALLOCATION':
                 message += f"• Step {seq}: ALLOCATE ${amt:,.2f} {token}{addr_info} | Cost: ${total:.4f}\n"
             elif t_type == 'WITHDRAWAL':
                 message += f"• Step {seq}: WITHDRAW ${amt:,.2f} {txn.get('token')}{addr_info} | Total Cost: ${total:.4f}\n"
             elif t_type == 'CONVERSION':
                 message += f"• Step {seq}: CONVERT ${amt:,.2f} {txn.get('from_token')} -> {txn.get('to_token')} | Total Cost: ${total:.4f}\n"
                 
    elif not results['transactions_df'].empty:
         # Fallback
         message += "\n*🔄 Transaction Sequence (Fallback):*\n"
         for _, row in results['transactions_df'].iterrows():
             message += f"• Step {row['step_number']}: {row['operation']} ${row['amount']}\n"
             
    message += f"\n*🔍 Run ID: {results['run_id']}*"
    message += f"\n*✅ Status: Optimization completed successfully*"
    
    return message


def main():
    logger.info("Starting Slack notification process...")
    results = fetch_optimization_results()
    if not results:
        logger.error("No optimization results found.")
        return
        
    yield_metrics = calculate_yield_metrics(results)
    message = format_slack_message(results, yield_metrics)
    
    logger.info("SLACK MESSAGE PREVIEW:")
    print(message)
    
    notifier = SlackNotifier()
    notifier.send_notification(message, "Daily Allocation Report")
    logger.info("Slack notification process completed.")

if __name__ == "__main__":
    main()