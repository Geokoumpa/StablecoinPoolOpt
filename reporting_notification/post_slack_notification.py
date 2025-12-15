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
    
    Returns:
        Dictionary with optimization results or None if not found
    """
    try:
        params_repo = ParameterRepository()
        alloc_repo = AllocationRepository()
        pool_repo = PoolMetricsRepository() # Actually using Metrics repo which has pool details too
        
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

        # Get allocations for this run
        allocations_objs = alloc_repo.get_allocations_by_run_id(run_id)
        
        # Convert to DataFrame
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
        
        # Get pool information
        pool_ids = []
        if not transactions_df.empty:
            pool_ids = transactions_df['pool_id'].dropna().unique().tolist()
            
        pools_data = []
        if pool_ids:
            pool_rows = pool_repo.get_pool_metrics_batch(pool_ids, date.today())
            for row in pool_rows:
                # row: pool_id, symbol, forecasted_apy, forecasted_tvl
                pools_data.append({
                    'pool_id': row[0],
                    'symbol': row[1],
                    'forecasted_apy': float(row[2]) if row[2] is not None else 0.0,
                    'forecasted_tvl': float(row[3]) if row[3] is not None else 0.0
                })
        
        pools_df = pd.DataFrame(pools_data)
        if pools_df.empty:
            pools_df = pd.DataFrame(columns=['pool_id', 'symbol', 'forecasted_apy', 'forecasted_tvl'])

        # Process allocation transactions
        allocation_transactions = transactions_df[transactions_df['operation'] == 'ALLOCATION']
        
        if not allocation_transactions.empty:
            allocations_with_pools = allocation_transactions.merge(pools_df, on='pool_id', how='left')
            allocations_with_pools['amount_usd'] = allocations_with_pools['amount']
            
            # Pool summary
            pool_summary = allocations_with_pools.groupby(['pool_id', 'symbol', 'forecasted_apy', 'forecasted_tvl']).agg({
                'amount_usd': 'sum'
            }).reset_index()
            pool_total_mapping = pool_summary.set_index('pool_id')['amount_usd'].to_dict()
            pool_summary = pool_summary.sort_values('amount_usd', ascending=False)
            
            # Token summary
            token_summary = allocations_with_pools.groupby('to_asset').agg({
                'amount_usd': 'sum'
            }).reset_index().rename(columns={'to_asset': 'token'})
            token_summary = token_summary.sort_values('amount_usd', ascending=False)
            
            total_allocated = allocations_with_pools['amount_usd'].sum()
        else:
             pool_summary = pd.DataFrame(columns=['pool_id', 'symbol', 'forecasted_apy', 'forecasted_tvl', 'amount_usd'])
             token_summary = pd.DataFrame(columns=['token', 'amount_usd'])
             total_allocated = 0.0
             pool_total_mapping = {}
             allocations_with_pools = pd.DataFrame()

        # Calculate costs from transaction_sequence if possible
        total_gas_cost = 0.0
        total_conversion_cost = 0.0
        
        parsed_sequence = []
        if isinstance(transaction_sequence, str):
            try:
                parsed_sequence = json.loads(transaction_sequence)
            except: pass
        elif isinstance(transaction_sequence, list):
            parsed_sequence = transaction_sequence
            
        for txn in parsed_sequence:
            if isinstance(txn, dict):
                 total_gas_cost += float(txn.get('gas_cost_usd', 0) or 0)
                 total_conversion_cost += float(txn.get('conversion_cost_usd', 0) or 0)

        return {
            'run_id': run_id,
            'timestamp': timestamp,
            'total_allocated': total_allocated,
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
            'pool_total_mapping': pool_total_mapping
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
        message += f"• {pool['symbol']} (ID: {pool_id}): ${pool_total_amount:,.2f} ({percentage:.1f}% of AUM) | APY: {apy_str} | TVL: {tvl_str}\n"

    message += "\n*💎 Token Allocation:*\n"
    for token in results['token_allocation']:
        percentage = (token['amount_usd'] / results['total_allocated']) * 100 if results['total_allocated'] > 0 else 0
        message += f"• {token['token']}: ${token['amount_usd']:,.2f} ({percentage:.1f}%)\n"

    # Transaction Sequence
    parsed_sequence = results.get('transaction_sequence', [])
    if parsed_sequence:
        message += "\n*🔄 Transaction Sequence:*\n"
        for txn in parsed_sequence:
             # Basic formatting similar to original
             t_type = txn.get('type')
             seq = txn.get('seq')
             amt = txn.get('amount_usd', 0)
             total = txn.get('total_cost_usd', 0)
             
             if t_type == 'ALLOCATION':
                 message += f"• Step {seq}: ALLOCATE ${amt:,.2f} {txn.get('token')} | Total Cost: ${total:.4f}\n"
             elif t_type == 'WITHDRAWAL':
                 message += f"• Step {seq}: WITHDRAW ${amt:,.2f} {txn.get('token')} | Total Cost: ${total:.4f}\n"
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