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

    def send_notification(self, blocks=None, text_fallback="Daily Allocation Report", title="Daily Allocation Report"):
        """
        Sends a formatted message to Slack via webhook using Block Kit.
        """
        if not self.webhook_url:
            logger.warning("Slack webhook URL not configured. Skipping notification.")
            return

        payload = {
            "text": text_fallback,
            "blocks": blocks if blocks else []
        }
        
        # If no blocks, use legacy structure (though we plan to move to blocks)
        if not blocks:
            payload = {
                "attachments": [
                    {
                        "fallback": title,
                        "color": "#36a64f",
                        "pretext": title,
                        "text": text_fallback,
                        "ts": os.path.getmtime(__file__)
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
            if response is not None:
                logger.error(f"Response: {response.text}")


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
        wallet_state = {}
        
        if MAIN_ASSET_HOLDING_ADDRESS:
             start_balances = balance_repo.get_current_balances(MAIN_ASSET_HOLDING_ADDRESS, date.today())
             for bal in start_balances:
                 # Allocated
                 if bal.allocated_balance and bal.allocated_balance > 0 and bal.pool_id:
                     pid = bal.pool_id
                     if pid not in portfolio_state: portfolio_state[pid] = {}
                     t = bal.token_symbol
                     portfolio_state[pid][t] = portfolio_state[pid].get(t, 0.0) + float(bal.allocated_balance)
                 
                 # Unallocated (Initial Wallet State)
                 if bal.unallocated_balance and bal.unallocated_balance > 0:
                     t = bal.token_symbol
                     wallet_state[t] = wallet_state.get(t, 0.0) + float(bal.unallocated_balance)
        
        # Replay transactions
        for txn in parsed_sequence:
            t_type = txn.get('type')
            amt = float(txn.get('amount', 0))
            token = txn.get('token')
            
            if t_type == 'WITHDRAWAL':
                pid = txn.get('from_location')
                if pid and pid in portfolio_state and token in portfolio_state[pid]:
                    # Remove from Pool
                    portfolio_state[pid][token] = max(0.0, portfolio_state[pid][token] - amt)
                    if portfolio_state[pid][token] <= 1e-9:
                        del portfolio_state[pid][token]
                    if not portfolio_state[pid]:
                        del portfolio_state[pid]
                    
                    # Add to Wallet
                    wallet_state[token] = wallet_state.get(token, 0.0) + amt
                        
            elif t_type == 'ALLOCATION':
                pid = txn.get('to_location')
                if pid:
                    # Add to Pool
                    if pid not in portfolio_state: portfolio_state[pid] = {}
                    portfolio_state[pid][token] = portfolio_state[pid].get(token, 0.0) + amt
                    
                    # Remove from Wallet
                    wallet_state[token] = max(0.0, wallet_state.get(token, 0.0) - amt)
                    
            elif t_type == 'HOLD':
                pid = txn.get('to_location') or txn.get('from_location')
                if pid:
                    if pid not in portfolio_state: portfolio_state[pid] = {}
                    if token not in portfolio_state[pid]:
                        portfolio_state[pid][token] = amt
            
            elif t_type == 'CONVERSION':
                from_t = txn.get('from_token') or txn.get('from_asset')
                to_t = txn.get('to_token') or txn.get('to_asset')
                
                if from_t and to_t:
                    # Remove from Wallet
                    wallet_state[from_t] = max(0.0, wallet_state.get(from_t, 0.0) - amt)
                    
                    # Add to Wallet (Estimate output based on price)
                    p_in = token_prices.get(from_t, 0.0)
                    p_out = token_prices.get(to_t, 0.0)
                    
                    if p_in > 0 and p_out > 0:
                        # Estimate output amount: (AmountIn * PriceIn * (1 - FeeRate)) / PriceOut
                        # Assuming standard fee/slippage of ~0.04% if not specified
                        val_usd = amt * p_in * (1 - 0.0004) 
                        amt_out = val_usd / p_out
                        wallet_state[to_t] = wallet_state.get(to_t, 0.0) + amt_out

        # Flatten for reporting
        final_allocations_list = []
        total_allocated_usd = 0.0
        
        # Add Pools
        for pid, tokens in portfolio_state.items():
            for t, amt in tokens.items():
                if amt > 1e-9:
                    price = token_prices.get(t, 1.0)
                    amt_usd = amt * price
                    final_allocations_list.append({
                        'pool_id': pid,
                        'token': t,
                        'amount': amt,
                        'amount_usd': amt_usd
                    })
                    total_allocated_usd += amt_usd
        
        # Add Wallet (Unallocated)
        for t, amt in wallet_state.items():
            if amt > 1e-9:
                price = token_prices.get(t, 1.0)
                amt_usd = amt * price
                final_allocations_list.append({
                    'pool_id': 'Unallocated',
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
        
        if 'Unallocated' in all_pool_ids:
            all_pool_ids.remove('Unallocated')
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
            'top_pools': pool_summary.to_dict('records'),
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
        if 'Unallocated' in pool_ids:
            pool_ids.remove('Unallocated')
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

        # Optimization Horizon (Use 30 as default to match optimizer, since not yet in DB)
        horizon = 30.0 
        
        return {
            'total_daily_yield': total_daily_yield,
            'total_annual_yield': total_annual_yield,
            'weighted_avg_apy': weighted_avg_apy,
            'net_daily_yield': total_daily_yield - (results['total_transaction_cost'] / horizon),
            'net_annual_yield': total_annual_yield - (results['total_transaction_cost'] * (365.0 / horizon)) 
        }
        
    except Exception as e:
        logger.error(f"Error calculating yield metrics: {e}")
        return {}


def create_markdown_table_block(df: pd.DataFrame, headers: list = None) -> dict:
    """
    Helper to create a Slack Markdown Code Block containing a dataframe table.
    Native Table blocks are not yet supported in Webhook messages.
    """
    if df.empty:
        return None
        
    # Pandas to_string usually does a good job
    # We can ensure headers are nice
    if headers:
        df.columns = headers
        
    table_str = df.to_string(index=False)
    
    return {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f"```\n{table_str}\n```"
        }
    }

def format_slack_message_blocks(results: Dict, yield_metrics: Dict) -> list:
    """Format results into Slack Block Kit blocks."""
    blocks = []
    date_str = datetime.now().strftime("%Y-%m-%d")
    pool_total_mapping = results.get('pool_total_mapping', {})
    
    total_aum = results['total_allocated']
    unallocated_mask = results['allocations_df']['pool_id'] == 'Unallocated'
    unallocated_usd = results['allocations_df'][unallocated_mask]['amount_usd'].sum()
    allocated_usd = total_aum - unallocated_usd

    # 1. Title
    blocks.append({
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": f"📊 Daily Allocation Report - {date_str}",
            "emoji": True
        }
    })
    
    # 2. Portfolio Summary (Section with Fields)
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": "*💰 Portfolio Summary*"}
    })
    
    summary_fields = [
        {"type": "mrkdwn", "text": f"*Total AUM:*\n${total_aum:,.2f}"},
        {"type": "mrkdwn", "text": f"*Allocated:*\n${allocated_usd:,.2f}"},
        {"type": "mrkdwn", "text": f"*Unallocated:*\n${unallocated_usd:,.2f}"},
        {"type": "mrkdwn", "text": f"*Transactions:*\n{results['transaction_count']}"}
    ]
    blocks.append({
        "type": "section",
        "fields": summary_fields
    })

    # 3. Yield Metrics
    if yield_metrics:
        blocks.append({"type": "divider"})
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*💵 Yield Metrics*"}
        })
        yield_fields = [
             {"type": "mrkdwn", "text": f"*Avg APY:*\n{yield_metrics.get('weighted_avg_apy', 0):.2f}%"},
             {"type": "mrkdwn", "text": f"*Daily Yield:*\n${yield_metrics.get('total_daily_yield', 0):.2f}"},
             {"type": "mrkdwn", "text": f"*Net Daily:*\n${yield_metrics.get('net_daily_yield', 0):.2f}"},
             {"type": "mrkdwn", "text": f"*Annual Yield:*\n${yield_metrics.get('total_annual_yield', 0):,.2f}"}
        ]
        blocks.append({"type": "section", "fields": yield_fields})

    # 4. Costs
    blocks.append({"type": "divider"})
    cost_text = f"*💸 Transaction Costs:* ${results['total_transaction_cost']:.4f}\n" \
                f"(Gas: ${results['total_gas_cost']:.4f} | Conv: ${results['total_conversion_cost']:.4f})"
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": cost_text}
    })

    # 5. Pool Allocations Table
    blocks.append({"type": "divider"})
    blocks.append({
        "type": "header",
        "text": {"type": "plain_text", "text": "🏆 Pool Allocations", "emoji": True}
    })

    pool_table_data = []
    pool_links = []
    
    for pool in results['top_pools']:
        pool_id = pool['pool_id']
        pool_total_amount = pool_total_mapping.get(pool_id, pool['amount_usd'])
        percentage = (pool_total_amount / total_aum) * 100 if total_aum > 0 else 0
        apy = pool.get('forecasted_apy', 0)
        tvl = pool.get('forecasted_tvl', 0)
        
        pool_table_data.append({
            "Pool": pool['symbol'],
            "ID": pool_id[:8] + "..",
            "Amt": f"${pool_total_amount:,.0f}",
            "%": f"{percentage:.1f}%",
            "APY": f"{apy:.2f}%",
            "TVL": f"${tvl/1e6:.1f}M" if tvl > 0 else "N/A"
        })
        
        # Links separate
        pool_url = f"https://defillama.com/yields/pool/{pool_id}"
        pool_links.append(f"• <{pool_url}|{pool['symbol']} ({pool_id})>")
    
    if pool_table_data:
        df_pools = pd.DataFrame(pool_table_data)
        blocks.append(create_markdown_table_block(df_pools))
        
        # Pool Links Context
        if pool_links:
             blocks.append({
                 "type": "context",
                 "elements": [{"type": "mrkdwn", "text": "*Pool Links:*\n" + "\n".join(pool_links)}]
             })

    # 6. Token Allocation Table
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": "*💎 Token Allocation*"}
    })
    token_table_data = []
    for token in results['token_allocation']:
        percentage = (token['amount_usd'] / results['total_allocated']) * 100 if results['total_allocated'] > 0 else 0
        token_table_data.append({
            "Token": token['token'],
            "Amt": f"${token['amount_usd']:,.0f}",
            "%": f"{percentage:.1f}%"
        })
    if token_table_data:
        df_tokens = pd.DataFrame(token_table_data)
        blocks.append(create_markdown_table_block(df_tokens))

    # 7. Transaction Sequence Table
    parsed_sequence = results.get('transaction_sequence', [])
    if parsed_sequence:
        blocks.append({"type": "divider"})
        blocks.append({
            "type": "header",
            "text": {"type": "plain_text", "text": "🔄 Transaction Sequence", "emoji": True}
        })
        
        txn_table_data = []
        pool_address_map = results.get('pool_address_map', {})
        
        type_map = {
             'ALLOCATION': 'ALLOC', 'WITHDRAWAL': 'WITHDR', 
             'CONVERSION': 'CONV', 'HOLD': 'HOLD'
        }

        for txn in parsed_sequence:
             t_type = txn.get('type')
             seq = txn.get('seq')
             amt = txn.get('amount_usd', 0)
             
             detail = ""
             if t_type == 'CONVERSION':
                 detail = f"{txn.get('from_token')}->{txn.get('to_token')}"
             else:
                 detail = f"{txn.get('token')}"
                 
             # Resolve address
             pool_id_ctx = txn.get('to_location') if t_type in ['ALLOCATION', 'HOLD'] else txn.get('from_location')
             addr_str = "-"
             if pool_id_ctx and pool_id_ctx != 'warm_wallet':
                  addr = pool_address_map.get(pool_id_ctx)
                  if addr:
                      addr_str = addr
                  else:
                      addr_str = pool_id_ctx
            
             txn_table_data.append({
                 "#": seq,
                 "Type": type_map.get(t_type, t_type),
                 "Details": detail,
                 "Amt": f"${amt:,.0f}",
                 "Address": addr_str
             })
             
        if txn_table_data:
            df_txn = pd.DataFrame(txn_table_data)
            # Full address can be long, but code block handles scrolling horizontally in Slack if needed
            # or just wraps. Standard markdown code block doesn't force wrap.
            blocks.append(create_markdown_table_block(df_txn))

    # Footer
    blocks.append({
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": f"Run ID: {results['run_id']} | Status: ✅ Success"}]
    })

    return blocks


def main():
    logger.info("Starting Slack notification process...")
    results = fetch_optimization_results()
    if not results:
        logger.error("No optimization results found.")
        return
        
    yield_metrics = calculate_yield_metrics(results)
    
    # Generate Blocks
    blocks = format_slack_message_blocks(results, yield_metrics)
    
    # Text fallback
    text_fallback = f"Daily Allocation Report: AUM ${results['total_allocated']:,.2f}"
    
    notifier = SlackNotifier()
    notifier.send_notification(blocks=blocks, text_fallback=text_fallback)
    logger.info("Slack notification process completed.")

if __name__ == "__main__":
    main()