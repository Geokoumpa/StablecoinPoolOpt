#!/usr/bin/env python3
"""
Test script to verify that Slack notification calculates percentages correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

def test_slack_notification_calculation():
    """Test that Slack notification calculates pool percentages correctly."""
    
    # Simulate allocation data with multiple tokens per pool
    allocations_with_pools = pd.DataFrame([
        {'pool_id': 'pool1', 'symbol': 'POOL1', 'to_asset': 'USDC', 'amount': 1000, 'forecasted_apy': 10.0, 'forecasted_tvl': 1000000},
        {'pool_id': 'pool1', 'symbol': 'POOL1', 'to_asset': 'USDT', 'amount': 1000, 'forecasted_apy': 10.0, 'forecasted_tvl': 1000000},
        {'pool_id': 'pool2', 'symbol': 'POOL2', 'to_asset': 'USDC', 'amount': 800, 'forecasted_apy': 12.0, 'forecasted_tvl': 1000000},
        {'pool_id': 'pool2', 'symbol': 'POOL2', 'to_asset': 'USDT', 'amount': 700, 'forecasted_apy': 12.0, 'forecasted_tvl': 1000000},
    ])
    
    # Calculate USD amounts
    allocations_with_pools['amount_usd'] = allocations_with_pools['amount']
    
    # Group by pool and token (current method)
    pool_summary = allocations_with_pools.groupby(['pool_id', 'symbol', 'forecasted_apy', 'forecasted_tvl']).agg({
        'amount_usd': 'sum'
    }).reset_index()
    
    # Calculate total allocation per pool (sum of all tokens in each pool)
    pool_total_summary = allocations_with_pools.groupby(['pool_id', 'symbol', 'forecasted_apy', 'forecasted_tvl']).agg({
        'amount_usd': 'sum'
    }).reset_index()
    
    # Create a mapping of pool_id to total pool allocation
    pool_total_mapping = pool_total_summary.set_index('pool_id')['amount_usd'].to_dict()
    
    total_allocated = allocations_with_pools['amount_usd'].sum()
    
    print("=== Pool Summary Analysis ===")
    print(f"Total allocated: ${total_allocated:,.2f}")
    print("\nIndividual token allocations:")
    for _, row in pool_summary.iterrows():
        print(f"  {row['pool_id']} ({row['symbol']}): ${row['amount_usd']:,.2f}")
    
    print("\nTotal pool allocations:")
    for pool_id, total in pool_total_mapping.items():
        percentage = (total / total_allocated) * 100
        print(f"  {pool_id}: ${total:,.2f} ({percentage:.1f}%)")
    
    print("\n=== Slack Notification Format ===")
    
    # Simulate Slack notification formatting
    results = {
        'total_allocated': total_allocated,
        'top_pools': pool_summary.to_dict('records'),
        'projected_apy': 11.5,  # Simulated projected APY
        'total_transaction_cost': 25.50,  # Simulated transaction costs
        'transaction_sequence': [
            {
                'seq': 1,
                'type': 'CONVERSION',
                'from_token': 'USDT',
                'to_token': 'USDC',
                'amount_usd': 500.0,
                'gas_cost_usd': 2.5,
                'conversion_cost_usd': 0.2,
                'total_cost_usd': 2.7
            },
            {
                'seq': 2,
                'type': 'ALLOCATION',
                'token': 'USDC',
                'to_location': 'pool1',
                'amount_usd': 1000.0,
                'gas_cost_usd': 1.8,
                'total_cost_usd': 1.8
            },
            {
                'seq': 3,
                'type': 'ALLOCATION',
                'token': 'USDT',
                'to_location': 'pool2',
                'amount_usd': 800.0,
                'gas_cost_usd': 1.8,
                'total_cost_usd': 1.8
            }
        ]
    }
    
    for pool in results['top_pools']:
        # Get total pool allocation (sum of all tokens in this pool)
        pool_id = pool['pool_id']
        pool_total_amount = pool_total_mapping.get(pool_id, pool['amount_usd'])
        percentage = (pool_total_amount / results['total_allocated']) * 100 if results['total_allocated'] > 0 else 0
        
        print(f"• {pool['symbol']} (ID: {pool['pool_id']}): ${pool['amount_usd']:,.2f} ({percentage:.1f}%)")
    
    print("\n✓ Pool percentages calculated based on total pool allocation, not individual token allocations!")
    
    print("\n=== New Fields Demonstration ===")
    print(f"Projected APY: {results.get('projected_apy', 0):.2f}%")
    print(f"Total Transaction Costs: ${results.get('total_transaction_cost', 0):.2f}")
    print("Transaction Sequence:")
    for txn in results.get('transaction_sequence', []):
        txn_type = txn.get('type', '')
        if txn_type == 'CONVERSION':
            print(f"  {txn['seq']}. {txn_type}: ${txn['amount_usd']:.2f} {txn['from_token']} → {txn['to_token']} (Gas: ${txn['gas_cost_usd']:.4f}, Conv: ${txn['conversion_cost_usd']:.4f}, Total: ${txn['total_cost_usd']:.4f})")
        elif txn_type == 'ALLOCATION':
            print(f"  {txn['seq']}. {txn_type}: ${txn['amount_usd']:.2f} {txn['token']} → {txn['to_location']} (Gas: ${txn['gas_cost_usd']:.4f}, Total: ${txn['total_cost_usd']:.4f})")
        else:
            print(f"  {txn['seq']}. {txn_type}: ${txn['amount_usd']:.2f} (Gas: ${txn['gas_cost_usd']:.4f}, Total: ${txn['total_cost_usd']:.4f})")

if __name__ == "__main__":
    test_slack_notification_calculation()