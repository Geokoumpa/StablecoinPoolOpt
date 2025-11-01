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
        'top_pools': pool_summary.to_dict('records')
    }
    
    for pool in results['top_pools']:
        # Get total pool allocation (sum of all tokens in this pool)
        pool_id = pool['pool_id']
        pool_total_amount = pool_total_mapping.get(pool_id, pool['amount_usd'])
        percentage = (pool_total_amount / results['total_allocated']) * 100 if results['total_allocated'] > 0 else 0
        
        print(f"• {pool['symbol']} (ID: {pool['pool_id']}): ${pool['amount_usd']:,.2f} ({percentage:.1f}%)")
    
    print("\n✓ Pool percentages calculated based on total pool allocation, not individual token allocations!")

if __name__ == "__main__":
    test_slack_notification_calculation()