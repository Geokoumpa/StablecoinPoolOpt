#!/usr/bin/env python3
"""
Test script to forecast a single pool to verify the fix.
"""

import sys
sys.path.append('.')

from forecasting.forecast_pools import train_and_forecast_pool, get_filtered_pool_ids

if __name__ == "__main__":
    # Get the first filtered pool to test
    filtered_pool_ids = get_filtered_pool_ids()
    if not filtered_pool_ids:
        print("No filtered pools found in the database.")
        exit(1)
    
    # Test with just the first pool
    test_pool_id = filtered_pool_ids[0]
    print(f"Testing forecasting with pool: {test_pool_id}")
    
    try:
        result = train_and_forecast_pool(test_pool_id, steps=1)
        if result:
            print(f"✅ Successfully forecasted pool {test_pool_id}")
            print(f"Forecast APY: {result.get('forecast_apy', {})}")
            print(f"Forecast TVL: {result.get('forecast_tvl', {})}")
        else:
            print(f"⏭️ Pool {test_pool_id} was skipped (insufficient data)")
    except Exception as e:
        print(f"❌ Error forecasting pool {test_pool_id}: {e}")
        import traceback
        traceback.print_exc()