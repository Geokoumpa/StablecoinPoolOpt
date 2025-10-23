#!/usr/bin/env python3
"""
Test script to verify TVL limit constraint implementation.

This script tests that the optimization algorithm correctly applies the TVL limit
constraint when allocating assets to pools.
"""

import logging
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asset_allocation.optimize_allocations import AllocationOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_data():
    """Create test data for TVL limit constraint testing."""
    
    # Create test pools with different TVL values and higher APY
    pools_df = pd.DataFrame([
        {
            'pool_id': 'pool_1',
            'symbol': 'USDC',
            'chain': 'ethereum',
            'protocol': 'compound',
            'forecasted_apy': 10.0,  # Higher APY
            'forecasted_tvl': 1000000  # $1M TVL
        },
        {
            'pool_id': 'pool_2',
            'symbol': 'USDT',
            'chain': 'ethereum',
            'protocol': 'aave',
            'forecasted_apy': 12.0,  # Higher APY
            'forecasted_tvl': 500000   # $500K TVL
        },
        {
            'pool_id': 'pool_3',
            'symbol': 'DAI',
            'chain': 'ethereum',
            'protocol': 'curve',
            'forecasted_apy': 15.0,  # Higher APY
            'forecasted_tvl': 2000000  # $2M TVL
        }
    ])
    
    # Token prices (all stablecoins at $1)
    token_prices = {
        'USDC': 1.0,
        'USDT': 1.0,
        'DAI': 1.0
    }
    
    # Warm wallet with significant funds to test constraints
    warm_wallet = {
        'USDC': 500000,  # $500K
        'USDT': 300000,  # $300K
        'DAI': 200000    # $200K
    }
    
    # No current allocations
    current_allocations = {}
    
    # Low gas fee for testing
    gas_fee_usd = 1.0  # Very low gas fee
    
    # Allocation parameters with TVL limit
    alloc_params = {
        'tvl_limit_percentage': 0.05,  # 5% of pool TVL
        'max_alloc_percentage': 0.50,   # 50% of total AUM (high enough not to be binding)
        'conversion_rate': 0.0001,     # Lower conversion rate
        'min_transaction_value': 10.0   # Lower minimum transaction
    }
    
    return pools_df, token_prices, warm_wallet, current_allocations, gas_fee_usd, alloc_params


def test_tvl_limit_constraint():
    """Test that TVL limit constraint is properly applied."""
    
    logger.info("Testing TVL limit constraint implementation...")
    
    # Create test data
    pools_df, token_prices, warm_wallet, current_allocations, gas_fee_usd, alloc_params = create_test_data()
    
    # Initialize optimizer
    optimizer = AllocationOptimizer(
        pools_df=pools_df,
        token_prices=token_prices,
        warm_wallet=warm_wallet,
        current_allocations=current_allocations,
        gas_fee_usd=gas_fee_usd,
        alloc_params=alloc_params
    )
    
    # Build and solve the model
    logger.info("Building optimization model...")
    problem = optimizer.build_model()
    
    logger.info("Solving optimization problem...")
    success = optimizer.solve()
    
    if not success:
        logger.error("Optimization failed!")
        return False
    
    # Extract results
    allocations_df, transactions = optimizer.extract_results()
    
    if allocations_df.empty:
        logger.warning("No allocations found in solution")
        return True
    
    # Check TVL constraints
    logger.info("\n=== Checking TVL Limit Constraints ===")
    
    constraints_satisfied = True
    
    for _, pool in pools_df.iterrows():
        pool_id = pool['pool_id']
        pool_tvl = pool['forecasted_tvl']
        tvl_limit = pool_tvl * alloc_params['tvl_limit_percentage']
        
        # Get total allocation to this pool
        pool_allocations = allocations_df[allocations_df['pool_id'] == pool_id]
        total_allocated_usd = pool_allocations['amount_usd'].sum()
        
        logger.info(f"\nPool: {pool_id}")
        logger.info(f"  Pool TVL: ${pool_tvl:,.2f}")
        logger.info(f"  TVL Limit (5%): ${tvl_limit:,.2f}")
        logger.info(f"  Total Allocated: ${total_allocated_usd:,.2f}")
        
        if total_allocated_usd > tvl_limit + 0.01:  # Small tolerance for numerical precision
            logger.error(f"  ❌ TVL limit violated! Allocation exceeds limit by ${total_allocated_usd - tvl_limit:,.2f}")
            constraints_satisfied = False
        else:
            logger.info(f"  ✅ TVL constraint satisfied")
    
    # Summary
    total_aum = sum(warm_wallet.values())
    logger.info(f"\n=== Summary ===")
    logger.info(f"Total AUM: ${total_aum:,.2f}")
    logger.info(f"TVL Limit Percentage: {alloc_params['tvl_limit_percentage']:.1%}")
    logger.info(f"Total allocated: ${allocations_df['amount_usd'].sum():,.2f}")
    
    if constraints_satisfied:
        logger.info("\n✅ All TVL limit constraints are satisfied!")
    else:
        logger.error("\n❌ Some TVL limit constraints were violated!")
    
    return constraints_satisfied


def test_tvl_limit_with_different_percentages():
    """Test TVL limit constraint with different percentage values."""
    
    logger.info("\n=== Testing Different TVL Limit Percentages ===")
    
    # Create test data
    pools_df, token_prices, warm_wallet, current_allocations, gas_fee_usd, _ = create_test_data()
    
    test_percentages = [0.01, 0.05, 0.10, 0.20]  # 1%, 5%, 10%, 20%
    
    for tvl_limit_pct in test_percentages:
        logger.info(f"\nTesting with TVL limit = {tvl_limit_pct:.1%}")
        
        alloc_params = {
            'tvl_limit_percentage': tvl_limit_pct,
            'max_alloc_percentage': 0.50,
            'conversion_rate': 0.0004,
            'min_transaction_value': 50.0
        }
        
        optimizer = AllocationOptimizer(
            pools_df=pools_df,
            token_prices=token_prices,
            warm_wallet=warm_wallet,
            current_allocations=current_allocations,
            gas_fee_usd=gas_fee_usd,
            alloc_params=alloc_params
        )
        
        success = optimizer.solve()
        
        if success:
            allocations_df, _ = optimizer.extract_results()
            if allocations_df.empty:
                logger.info(f"  No allocations made")
                continue
                
            total_allocated = allocations_df['amount_usd'].sum()
            logger.info(f"  Total allocated: ${total_allocated:,.2f}")
            
            # Check if allocations respect TVL limits
            for _, pool in pools_df.iterrows():
                pool_id = pool['pool_id']
                pool_tvl = pool['forecasted_tvl']
                tvl_limit = pool_tvl * tvl_limit_pct
                
                pool_allocations = allocations_df[allocations_df['pool_id'] == pool_id]
                total_pool_allocated = pool_allocations['amount_usd'].sum()
                
                if total_pool_allocated > tvl_limit + 0.01:
                    logger.error(f"  ❌ Pool {pool_id} exceeded TVL limit")
        else:
            logger.warning(f"  Optimization failed with TVL limit {tvl_limit_pct:.1%}")


if __name__ == "__main__":
    logger.info("Starting TVL limit constraint tests...")
    
    # Run main test
    success = test_tvl_limit_constraint()
    
    # Run additional tests
    test_tvl_limit_with_different_percentages()
    
    if success:
        logger.info("\n✅ All tests passed!")
        sys.exit(0)
    else:
        logger.error("\n❌ Some tests failed!")
        sys.exit(1)