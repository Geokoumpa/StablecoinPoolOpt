#!/usr/bin/env python3
"""
Test script for Phase 3: Gas Fee & Cost Calculations

This script tests the updated gas fee and cost calculation implementation
to ensure it matches the requirements in optimization.md.
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def create_test_data():
    """Create test data for Phase 3 validation"""
    
    # Test pools with different APYs
    pools_df = pd.DataFrame([
        {'pool_id': 'pool1', 'symbol': 'USDC', 'chain': 'ethereum', 'protocol': 'curve', 'forecasted_apy': 0.05},
        {'pool_id': 'pool2', 'symbol': 'USDT', 'chain': 'ethereum', 'protocol': 'curve', 'forecasted_apy': 0.045},
        {'pool_id': 'pool3', 'symbol': 'USDC-USDT', 'chain': 'ethereum', 'protocol': 'curve', 'forecasted_apy': 0.055}
    ])
    
    # Token prices
    token_prices = {
        'USDC': 1.0,
        'USDT': 1.0,
        'ETH': 3000.0
    }
    
    # Warm wallet balances
    warm_wallet = {
        'USDC': 1000.0,
        'USDT': 500.0
    }
    
    # Current allocations
    current_allocations = {
        ('pool1', 'USDC'): 200.0,
        ('pool2', 'USDT'): 100.0
    }
    
    # Gas fee in USD
    gas_fee_usd = 5.0
    
    # Allocation parameters
    alloc_params = {
        'max_alloc_percentage': 0.20,
        'conversion_rate': 0.0004,
        'min_transaction_value': 50.0
    }
    
    return pools_df, token_prices, warm_wallet, current_allocations, gas_fee_usd, alloc_params

def test_cost_formulas():
    """Test that the cost formulas match the requirements"""
    
    logger.info("=" * 80)
    logger.info("TESTING PHASE 3: GAS FEE & COST CALCULATIONS")
    logger.info("=" * 80)
    
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
    
    # Test 1: Check that binary variables are created
    logger.info("\n[TEST 1] Binary Variables Creation")
    # Build model first to create the binary variables
    problem = optimizer.build_model()
    assert problem is not None, "Model building failed"
    
    assert hasattr(optimizer, 'needs_conversion'), "needs_conversion binary variable not created"
    assert hasattr(optimizer, 'has_allocation'), "has_allocation binary variable not created"
    assert hasattr(optimizer, 'is_withdrawal'), "is_withdrawal binary variable not created"
    assert hasattr(optimizer, 'is_conversion'), "is_conversion binary variable not created"
    assert optimizer.needs_conversion.shape == (optimizer.n_pools, optimizer.n_tokens), "needs_conversion has wrong shape"
    assert optimizer.has_allocation.shape == (optimizer.n_pools,), "has_allocation has wrong shape"
    assert optimizer.is_withdrawal.shape == (optimizer.n_pools, optimizer.n_tokens), "is_withdrawal has wrong shape"
    assert optimizer.is_conversion.shape == (optimizer.n_tokens, optimizer.n_tokens), "is_conversion has wrong shape"
    logger.info("✓ Binary variables created correctly")
    
    # Test 2: Check constraints
    logger.info("\n[TEST 2] Model Constraints Validation")
    logger.info(f"✓ Model built with {len(problem.constraints)} constraints")
    
    # Test 3: Solve optimization
    logger.info("\n[TEST 3] Optimization Solving")
    success = optimizer.solve(verbose=False)
    if not success:
        logger.warning("Optimization failed, but this might be due to test data")
        return False
    
    logger.info("✓ Optimization solved successfully")
    
    # Test 4: Extract and validate results
    logger.info("\n[TEST 4] Results Validation")
    allocations_df, transactions = optimizer.extract_results()
    
    # Check that transactions have the new cost fields
    for txn in transactions:
        assert 'conversion_cost_usd' in txn, f"Transaction missing conversion_cost_usd: {txn}"
        assert 'total_cost_usd' in txn, f"Transaction missing total_cost_usd: {txn}"
        
        # Validate cost formulas based on transaction type
        if txn['type'] == 'WITHDRAWAL':
            expected_cost = txn['amount_usd'] * alloc_params['conversion_rate'] + gas_fee_usd
            actual_cost = txn['total_cost_usd']
            assert abs(actual_cost - expected_cost) < 0.01, f"Withdrawal cost mismatch: expected {expected_cost}, got {actual_cost}"
            
        elif txn['type'] == 'CONVERSION':
            expected_cost = txn['amount_usd'] * alloc_params['conversion_rate'] + gas_fee_usd
            actual_cost = txn['total_cost_usd']
            assert abs(actual_cost - expected_cost) < 0.01, f"Conversion cost mismatch: expected {expected_cost}, got {actual_cost}"
            
        elif txn['type'] == 'ALLOCATION':
            conversion_cost = txn['amount_usd'] * alloc_params['conversion_rate']
            if txn.get('needs_conversion', False):
                expected_gas = 2 * gas_fee_usd
            else:
                expected_gas = gas_fee_usd
            expected_total = conversion_cost + expected_gas
            actual_total = txn['total_cost_usd']
            assert abs(actual_total - expected_total) < 0.01, f"Allocation cost mismatch: expected {expected_total}, got {actual_total}"
    
    logger.info(f"✓ {len(transactions)} transactions validated with correct cost formulas")
    
    # Test 5: Validate formatted results
    logger.info("\n[TEST 5] Formatted Results Validation")
    formatted_results = optimizer.format_results()
    
    assert 'final_allocations' in formatted_results, "Missing final_allocations in results"
    assert 'unallocated_tokens' in formatted_results, "Missing unallocated_tokens in results"
    assert 'transactions' in formatted_results, "Missing transactions in results"
    
    # Check transactions have all required fields
    for txn in formatted_results['transactions']:
        required_fields = ['seq', 'type', 'from_location', 'to_location', 'amount', 'amount_usd', 'gas_cost_usd']
        for field in required_fields:
            assert field in txn, f"Transaction missing required field: {field}"
        
        # Check cost fields are present where expected
        if txn['type'] in ['WITHDRAWAL', 'CONVERSION', 'ALLOCATION']:
            assert 'conversion_cost_usd' in txn, f"{txn['type']} missing conversion_cost_usd"
            assert 'total_cost_usd' in txn, f"{txn['type']} missing total_cost_usd"
    
    logger.info("✓ Formatted results contain all required fields")
    
    # Print summary of results
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3 TEST RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total pools: {len(pools_df)}")
    logger.info(f"Total tokens: {len(optimizer.tokens)}")
    logger.info(f"Final allocations: {len(formatted_results['final_allocations'])}")
    logger.info(f"Unallocated tokens: {len(formatted_results['unallocated_tokens'])}")
    logger.info(f"Total transactions: {len(formatted_results['transactions'])}")
    
    # Calculate total costs
    total_conversion_costs = sum(txn.get('conversion_cost_usd', 0) for txn in formatted_results['transactions'])
    total_gas_costs = sum(txn.get('gas_cost_usd', 0) for txn in formatted_results['transactions'])
    total_costs = sum(txn.get('total_cost_usd', 0) for txn in formatted_results['transactions'])
    
    logger.info(f"\nCost Breakdown:")
    logger.info(f"  Total conversion costs: ${total_conversion_costs:.4f}")
    logger.info(f"  Total gas costs: ${total_gas_costs:.4f}")
    logger.info(f"  Total transaction costs: ${total_costs:.4f}")
    
    return True

def main():
    """Main test function"""
    try:
        success = test_cost_formulas()
        if success:
            logger.info("\n✓ PHASE 3 TESTS PASSED - Gas fee and cost calculations working correctly")
            return 0
        else:
            logger.error("\n✗ PHASE 3 TESTS FAILED")
            return 1
    except Exception as e:
        logger.error(f"\n✗ PHASE 3 TESTS FAILED WITH ERROR: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())