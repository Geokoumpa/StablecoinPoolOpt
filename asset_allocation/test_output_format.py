#!/usr/bin/env python3
"""
Test script to validate the output format of the optimization results.
This script creates mock data and tests the format_results() method.
"""

import sys
import pandas as pd
import numpy as np
import json
from unittest.mock import Mock
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the path to import the optimization module
sys.path.append('..')

from asset_allocation.optimize_allocations import AllocationOptimizer


def create_mock_data():
    """Creates mock data for testing the output format."""
    
    # Mock pool data
    pools_df = pd.DataFrame([
        {
            'pool_id': 'pool-001',
            'symbol': 'USDC-USDT',
            'chain': 'ethereum',
            'protocol': 'curve',
            'forecasted_apy': 5.5
        },
        {
            'pool_id': 'pool-002', 
            'symbol': 'DAI-USDC',
            'chain': 'ethereum',
            'protocol': 'curve',
            'forecasted_apy': 6.2
        },
        {
            'pool_id': 'pool-003',
            'symbol': 'BOLD',
            'chain': 'ethereum',
            'protocol': 'bold',
            'forecasted_apy': 8.1
        }
    ])
    
    # Mock token prices
    token_prices = {
        'USDC': 1.0,
        'USDT': 1.0,
        'DAI': 1.0,
        'BOLD': 1.0
    }
    
    # Mock warm wallet balances
    warm_wallet = {
        'USDC': 1000.0,
        'USDT': 500.0,
        'DAI': 200.0
    }
    
    # Mock current allocations
    current_allocations = {
        ('pool-001', 'USDC'): 500.0,
        ('pool-001', 'USDT'): 500.0,
        ('pool-003', 'BOLD'): 300.0
    }
    
    # Mock allocation parameters
    alloc_params = {
        'max_alloc_percentage': 0.20,
        'conversion_rate': 0.0004,
        'min_transaction_value': 50.0
    }
    
    return pools_df, token_prices, warm_wallet, current_allocations, alloc_params


def create_mock_optimizer():
    """Creates a mock optimizer with solved variables."""
    
    pools_df, token_prices, warm_wallet, current_allocations, alloc_params = create_mock_data()
    
    # Create optimizer instance
    optimizer = AllocationOptimizer(
        pools_df=pools_df,
        token_prices=token_prices,
        warm_wallet=warm_wallet,
        current_allocations=current_allocations,
        gas_fee_usd=5.0,
        alloc_params=alloc_params
    )
    
    # Mock the solved variables
    # Create mock allocation matrix
    n_pools = len(optimizer.pools)
    n_tokens = len(optimizer.tokens)
    
    # Mock final allocations
    alloc_values = np.zeros((n_pools, n_tokens))
    alloc_values[0, 0] = 800.0  # pool-001, USDC
    alloc_values[0, 1] = 800.0  # pool-001, USDT  
    alloc_values[1, 2] = 400.0  # pool-002, DAI
    alloc_values[1, 0] = 400.0  # pool-002, USDC
    alloc_values[2, 3] = 500.0  # pool-003, BOLD
    
    # Mock withdrawals
    withdraw_values = np.zeros((n_pools, n_tokens))
    withdraw_values[0, 0] = 100.0  # Withdraw 100 USDC from pool-001
    withdraw_values[2, 3] = 50.0   # Withdraw 50 BOLD from pool-003
    
    # Mock conversions
    convert_values = np.zeros((n_tokens, n_tokens))
    convert_values[1, 2] = 100.0  # Convert 100 USDT to DAI
    
    # Mock final warm wallet balances
    final_warm_wallet_values = np.zeros(n_tokens)
    final_warm_wallet_values[0] = 200.0  # 200 USDC remaining
    final_warm_wallet_values[1] = 300.0  # 300 USDT remaining
    final_warm_wallet_values[2] = 100.0  # 100 DAI remaining
    
    # Set the mock values
    optimizer.alloc = Mock()
    optimizer.alloc.value = alloc_values
    optimizer.withdraw = Mock()
    optimizer.withdraw.value = withdraw_values
    optimizer.convert = Mock()
    optimizer.convert.value = convert_values
    optimizer.final_warm_wallet = Mock()
    optimizer.final_warm_wallet.value = final_warm_wallet_values
    
    return optimizer


def validate_output_format(results):
    """Validates that the output format matches the requirements."""
    
    logger.info("Validating output format...")
    
    # Check top-level structure
    required_keys = ['final_allocations', 'unallocated_tokens', 'transactions']
    for key in required_keys:
        if key not in results:
            logger.error(f"Missing required key: {key}")
            return False
        logger.info(f"✓ Found required key: {key}")
    
    # Validate final allocations
    final_allocations = results['final_allocations']
    if not isinstance(final_allocations, dict):
        logger.error("final_allocations should be a dictionary")
        return False
    
    for pool_id, pool_data in final_allocations.items():
        if not isinstance(pool_data, dict):
            logger.error(f"Pool data for {pool_id} should be a dictionary")
            return False
        
        if 'pool_symbol' not in pool_data:
            logger.error(f"Missing pool_symbol for pool {pool_id}")
            return False
        
        if 'tokens' not in pool_data:
            logger.error(f"Missing tokens for pool {pool_id}")
            return False
        
        tokens = pool_data['tokens']
        if not isinstance(tokens, dict):
            logger.error(f"Tokens for pool {pool_id} should be a dictionary")
            return False
        
        for token, token_data in tokens.items():
            if not isinstance(token_data, dict):
                logger.error(f"Token data for {token} should be a dictionary")
                return False
            
            if 'amount' not in token_data or 'amount_usd' not in token_data:
                logger.error(f"Missing amount or amount_usd for token {token}")
                return False
    
    logger.info("✓ final_allocations format is valid")
    
    # Validate unallocated tokens
    unallocated_tokens = results['unallocated_tokens']
    if not isinstance(unallocated_tokens, dict):
        logger.error("unallocated_tokens should be a dictionary")
        return False
    
    for token, token_data in unallocated_tokens.items():
        if not isinstance(token_data, dict):
            logger.error(f"Token data for {token} should be a dictionary")
            return False
        
        if 'amount' not in token_data or 'amount_usd' not in token_data:
            logger.error(f"Missing amount or amount_usd for unallocated token {token}")
            return False
    
    logger.info("✓ unallocated_tokens format is valid")
    
    # Validate transactions
    transactions = results['transactions']
    if not isinstance(transactions, list):
        logger.error("transactions should be a list")
        return False
    
    for i, txn in enumerate(transactions):
        if not isinstance(txn, dict):
            logger.error(f"Transaction {i} should be a dictionary")
            return False
        
        required_txn_fields = ['seq', 'type', 'from_location', 'to_location', 'amount', 'amount_usd', 'gas_cost_usd']
        for field in required_txn_fields:
            if field not in txn:
                logger.error(f"Missing required field {field} in transaction {i}")
                return False
        
        # Check for token field in non-conversion transactions
        if txn['type'] != 'CONVERSION' and 'token' not in txn:
            logger.error(f"Missing token field in non-conversion transaction {i}")
            return False
        
        # Check for conversion-specific fields
        if txn['type'] == 'CONVERSION':
            if 'from_token' not in txn or 'to_token' not in txn:
                logger.error(f"Missing from_token or to_token in conversion transaction {i}")
                return False
    
    logger.info("✓ transactions format is valid")
    
    return True


def main():
    """Main test function."""
    
    logger.info("=" * 80)
    logger.info("TESTING OUTPUT FORMAT COMPLIANCE")
    logger.info("=" * 80)
    
    try:
        # Create mock optimizer
        logger.info("\n[1/3] Creating mock optimizer...")
        optimizer = create_mock_optimizer()
        logger.info(f"Created optimizer with {len(optimizer.pools)} pools and {len(optimizer.tokens)} tokens")
        
        # Test format_results
        logger.info("\n[2/3] Testing format_results()...")
        results = optimizer.format_results()
        
        # Print results for visual inspection
        logger.info("\n[3/3] Validating and displaying results...")
        logger.info("\n" + "=" * 80)
        logger.info("FORMATTED RESULTS")
        logger.info("=" * 80)
        
        print(json.dumps(results, indent=2))
        
        # Validate format
        if validate_output_format(results):
            logger.info("\n✓ OUTPUT FORMAT VALIDATION PASSED")
            logger.info("The output format matches the requirements in optimization.md")
        else:
            logger.error("\n✗ OUTPUT FORMAT VALIDATION FAILED")
            logger.error("The output format does not match the requirements")
            return False
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ TEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)