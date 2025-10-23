#!/usr/bin/env python3
"""
Diagnostic Comparison Test

This script compares the characteristics of successful tests with real data tests
to identify what's causing the solver failures.
"""

import logging
import pandas as pd
import numpy as np
import json
import sys
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asset_allocation.optimize_allocations import (
    AllocationOptimizer,
    fetch_pool_data,
    fetch_token_prices,
    fetch_gas_fee_data,
    fetch_allocation_parameters,
    calculate_aum
)
from database.db_utils import get_db_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_successful_test_data():
    """Create test data that we know works (from test_phase4_comprehensive_low_costs)"""
    
    # Pools that worked in our successful test (APY values as percentages like real data)
    pools_df = pd.DataFrame([
        {'pool_id': 'pool-low-1', 'symbol': 'USDC', 'chain': 'ethereum', 'protocol': 'aave', 'forecasted_apy': 1.2},
        {'pool_id': 'pool-low-2', 'symbol': 'USDT', 'chain': 'ethereum', 'protocol': 'compound', 'forecasted_apy': 1.5},
        {'pool_id': 'pool-low-3', 'symbol': 'DAI', 'chain': 'ethereum', 'protocol': 'aave', 'forecasted_apy': 1.8},
        {'pool_id': 'pool-med-1', 'symbol': 'USDC', 'chain': 'ethereum', 'protocol': 'curve', 'forecasted_apy': 3.5},
        {'pool_id': 'pool-med-2', 'symbol': 'USDT', 'chain': 'ethereum', 'protocol': 'curve', 'forecasted_apy': 3.8},
        {'pool_id': 'pool-med-3', 'symbol': 'DAI', 'chain': 'ethereum', 'protocol': 'compound', 'forecasted_apy': 4.2},
        {'pool_id': 'pool-high-1', 'symbol': 'USDC', 'chain': 'ethereum', 'protocol': 'balancer', 'forecasted_apy': 7.5},
        {'pool_id': 'pool-high-2', 'symbol': 'USDT', 'chain': 'ethereum', 'protocol': 'yearn', 'forecasted_apy': 8.2},
        {'pool_id': 'pool-high-3', 'symbol': 'DAI', 'chain': 'ethereum', 'protocol': 'yearn', 'forecasted_apy': 7.8},
        {'pool_id': 'pool-vhigh-1', 'symbol': 'USDC-USDT', 'chain': 'ethereum', 'protocol': 'curve', 'forecasted_apy': 9.5},
        {'pool_id': 'pool-vhigh-2', 'symbol': 'DAI-USDC', 'chain': 'ethereum', 'protocol': 'balancer', 'forecasted_apy': 9.2},
        {'pool_id': 'pool-vhigh-3', 'symbol': 'USDT-DAI', 'chain': 'ethereum', 'protocol': 'curve', 'forecasted_apy': 9.8},
        {'pool_id': 'pool-prem-1', 'symbol': 'USDC-USDT-DAI', 'chain': 'ethereum', 'protocol': 'curve', 'forecasted_apy': 10.5},
        {'pool_id': 'pool-prem-2', 'symbol': 'USDC-DAI', 'chain': 'ethereum', 'protocol': 'balancer', 'forecasted_apy': 10.2},
        {'pool_id': 'pool-prem-3', 'symbol': 'USDT-USDC', 'chain': 'ethereum', 'protocol': 'yearn', 'forecasted_apy': 10.8}
    ])
    
    # Token prices
    token_prices = {
        'USDC': 1.00,
        'USDT': 0.999,
        'DAI': 1.001,
        'ETH': 3200.0,
        'WBTC': 45000.0
    }
    
    # Warm wallet balances
    warm_wallet = {
        'USDC': 15000.0,
        'USDT': 12000.0,
        'DAI': 10000.0,
        'ETH': 5.0,
        'WBTC': 0.1
    }
    
    # Current allocations
    current_allocations = {
        ('pool-low-1', 'USDC'): 3000.0,
        ('pool-low-2', 'USDT'): 2500.0,
        ('pool-low-3', 'DAI'): 2000.0,
        ('pool-med-1', 'USDC'): 1500.0,
        ('pool-med-2', 'USDT'): 1000.0,
        ('pool-vhigh-1', 'USDC'): 500.0,
        ('pool-vhigh-1', 'USDT'): 300.0
    }
    
    # Low transaction costs that worked
    gas_fee_usd = 0.1
    alloc_params = {
        'max_alloc_percentage': 0.25,
        'conversion_rate': 0.00005,
        'min_transaction_value': 25.0
    }
    
    return pools_df, token_prices, warm_wallet, current_allocations, gas_fee_usd, alloc_params


def create_real_pool_data():
    """Create test data with real pools that are failing"""
    
    # Connect to database
    engine = get_db_connection()
    
    # Fetch real data
    pools_df = fetch_pool_data(engine)
    
    # If no pools for today, use fallback
    if len(pools_df) == 0:
        fallback_query = """
        SELECT DISTINCT ON (pdm.pool_id)
            pdm.pool_id,
            p.symbol,
            p.chain,
            p.protocol,
            pdm.forecasted_apy
        FROM pool_daily_metrics pdm
        JOIN pools p ON pdm.pool_id = p.pool_id
        WHERE pdm.date = CURRENT_DATE - INTERVAL '1 day'
          AND pdm.forecasted_apy IS NOT NULL
          AND pdm.forecasted_apy > 0
        ORDER BY pdm.pool_id, pdm.forecasted_apy DESC
        LIMIT 25
        """
        pools_df = pd.read_sql(fallback_query, engine)
    
    # Define common tokens
    common_tokens = ['USDC', 'USDT', 'DAI', 'ETH', 'WBTC', 'WETH', 'BOLD', 'GHO', 'USDE', 'CRVUSD']
    token_prices = fetch_token_prices(engine, common_tokens)
    
    # Get gas fee data
    gas_gwei, eth_price = fetch_gas_fee_data(engine)
    gas_fee_usd = gas_gwei * 1e-9 * eth_price
    
    # Get allocation parameters
    alloc_params = fetch_allocation_parameters(engine)
    
    # Create mock data
    pool_tokens = {}
    for _, pool in pools_df.iterrows():
        pool_tokens[pool['pool_id']] = pool['symbol'].split('-')
    
    # Get all unique tokens from pools
    available_tokens = list(set(token for tokens in pool_tokens.values() for token in tokens))
    
    # Create mock wallet balances
    warm_wallet = {}
    for token in available_tokens[:5]:  # Limit to first 5 tokens
        warm_wallet[token] = 10000.0  # $10,000 each
    
    # Create mock current allocations
    current_allocations = {}
    for token in list(warm_wallet.keys())[:3]:  # Allocate first 3 tokens
        pools_for_token = [pool_id for pool_id, tokens in pool_tokens.items() if token in tokens]
        if pools_for_token:
            pool_id = pools_for_token[0]
            current_allocations[(pool_id, token)] = 1000.0  # $1,000 each
    
    return pools_df, token_prices, warm_wallet, current_allocations, gas_fee_usd, alloc_params


def analyze_pool_characteristics(pools_df: pd.DataFrame, test_name: str) -> Dict:
    """Analyze and return pool characteristics"""
    
    # Token diversity
    all_tokens = []
    token_counts = []
    for _, pool in pools_df.iterrows():
        tokens = pool['symbol'].split('-')
        all_tokens.extend(tokens)
        token_counts.append(len(tokens))
    
    unique_tokens = list(set(all_tokens))
    
    # APY statistics
    apy_stats = {
        'min': float(pools_df['forecasted_apy'].min()),
        'max': float(pools_df['forecasted_apy'].max()),
        'mean': float(pools_df['forecasted_apy'].mean()),
        'median': float(pools_df['forecasted_apy'].median()),
        'std': float(pools_df['forecasted_apy'].std())
    }
    
    # Protocol diversity
    protocols = pools_df['protocol'].value_counts().to_dict()
    
    # Chain diversity
    chains = pools_df['chain'].value_counts().to_dict()
    
    # Symbol patterns
    symbol_lengths = [len(symbol) for symbol in pools_df['symbol']]
    multi_token_pools = sum(1 for symbol in pools_df['symbol'] if '-' in symbol)
    
    characteristics = {
        'test_name': test_name,
        'pool_count': len(pools_df),
        'token_diversity': {
            'unique_tokens': len(unique_tokens),
            'unique_token_list': unique_tokens,
            'avg_tokens_per_pool': np.mean(token_counts),
            'max_tokens_per_pool': max(token_counts),
            'min_tokens_per_pool': min(token_counts)
        },
        'apy_stats': apy_stats,
        'protocol_diversity': {
            'unique_protocols': len(protocols),
            'protocol_counts': protocols
        },
        'chain_diversity': {
            'unique_chains': len(chains),
            'chain_counts': chains
        },
        'symbol_patterns': {
            'avg_symbol_length': np.mean(symbol_lengths),
            'max_symbol_length': max(symbol_lengths),
            'multi_token_pool_ratio': multi_token_pools / len(pools_df)
        },
        'pool_ids': pools_df['pool_id'].tolist()[:5],  # First 5 pool IDs
        'sample_symbols': pools_df['symbol'].tolist()[:5]  # First 5 symbols
    }
    
    return characteristics


def test_solver_success(pools_df, token_prices, warm_wallet, current_allocations, gas_fee_usd, alloc_params, test_name):
    """Test if the solver succeeds with the given data"""
    
    try:
        # Initialize optimizer
        optimizer = AllocationOptimizer(
            pools_df=pools_df,
            token_prices=token_prices,
            warm_wallet=warm_wallet,
            current_allocations=current_allocations,
            gas_fee_usd=gas_fee_usd,
            alloc_params=alloc_params
        )
        
        # Try to solve with SCIPY
        import cvxpy as cp
        
        logger.info(f"Testing {test_name} with SCIPY...")
        start_time = time.time()
        success = optimizer.solve(solver=cp.SCIPY, verbose=False)
        solve_time = time.time() - start_time
        
        if success:
            # Extract results
            allocations_df, transactions = optimizer.extract_results()
            formatted_results = optimizer.format_results()
            
            return {
                'success': True,
                'solver': 'SCIPY',
                'solve_time': solve_time,
                'num_allocations': len(allocations_df),
                'num_transactions': len(transactions),
                'num_pools_used': len(formatted_results['final_allocations']),
                'error': None
            }
        else:
            return {
                'success': False,
                'solver': 'SCIPY',
                'solve_time': solve_time,
                'error': 'Solver returned False',
                'num_allocations': 0,
                'num_transactions': 0,
                'num_pools_used': 0
            }
    except Exception as e:
        return {
            'success': False,
            'solver': 'SCIPY',
            'solve_time': 0,
            'error': str(e),
            'num_allocations': 0,
            'num_transactions': 0,
            'num_pools_used': 0
        }


def main():
    """Main diagnostic function"""
    logger.info("\n" + "="*80)
    logger.info("DIAGNOSTIC COMPARISON TEST")
    logger.info("="*80)
    
    # Test successful data
    logger.info("\n" + "="*50)
    logger.info("TESTING SUCCESSFUL DATA")
    logger.info("="*50)
    
    successful_pools, successful_tokens, successful_wallet, successful_allocations, successful_gas, successful_params = create_successful_test_data()
    successful_characteristics = analyze_pool_characteristics(successful_pools, "Successful Test")
    successful_result = test_solver_success(
        successful_pools, successful_tokens, successful_wallet, 
        successful_allocations, successful_gas, successful_params, "Successful Test"
    )
    
    # Test real data
    logger.info("\n" + "="*50)
    logger.info("TESTING REAL DATA")
    logger.info("="*50)
    
    real_pools, real_tokens, real_wallet, real_allocations, real_gas, real_params = create_real_pool_data()
    real_characteristics = analyze_pool_characteristics(real_pools, "Real Data Test")
    real_result = test_solver_success(
        real_pools, real_tokens, real_wallet, 
        real_allocations, real_gas, real_params, "Real Data Test"
    )
    
    # Compare characteristics
    logger.info("\n" + "="*50)
    logger.info("COMPARISON RESULTS")
    logger.info("="*50)
    
    logger.info(f"\nSolver Results:")
    logger.info(f"  Successful Test: {'✓' if successful_result['success'] else '✗'}")
    if successful_result['success']:
        logger.info(f"    - Solve time: {successful_result['solve_time']:.3f}s")
        logger.info(f"    - Allocations: {successful_result['num_allocations']}")
        logger.info(f"    - Transactions: {successful_result['num_transactions']}")
    else:
        logger.info(f"    - Error: {successful_result['error']}")
    
    logger.info(f"  Real Data Test: {'✓' if real_result['success'] else '✗'}")
    if real_result['success']:
        logger.info(f"    - Solve time: {real_result['solve_time']:.3f}s")
        logger.info(f"    - Allocations: {real_result['num_allocations']}")
        logger.info(f"    - Transactions: {real_result['num_transactions']}")
    else:
        logger.info(f"    - Error: {real_result['error']}")
    
    logger.info(f"\nPool Characteristics Comparison:")
    logger.info(f"  Pool Count:")
    logger.info(f"    - Successful: {successful_characteristics['pool_count']}")
    logger.info(f"    - Real Data: {real_characteristics['pool_count']}")
    
    logger.info(f"  Token Diversity:")
    logger.info(f"    - Successful: {successful_characteristics['token_diversity']['unique_tokens']} unique tokens")
    logger.info(f"    - Real Data: {real_characteristics['token_diversity']['unique_tokens']} unique tokens")
    logger.info(f"    - Successful tokens: {successful_characteristics['token_diversity']['unique_token_list']}")
    logger.info(f"    - Real Data tokens: {real_characteristics['token_diversity']['unique_token_list']}")
    
    logger.info(f"  APY Stats:")
    logger.info(f"    - Successful:")
    logger.info(f"      - Range: {successful_characteristics['apy_stats']['min']:.2f}% - {successful_characteristics['apy_stats']['max']:.2f}%")
    logger.info(f"      - Mean: {successful_characteristics['apy_stats']['mean']:.2f}%")
    logger.info(f"    - Real Data:")
    logger.info(f"      - Range: {real_characteristics['apy_stats']['min']:.2f}% - {real_characteristics['apy_stats']['max']:.2f}%")
    logger.info(f"      - Mean: {real_characteristics['apy_stats']['mean']:.2f}%")
    
    logger.info(f"  Protocol Diversity:")
    logger.info(f"    - Successful: {successful_characteristics['protocol_diversity']['unique_protocols']} protocols")
    logger.info(f"    - Real Data: {real_characteristics['protocol_diversity']['unique_protocols']} protocols")
    
    logger.info(f"  Symbol Patterns:")
    logger.info(f"    - Successful:")
    logger.info(f"      - Avg length: {successful_characteristics['symbol_patterns']['avg_symbol_length']:.1f}")
    logger.info(f"      - Multi-token ratio: {successful_characteristics['symbol_patterns']['multi_token_pool_ratio']:.2%}")
    logger.info(f"      - Sample: {successful_characteristics['sample_symbols']}")
    logger.info(f"    - Real Data:")
    logger.info(f"      - Avg length: {real_characteristics['symbol_patterns']['avg_symbol_length']:.1f}")
    logger.info(f"      - Multi-token ratio: {real_characteristics['symbol_patterns']['multi_token_pool_ratio']:.2%}")
    logger.info(f"      - Sample: {real_characteristics['sample_symbols']}")
    
    # Save comparison results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"diagnostic_comparison_results_{timestamp}.json"
    
    comparison_data = {
        'timestamp': datetime.now().isoformat(),
        'successful_characteristics': successful_characteristics,
        'successful_result': successful_result,
        'real_characteristics': real_characteristics,
        'real_result': real_result
    }
    
    with open(filename, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    logger.info(f"\nComparison results saved to: {filename}")
    
    # Key differences analysis
    logger.info(f"\nKey Differences Analysis:")
    
    # Check for extreme APY values in real data
    if real_characteristics['apy_stats']['max'] > 100.0:  # > 100%
        logger.warning(f"  ⚠ Real data has extremely high APYs (max: {real_characteristics['apy_stats']['max']:.2f}%)")
        logger.warning(f"    This may cause numerical instability in the solver")
    
    # Check for unusual tokens
    real_tokens = set(real_characteristics['token_diversity']['unique_token_list'])
    successful_tokens = set(successful_characteristics['token_diversity']['unique_token_list'])
    unusual_tokens = real_tokens - successful_tokens
    
    if unusual_tokens:
        logger.warning(f"  ⚠ Real data has unusual tokens: {list(unusual_tokens)[:5]}...")
        logger.warning(f"    These may have special characteristics that affect the model")
    
    # Check for complex symbols
    if real_characteristics['symbol_patterns']['max_symbol_length'] > 50:
        logger.warning(f"  ⚠ Real data has very long symbols (max: {real_characteristics['symbol_patterns']['max_symbol_length']})")
        logger.warning(f"    This may indicate complex multi-token pools")
    
    return 0


if __name__ == "__main__":
    exit(main())