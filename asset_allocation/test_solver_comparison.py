#!/usr/bin/env python3
"""
Solver Comparison Test Script

This script compares the performance and solution quality of different solvers
on the same optimization problem.
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
    calculate_aum
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_data():
    """Create test data for solver comparison"""
    
    # Use the same comprehensive dataset as our successful test
    pools_df = pd.DataFrame([
        # Low APY pools (current suboptimal allocations)
        {'pool_id': 'pool-low-1', 'symbol': 'USDC', 'chain': 'ethereum', 'protocol': 'aave', 'forecasted_apy': 0.012},
        {'pool_id': 'pool-low-2', 'symbol': 'USDT', 'chain': 'ethereum', 'protocol': 'compound', 'forecasted_apy': 0.015},
        {'pool_id': 'pool-low-3', 'symbol': 'DAI', 'chain': 'ethereum', 'protocol': 'aave', 'forecasted_apy': 0.018},
        
        # Medium APY pools
        {'pool_id': 'pool-med-1', 'symbol': 'USDC', 'chain': 'ethereum', 'protocol': 'curve', 'forecasted_apy': 0.035},
        {'pool_id': 'pool-med-2', 'symbol': 'USDT', 'chain': 'ethereum', 'protocol': 'curve', 'forecasted_apy': 0.038},
        {'pool_id': 'pool-med-3', 'symbol': 'DAI', 'chain': 'ethereum', 'protocol': 'compound', 'forecasted_apy': 0.042},
        
        # High APY single-token pools
        {'pool_id': 'pool-high-1', 'symbol': 'USDC', 'chain': 'ethereum', 'protocol': 'balancer', 'forecasted_apy': 0.075},
        {'pool_id': 'pool-high-2', 'symbol': 'USDT', 'chain': 'ethereum', 'protocol': 'yearn', 'forecasted_apy': 0.082},
        {'pool_id': 'pool-high-3', 'symbol': 'DAI', 'chain': 'ethereum', 'protocol': 'yearn', 'forecasted_apy': 0.078},
        
        # Very high APY multi-token pools
        {'pool_id': 'pool-vhigh-1', 'symbol': 'USDC-USDT', 'chain': 'ethereum', 'protocol': 'curve', 'forecasted_apy': 0.095},
        {'pool_id': 'pool-vhigh-2', 'symbol': 'DAI-USDC', 'chain': 'ethereum', 'protocol': 'balancer', 'forecasted_apy': 0.092},
        {'pool_id': 'pool-vhigh-3', 'symbol': 'USDT-DAI', 'chain': 'ethereum', 'protocol': 'curve', 'forecasted_apy': 0.098},
        
        # Premium pools
        {'pool_id': 'pool-prem-1', 'symbol': 'USDC-USDT-DAI', 'chain': 'ethereum', 'protocol': 'curve', 'forecasted_apy': 0.105},
        {'pool_id': 'pool-prem-2', 'symbol': 'USDC-DAI', 'chain': 'ethereum', 'protocol': 'balancer', 'forecasted_apy': 0.102},
        {'pool_id': 'pool-prem-3', 'symbol': 'USDT-USDC', 'chain': 'ethereum', 'protocol': 'yearn', 'forecasted_apy': 0.108}
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
    
    # Low transaction costs to ensure reallocation
    gas_fee_usd = 0.1
    alloc_params = {
        'max_alloc_percentage': 0.25,
        'conversion_rate': 0.00005,
        'min_transaction_value': 25.0
    }
    
    return pools_df, token_prices, warm_wallet, current_allocations, gas_fee_usd, alloc_params


def test_solver(pools_df, token_prices, warm_wallet, current_allocations, gas_fee_usd, alloc_params, solver_name):
    """Test optimization with a specific solver"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TESTING SOLVER: {solver_name}")
    logger.info(f"{'='*60}")
    
    try:
        # Create fresh optimizer for each test
        optimizer = AllocationOptimizer(
            pools_df=pools_df,
            token_prices=token_prices,
            warm_wallet=warm_wallet,
            current_allocations=current_allocations,
            gas_fee_usd=gas_fee_usd,
            alloc_params=alloc_params
        )
        
        # Get the solver object
        import cvxpy as cp
        solver = getattr(cp, solver_name)
        
        # Time the solving process
        start_time = time.time()
        success = optimizer.solve(solver=solver, verbose=False)
        solve_time = time.time() - start_time
        
        if not success:
            return {
                'solver': solver_name,
                'success': False,
                'error': 'Solver failed to find solution',
                'solve_time': solve_time
            }
        
        # Extract results
        allocations_df, transactions = optimizer.extract_results()
        formatted_results = optimizer.format_results()
        
        # Calculate metrics
        optimized_yield = 0
        for pool_id, pool_data in formatted_results['final_allocations'].items():
            pool_apy = pools_df[pools_df['pool_id'] == pool_id]['forecasted_apy'].iloc[0]
            pool_total = sum(token['amount_usd'] for token in pool_data['tokens'].values())
            optimized_yield += pool_total * pool_apy / 100 / 365
        
        # Calculate current yield for comparison
        current_yield = sum(
            amount * pools_df[pools_df['pool_id'] == pool_id]['forecasted_apy'].iloc[0] / 100 / 365
            for (pool_id, token), amount in current_allocations.items()
        )
        
        improvement = optimized_yield - current_yield
        improvement_pct = (improvement / current_yield * 100) if current_yield > 0 else 0
        
        # Calculate total transaction costs
        total_costs = sum(txn.get('total_cost_usd', 0) for txn in transactions)
        
        result = {
            'solver': solver_name,
            'success': True,
            'solve_time': solve_time,
            'objective_value': None,  # Not directly accessible from optimizer
            'current_yield': current_yield,
            'optimized_yield': optimized_yield,
            'daily_improvement': improvement,
            'improvement_pct': improvement_pct,
            'annual_improvement': improvement * 365,
            'num_allocations': len(allocations_df),
            'num_transactions': len(transactions),
            'num_pools_used': len(formatted_results['final_allocations']),
            'total_costs': total_costs,
            'status': 'optimal'  # We know it's optimal if solve() returned True
        }
        
        # Log results
        logger.info(f"✓ {solver_name} Results:")
        logger.info(f"  - Solve time: {solve_time:.3f} seconds")
        logger.info(f"  - Status: optimal")
        logger.info(f"  - Daily yield improvement: ${improvement:.2f} ({improvement_pct:+.1f}%)")
        logger.info(f"  - Annual improvement: ${improvement * 365:,.2f}")
        logger.info(f"  - Allocations: {len(allocations_df)} positions")
        logger.info(f"  - Transactions: {len(transactions)} total")
        logger.info(f"  - Transaction costs: ${total_costs:.2f}")
        
        return result
        
    except Exception as e:
        logger.error(f"✗ {solver_name} failed: {e}")
        return {
            'solver': solver_name,
            'success': False,
            'error': str(e),
            'solve_time': 0
        }


def compare_solvers(results):
    """Compare and analyze solver performance"""
    
    logger.info(f"\n{'='*80}")
    logger.info("SOLVER COMPARISON SUMMARY")
    logger.info(f"{'='*80}")
    
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    if not successful_results:
        logger.error("All solvers failed!")
        return
    
    # Create comparison table
    logger.info(f"\nPerformance Comparison:")
    logger.info(f"{'Solver':<10} {'Time(s)':<8} {'Yield Imp':<12} {'Allocations':<12} {'Transactions':<12} {'Costs':<8}")
    logger.info(f"{'-'*80}")
    
    best_yield = max(r['daily_improvement'] for r in successful_results)
    fastest_time = min(r['solve_time'] for r in successful_results)
    
    for result in successful_results:
        solver = result['solver']
        time_str = f"{result['solve_time']:.3f}"
        yield_str = f"${result['daily_improvement']:.2f} ({result['improvement_pct']:+.1f}%)"
        alloc_str = f"{result['num_allocations']} positions"
        txn_str = f"{result['num_transactions']} total"
        cost_str = f"${result['total_costs']:.2f}"
        
        # Highlight best performers
        if result['daily_improvement'] == best_yield:
            solver += " *BEST YIELD*"
        if result['solve_time'] == fastest_time:
            time_str += " *FASTEST*"
        
        logger.info(f"{solver:<10} {time_str:<8} {yield_str:<12} {alloc_str:<12} {txn_str:<12} {cost_str:<8}")
    
    # Detailed analysis
    logger.info(f"\nDetailed Analysis:")
    
    # Yield improvement comparison
    logger.info(f"\nYield Improvement Ranking:")
    sorted_by_yield = sorted(successful_results, key=lambda x: x['daily_improvement'], reverse=True)
    for i, result in enumerate(sorted_by_yield, 1):
        logger.info(f"  {i}. {result['solver']}: ${result['daily_improvement']:.2f}/day "
                   f"({result['improvement_pct']:+.1f}%)")
    
    # Solve time comparison
    logger.info(f"\nSolve Time Ranking:")
    sorted_by_time = sorted(successful_results, key=lambda x: x['solve_time'])
    for i, result in enumerate(sorted_by_time, 1):
        logger.info(f"  {i}. {result['solver']}: {result['solve_time']:.3f} seconds")
    
    # Solution consistency check
    logger.info(f"\nSolution Consistency:")
    if len(set(r['num_allocations'] for r in successful_results)) == 1:
        logger.info(f"  ✓ All solvers found same number of allocations: {successful_results[0]['num_allocations']}")
    else:
        logger.info(f"  ⚠ Solvers found different numbers of allocations:")
        for result in successful_results:
            logger.info(f"    - {result['solver']}: {result['num_allocations']} allocations")
    
    if len(set(r['num_transactions'] for r in successful_results)) == 1:
        logger.info(f"  ✓ All solvers found same number of transactions: {successful_results[0]['num_transactions']}")
    else:
        logger.info(f"  ⚠ Solvers found different numbers of transactions:")
        for result in successful_results:
            logger.info(f"    - {result['solver']}: {result['num_transactions']} transactions")
    
    # Failed solvers
    if failed_results:
        logger.info(f"\nFailed Solvers:")
        for result in failed_results:
            logger.info(f"  - {result['solver']}: {result['error']}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"solver_comparison_results_{timestamp}.json"
    
    comparison_data = {
        'timestamp': datetime.now().isoformat(),
        'successful_results': successful_results,
        'failed_results': failed_results,
        'best_yield_solver': max(successful_results, key=lambda x: x['daily_improvement'])['solver'],
        'fastest_solver': min(successful_results, key=lambda x: x['solve_time'])['solver']
    }
    
    with open(filename, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    logger.info(f"\nResults saved to: {filename}")


def main():
    """Main solver comparison function"""
    logger.info("\n" + "="*80)
    logger.info("SOLVER COMPARISON TEST")
    logger.info("="*80)
    
    # Create test data
    pools_df, token_prices, warm_wallet, current_allocations, gas_fee_usd, alloc_params = create_test_data()
    
    total_aum = calculate_aum(warm_wallet, current_allocations, token_prices)
    
    logger.info(f"Test Problem:")
    logger.info(f"  - Pools: {len(pools_df)}")
    logger.info(f"  - Tokens: {len(token_prices)}")
    logger.info(f"  - Total AUM: ${total_aum:,.2f}")
    logger.info(f"  - Current allocations: ${sum(current_allocations.values()):,.2f}")
    logger.info(f"  - Gas fee: ${gas_fee_usd}")
    
    # List of solvers to test
    solvers_to_test = ['SCIPY', 'ECOS', 'CBC', 'HIGHS']
    
    # Test each solver
    results = []
    for solver_name in solvers_to_test:
        result = test_solver(
            pools_df, token_prices, warm_wallet, current_allocations, 
            gas_fee_usd, alloc_params, solver_name
        )
        results.append(result)
    
    # Compare results
    compare_solvers(results)
    
    # Return success if at least one solver worked
    successful = any(r['success'] for r in results)
    if successful:
        logger.info("\n✓ SOLVER COMPARISON TEST COMPLETED")
        return 0
    else:
        logger.error("\n✗ ALL SOLVERS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())