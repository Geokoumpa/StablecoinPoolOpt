#!/usr/bin/env python3
"""
Test Script with Real Pools and GUROBI Solver

This script uses the actual filtered pools from the database with real forecasted APYs
and gas fee forecasts, but uses mock wallet balances and the GUROBI solver.
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


def create_mock_wallet_balances(token_prices: Dict[str, float], available_tokens: List[str]) -> Dict[str, float]:
    """Create mock wallet balances with significant amounts for testing"""
    
    # Create mock balances based on available tokens in pools
    # Using substantial amounts to ensure meaningful allocations
    mock_balances = {}
    
    # Define default prices for tokens if not available
    default_prices = {
        'USDC': 1.0, 'USDT': 1.0, 'DAI': 1.0, 'ETH': 3200.0, 'WBTC': 45000.0,
        'BOLD': 1.0, 'GHO': 1.0, 'USDE': 1.0, 'CRVUSD': 1.0, 'LUSD': 1.0
    }
    
    # Use actual prices if available, otherwise defaults
    prices = {token: token_prices.get(token, default_prices.get(token, 1.0)) for token in available_tokens}
    
    # Major stablecoins with large amounts
    for token in ['USDC', 'USDT', 'DAI']:
        if token in prices:
            mock_balances[token] = 50000.0  # $50,000 each
    
    # ETH and WBTC with smaller amounts
    if 'ETH' in prices:
        mock_balances['ETH'] = 10.0  # ~$32,000
    
    if 'WBTC' in prices:
        mock_balances['WBTC'] = 0.5  # ~$22,500
    
    # Other tokens with moderate amounts
    for token in available_tokens:
        if token not in mock_balances:
            mock_balances[token] = 10000.0  # $10,000 each
    
    total_value = sum(balance * prices[token] for token, balance in mock_balances.items())
    logger.info(f"Created mock wallet with ${total_value:,.2f}")
    return mock_balances


def create_mock_current_allocations(token_prices: Dict[str, float], pool_tokens: Dict) -> Dict:
    """Create mock current allocations in lower-APY pools"""
    
    # Create some mock allocations in lower-APY pools to demonstrate reallocation
    mock_allocations = {}
    
    # Pick a few tokens to allocate
    tokens_to_allocate = ['USDC', 'USDT', 'DAI']
    
    for token in tokens_to_allocate:
        if token in token_prices:
            # Find pools that accept this token
            pools_for_token = [pool_id for pool_id, tokens in pool_tokens.items() if token in tokens]
            
            if pools_for_token:
                # Allocate to the first pool (likely has lower APY)
                pool_id = pools_for_token[0]
                amount = 5000.0  # $5,000 in each
                mock_allocations[(pool_id, token)] = amount
    
    total_allocated = sum(mock_allocations.values())
    logger.info(f"Created mock current allocations totaling ${total_allocated:,.2f}")
    
    return mock_allocations


def test_real_pools_with_gurobi():
    """Test optimization with real pools and GUROBI solver"""
    logger.info("\n" + "="*80)
    logger.info("TESTING WITH REAL POOLS AND GUROBI SOLVER")
    logger.info("="*80)
    
    try:
        # Check GUROBI availability
        try:
            import gurobipy as gp
            logger.info(f"✓ GUROBI available: version {gp.gurobi.version()}")
        except ImportError:
            logger.error("✗ GUROBI not available")
            return False
        
        # Connect to database
        engine = get_db_connection()
        logger.info("✓ Database connection established")
        
        # Fetch real data
        logger.info("Fetching real pool data...")
        pools_df = fetch_pool_data(engine)
        logger.info(f"✓ Loaded {len(pools_df)} approved pools")
        
        # If no pools for today, use fallback
        if len(pools_df) == 0:
            logger.warning("Using fallback query to get pools from yesterday...")
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
            LIMIT 175
            """
            pools_df = pd.read_sql(fallback_query, engine)
            logger.info(f"✓ Loaded {len(pools_df)} pools using fallback query")
        
        logger.info("Fetching token prices...")
        # Define common tokens to fetch prices for
        common_tokens = ['USDC', 'USDT', 'DAI', 'ETH', 'WBTC', 'WETH', 'BOLD', 'GHO', 'USDE', 'CRVUSD']
        token_prices = fetch_token_prices(engine, common_tokens)
        logger.info(f"✓ Loaded {len(token_prices)} token prices")
        
        logger.info("Fetching gas fee data...")
        gas_gwei, eth_price = fetch_gas_fee_data(engine)
        gas_fee_usd = gas_gwei * 1e-9 * eth_price
        logger.info(f"✓ Loaded gas fee: ${gas_fee_usd}")
        
        logger.info("Fetching allocation parameters...")
        alloc_params = fetch_allocation_parameters(engine)
        logger.info(f"✓ Loaded allocation parameters: max_alloc={alloc_params['max_alloc_percentage']:.2f%}")
        
        # Create mock data
        logger.info("\nCreating mock wallet balances...")
        pool_tokens = {}
        for _, pool in pools_df.iterrows():
            pool_tokens[pool['pool_id']] = pool['symbol'].split('-')
        
        # Get all unique tokens from pools
        available_tokens = list(set(token for tokens in pool_tokens.values() for token in tokens))
        logger.info(f"Available tokens from pools: {len(available_tokens)}")
        
        warm_wallet = create_mock_wallet_balances(token_prices, available_tokens)
        current_allocations = create_mock_current_allocations(token_prices, pool_tokens)
        
        # Calculate AUM
        total_aum = calculate_aum(warm_wallet, current_allocations, token_prices)
        
        # Log summary
        logger.info(f"\nTest Setup Summary:")
        logger.info(f"  - Real pools from database: {len(pools_df)}")
        logger.info(f"  - APY range: {pools_df['forecasted_apy'].min():.2f%} - {pools_df['forecasted_apy'].max():.2f%}")
        logger.info(f"  - Tokens available: {len(available_tokens)}")
        logger.info(f"  - Total AUM: ${total_aum:,.2f}")
        logger.info(f"  - Currently allocated: ${sum(current_allocations.values()):,.2f}")
        logger.info(f"  - Warm wallet available: ${sum(warm_wallet.values()):,.2f}")
        logger.info(f"  - Real gas fee: ${gas_fee_usd}")
        
        # Show top pools by APY
        top_pools = pools_df.nlargest(5, 'forecasted_apy')
        logger.info(f"\nTop 5 Pools by APY:")
        for _, pool in top_pools.iterrows():
            logger.info(f"  - {pool['pool_id']}: {pool['forecasted_apy']:.2f%} ({pool['protocol']})")
        
        # Initialize optimizer
        logger.info("\nInitializing optimizer with real pools and mock assets...")
        optimizer = AllocationOptimizer(
            pools_df=pools_df,
            token_prices=token_prices,
            warm_wallet=warm_wallet,
            current_allocations=current_allocations,
            gas_fee_usd=gas_fee_usd,
            alloc_params=alloc_params
        )
        
        # Solve optimization with GUROBI
        logger.info(f"\nBuilding optimization model...")
        logger.info(f"  - Pools: {optimizer.n_pools}")
        logger.info(f"  - Tokens: {optimizer.n_tokens}")
        logger.info(f"  - Variables: {optimizer.n_pools * optimizer.n_tokens}")
        
        import cvxpy as cp
        
        # Try GUROBI solver
        try:
            logger.info(f"\nAttempting to solve with GUROBI...")
            start_time = time.time()
            success = optimizer.solve(solver=cp.GUROBI, verbose=True)
            solve_time = time.time() - start_time
            
            if success:
                logger.info(f"✓ Solved with GUROBI in {solve_time:.2f} seconds")
            else:
                logger.error(f"✗ GUROBI solver failed")
                return False
        except Exception as e:
            logger.error(f"✗ GUROBI solver failed with error: {e}")
            return False
        
        # Extract and analyze results
        allocations_df, transactions = optimizer.extract_results()
        formatted_results = optimizer.format_results()
        
        # Calculate current yield
        current_yield = 0
        for (pool_id, token), amount in current_allocations.items():
            pool_apy = pools_df[pools_df['pool_id'] == pool_id]['forecasted_apy'].iloc[0]
            current_yield += amount * pool_apy / 100 / 365
        
        # Calculate optimized yield
        optimized_yield = 0
        for pool_id, pool_data in formatted_results['final_allocations'].items():
            pool_apy = pools_df[pools_df['pool_id'] == pool_id]['forecasted_apy'].iloc[0]
            pool_total = sum(token['amount_usd'] for token in pool_data['tokens'].values())
            optimized_yield += pool_total * pool_apy / 100 / 365
        
        # Calculate improvement
        improvement = optimized_yield - current_yield
        improvement_pct = (improvement / current_yield * 100) if current_yield > 0 else 0
        
        # Log results
        logger.info(f"\nOptimization Results:")
        logger.info(f"  - Final allocations: {len(allocations_df)} positions")
        logger.info(f"  - Transactions: {len(transactions)} total")
        logger.info(f"  - Final pools used: {len(formatted_results['final_allocations'])}")
        logger.info(f"  - Unallocated tokens: {len(formatted_results['unallocated_tokens'])}")
        
        logger.info(f"\nYield Analysis:")
        logger.info(f"  - Current daily yield: ${current_yield:.2f}")
        logger.info(f"  - Optimized daily yield: ${optimized_yield:.2f}")
        logger.info(f"  - Daily improvement: ${improvement:.2f} ({improvement_pct:+.1f}%)")
        logger.info(f"  - Annualized improvement: ${improvement * 365:,.2f}")
        
        # Show transaction summary
        if transactions:
            txn_types = {}
            total_costs = 0
            for txn in transactions:
                txn_type = txn['type']
                txn_types[txn_type] = txn_types.get(txn_type, 0) + 1
                total_costs += txn.get('total_cost_usd', 0)
            
            logger.info(f"\nTransaction Summary:")
            for txn_type, count in txn_types.items():
                logger.info(f"  - {txn_type}: {count} transactions")
            logger.info(f"  - Total transaction costs: ${total_costs:.2f}")
        
        # Show top allocations
        if formatted_results['final_allocations']:
            logger.info(f"\nTop 5 Allocations by Amount:")
            allocations = []
            for pool_id, pool_data in formatted_results['final_allocations'].items():
                pool_apy = pools_df[pools_df['pool_id'] == pool_id]['forecasted_apy'].iloc[0]
                pool_total = sum(token['amount_usd'] for token in pool_data['tokens'].values())
                allocations.append((pool_id, pool_apy, pool_total))
            
            allocations.sort(key=lambda x: x[2], reverse=True)
            for pool_id, pool_apy, pool_total in allocations[:5]:
                logger.info(f"  {pool_id} ({pool_apy:.2f%}): ${pool_total:,.2f}")
        
        # Validate results
        if improvement > 0:
            logger.info(f"\n✓ SUCCESS: Optimization improved yield by {improvement_pct:+.1f}%")
            logger.info("The optimization successfully reallocated from lower-APY to higher-APY pools")
        elif abs(improvement) < 0.01:
            logger.info(f"\n✓ NEUTRAL: Optimization maintained yield (change: {improvement_pct:+.1f}%)")
            logger.info("Current allocations appear to be already optimal")
        else:
            logger.warning(f"\n⚠ DECLINE: Optimization reduced yield by {improvement_pct:+.1f}%")
            logger.info("Transaction costs may outweigh APY benefits")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"real_pools_gurobi_results_{timestamp}.json"
        
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'test_type': 'real_pools_with_gurobi',
            'pools_count': len(pools_df),
            'apy_range': [float(pools_df['forecasted_apy'].min()), float(pools_df['forecasted_apy'].max())],
            'total_aum': float(total_aum),
            'gas_fee_usd': float(gas_fee_usd),
            'current_yield': float(current_yield),
            'optimized_yield': float(optimized_yield),
            'daily_improvement': float(improvement),
            'improvement_pct': float(improvement_pct),
            'num_allocations': len(allocations_df),
            'num_transactions': len(transactions),
            'solver': 'GUROBI',
            'solve_time': solve_time
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"\nResults saved to: {filename}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed with error: {e}", exc_info=True)
        return False


def main():
    """Main test function"""
    try:
        success = test_real_pools_with_gurobi()
        if success:
            logger.info("\n✓ REAL POOLS WITH GUROBI TEST COMPLETED")
            logger.info("Successfully tested optimization with:")
            logger.info("  - Real filtered pools from database")
            logger.info("  - Real forecasted APYs")
            logger.info("  - Real gas fee forecasts")
            logger.info("  - Mock wallet balances for meaningful allocations")
            logger.info("  - GUROBI commercial solver")
            return 0
        else:
            logger.error("\n✗ REAL POOLS WITH GUROBI TEST FAILED")
            return 1
    except Exception as e:
        logger.error(f"\n✗ REAL POOLS WITH GUROBI TEST FAILED WITH ERROR: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())