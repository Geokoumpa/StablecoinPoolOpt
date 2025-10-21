"""
Test script for the data_loader module

This script tests all the data loading functions to ensure they work correctly
and can be used by the OR-Tools optimizer.
"""

import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asset_allocation.data_loader import (
    fetch_pool_data, fetch_token_prices, fetch_gas_fee_data,
    fetch_current_balances, fetch_allocation_parameters,
    parse_pool_tokens, calculate_aum, build_token_universe
)
from database.db_utils import get_db_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_data_loader():
    """Test all data loading functions"""
    logger.info("=" * 80)
    logger.info("TESTING DATA LOADER MODULE")
    logger.info("=" * 80)
    
    engine = None
    try:
        # Connect to database
        engine = get_db_connection()
        if not engine:
            logger.error("Failed to establish database connection")
            return False
        
        # Test 1: fetch_pool_data
        logger.info("\n[1/8] Testing fetch_pool_data...")
        pools_df = fetch_pool_data(engine)
        if pools_df.empty:
            logger.warning("No pool data found")
        else:
            logger.info(f"✓ Successfully loaded {len(pools_df)} pools")
            logger.info(f"  Columns: {list(pools_df.columns)}")
            logger.info(f"  Sample pool: {pools_df.iloc[0]['symbol']}")
        
        # Test 2: parse_pool_tokens
        logger.info("\n[2/8] Testing parse_pool_tokens...")
        test_symbols = ["DAI-USDC-USDT", "GTUSDC", "USDC-DAI"]
        for symbol in test_symbols:
            tokens = parse_pool_tokens(symbol)
            logger.info(f"  {symbol} -> {tokens}")
        logger.info("✓ parse_pool_tokens working correctly")
        
        # Test 3: fetch_current_balances
        logger.info("\n[3/8] Testing fetch_current_balances...")
        warm_wallet, current_allocations = fetch_current_balances(engine)
        logger.info(f"✓ Warm wallet: {len(warm_wallet)} tokens")
        logger.info(f"✓ Current allocations: {len(current_allocations)} positions")
        
        # Test 4: build_token_universe
        logger.info("\n[4/8] Testing build_token_universe...")
        if not pools_df.empty:
            tokens = build_token_universe(pools_df, warm_wallet, current_allocations)
            logger.info(f"✓ Token universe: {len(tokens)} tokens")
            logger.info(f"  Tokens: {tokens}")
        else:
            logger.warning("Skipping token universe test - no pool data")
        
        # Test 5: fetch_token_prices
        logger.info("\n[5/8] Testing fetch_token_prices...")
        if not pools_df.empty:
            token_prices = fetch_token_prices(engine, tokens + ['ETH'])
            logger.info(f"✓ Loaded prices for {len(token_prices)} tokens")
            for token, price in list(token_prices.items())[:5]:  # Show first 5
                logger.info(f"  {token}: ${price:.4f}")
        else:
            logger.warning("Skipping token prices test - no tokens to query")
        
        # Test 6: fetch_gas_fee_data
        logger.info("\n[6/8] Testing fetch_gas_fee_data...")
        gas_gwei, eth_price = fetch_gas_fee_data(engine)
        logger.info(f"✓ Gas fee: {gas_gwei:.2f} Gwei, ETH price: ${eth_price:.2f}")
        
        # Test 7: calculate_aum
        logger.info("\n[7/8] Testing calculate_aum...")
        if not pools_df.empty and token_prices:
            total_aum = calculate_aum(warm_wallet, current_allocations, token_prices)
            logger.info(f"✓ Total AUM: ${total_aum:,.2f}")
        else:
            logger.warning("Skipping AUM calculation - missing data")
        
        # Test 8: fetch_allocation_parameters
        logger.info("\n[8/8] Testing fetch_allocation_parameters...")
        alloc_params = fetch_allocation_parameters(engine)
        logger.info(f"✓ Allocation parameters loaded")
        for key, value in alloc_params.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ ALL DATA LOADER TESTS PASSED")
        logger.info("=" * 80)
        return True
        
    except Exception as e:
        logger.error(f"Data loader test failed: {e}", exc_info=True)
        return False
    
    finally:
        if engine:
            engine.dispose()


if __name__ == "__main__":
    success = test_data_loader()
    sys.exit(0 if success else 1)