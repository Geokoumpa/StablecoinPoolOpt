"""
Shared Data Loading Module for Asset Allocation Optimization

This module provides shared data loading functions that can be used by both
the CVXPY/GUROBI optimizer and the new OR-Tools hybrid optimizer.

Functions:
- fetch_pool_data() - Load approved pools with forecasted APY and TVL
- fetch_token_prices() - Load latest token prices from CoinMarketCap
- fetch_gas_fee_data() - Load forecasted gas fees and ETH price
- fetch_current_balances() - Load warm wallet balances and current allocations
- fetch_allocation_parameters() - Load allocation configuration parameters
- parse_pool_tokens() - Extract tokens from pool symbols
- calculate_aum() - Calculate total Assets Under Management
- build_token_universe() - Build complete token set for optimization
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from database.db_utils import get_db_connection

# Configure module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def fetch_pool_data(engine) -> pd.DataFrame:
    """
    Fetches approved pools with forecasted APY and metadata.
    
    Args:
        engine: Database connection engine
        
    Returns:
        DataFrame with columns: pool_id, symbol, chain, protocol, forecasted_apy, forecasted_tvl
    """
    query = """
    SELECT
        pdm.pool_id,
        p.symbol,
        p.chain,
        p.protocol,
        pdm.forecasted_apy,
        pdm.forecasted_tvl
    FROM pool_daily_metrics pdm
    JOIN pools p ON pdm.pool_id = p.pool_id
    WHERE pdm.date = CURRENT_DATE 
      AND pdm.is_filtered_out = FALSE
      AND pdm.forecasted_apy IS NOT NULL
      AND pdm.forecasted_apy > 0
      AND pdm.forecasted_tvl IS NOT NULL
      AND pdm.forecasted_tvl > 0;
    """
    try:
        df = pd.read_sql(query, engine)
        logger.info(f"Loaded {len(df)} approved pools")
        return df
    except Exception as e:
        logger.error(f"Error fetching pool data: {e}")
        return pd.DataFrame()


def fetch_token_prices(engine, tokens: List[str]) -> Dict[str, float]:
    """
    Fetches latest closing prices for given tokens.
    
    Args:
        engine: Database connection engine
        tokens: List of token symbols
        
    Returns:
        Dictionary mapping token symbol to USD price
    """
    if not tokens:
        return {}
    
    try:
        tokens_str = "','".join(tokens)
        query = f"""
        WITH ranked_ohlcv AS (
            SELECT
                raw_json_data->>'symbol' as symbol,
                (raw_json_data->>'close')::float as close_price,
                (raw_json_data->>'timestamp')::timestamp as ts,
                ROW_NUMBER() OVER(
                    PARTITION BY raw_json_data->>'symbol' 
                    ORDER BY (raw_json_data->>'timestamp')::timestamp DESC
                ) as rn
            FROM raw_coinmarketcap_ohlcv
            WHERE raw_json_data->>'symbol' IN ('{tokens_str}')
        )
        SELECT symbol, close_price
        FROM ranked_ohlcv
        WHERE rn = 1;
        """
        df = pd.read_sql(query, engine)
        prices = dict(zip(df['symbol'], df['close_price']))
        logger.info(f"Loaded prices for {len(prices)} tokens")
        return prices
    except Exception as e:
        logger.error(f"Error fetching token prices: {e}")
        return {}


def fetch_gas_fee_data(engine) -> Tuple[float, float]:
    """
    Fetches forecasted gas fee and ETH price.
    
    Args:
        engine: Database connection engine
        
    Returns:
        Tuple of (forecasted_max_gas_gwei, eth_price_usd)
    """
    try:
        # Fetch gas fee data
        gas_query = """
        SELECT forecasted_max_gas_gwei
        FROM gas_fees_daily
        WHERE date = CURRENT_DATE
        ORDER BY date DESC
        LIMIT 1;
        """
        gas_df = pd.read_sql(gas_query, engine)
        gas_gwei = gas_df['forecasted_max_gas_gwei'].iloc[0] if not gas_df.empty and pd.notna(gas_df['forecasted_max_gas_gwei'].iloc[0]) else 50.0
        
        # Fetch ETH price
        eth_price_query = """
        WITH ranked_eth AS (
            SELECT
                (raw_json_data->>'close')::float as close_price,
                ROW_NUMBER() OVER(ORDER BY (raw_json_data->>'timestamp')::timestamp DESC) as rn
            FROM raw_coinmarketcap_ohlcv
            WHERE raw_json_data->>'symbol' = 'ETH'
        )
        SELECT close_price
        FROM ranked_eth
        WHERE rn = 1;
        """
        eth_df = pd.read_sql(eth_price_query, engine)
        eth_price = eth_df['close_price'].iloc[0] if not eth_df.empty and pd.notna(eth_df['close_price'].iloc[0]) else 3000.0
        
        gas_fee_usd = gas_gwei * 1e-9 * eth_price
        logger.info(f"Gas fee: {gas_gwei:.2f} Gwei, ETH price: ${eth_price:.2f}, Gas fee USD: ${gas_fee_usd:.6f}")
        
        return gas_gwei, eth_price
    except Exception as e:
        logger.error(f"Error fetching gas fee data: {e}")
        # Return default values
        return 50.0, 3000.0


def fetch_current_balances(engine) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
    """
    Fetches current token balances from warm wallet and allocated positions.
    Only queries today's data.
    
    Args:
        engine: Database connection engine
        
    Returns:
        Tuple of (warm_wallet_balances, current_allocations)
        - warm_wallet_balances: {token_symbol: amount}
        - current_allocations: {(pool_id, token_symbol): amount}
    """
    from config import MAIN_ASSET_HOLDING_ADDRESS
    
    if not MAIN_ASSET_HOLDING_ADDRESS:
        logger.error("MAIN_ASSET_HOLDING_ADDRESS not configured")
        return {}, {}
    
    warm_wallet = {}
    allocations = {}
    
    # Query today's data only
    # Include records with NULL wallet_address as they may contain the allocated assets
    query = """
    SELECT
        token_symbol,
        unallocated_balance,
        allocated_balance,
        pool_id
    FROM daily_balances
    WHERE date = CURRENT_DATE AND (wallet_address = %s OR wallet_address IS NULL);
    """
    
    try:
        df = pd.read_sql(query, engine, params=(MAIN_ASSET_HOLDING_ADDRESS,))
        
        if df.empty:
            logger.info(f"No balance data found for wallet {MAIN_ASSET_HOLDING_ADDRESS} today")
            return {}, {}
            
        logger.info(f"Using today's data for wallet {MAIN_ASSET_HOLDING_ADDRESS}")
        
    except Exception as e:
        logger.error(f"Error fetching balance data: {e}")
        return {}, {}
    
    # Process the data
    for _, row in df.iterrows():
        token = row['token_symbol']
        
        # Unallocated balance in warm wallet (handle NULL values properly)
        if pd.notna(row['unallocated_balance']) and row['unallocated_balance'] > 0:
            warm_wallet[token] = warm_wallet.get(token, 0) + float(row['unallocated_balance'])
        
        # Allocated balance to pools (handle NULL values properly)
        if pd.notna(row['allocated_balance']) and row['allocated_balance'] > 0 and pd.notna(row['pool_id']):
            key = (row['pool_id'], token)
            allocations[key] = allocations.get(key, 0) + float(row['allocated_balance'])
    
    logger.info(f"Warm wallet: {len(warm_wallet)} tokens, Total allocated positions: {len(allocations)}")
    return warm_wallet, allocations


def fetch_allocation_parameters(engine) -> Dict:
    """
    Fetches the latest allocation parameters.
    
    Args:
        engine: Database connection engine
        
    Returns:
        Dictionary of allocation parameters
    """
    try:
        query = """
        SELECT *
        FROM allocation_parameters
        ORDER BY timestamp DESC
        LIMIT 1;
        """
        df = pd.read_sql(query, engine)
        if df.empty:
            logger.warning("No allocation parameters found, using defaults")
            return {
                'max_alloc_percentage': 0.20,
                'conversion_rate': 0.0004,
                'min_transaction_value': 50.0
            }
        
        params = df.iloc[0].to_dict()
        logger.info(f"Loaded allocation parameters: max_alloc={params.get('max_alloc_percentage')}, tvl_limit={params.get('tvl_limit_percentage')}")
        return params
    except Exception as e:
        logger.error(f"Error fetching allocation parameters: {e}")
        # Return default values
        return {
            'max_alloc_percentage': 0.20,
            'conversion_rate': 0.0004,
            'min_transaction_value': 50.0
        }


def parse_pool_tokens(symbol: str) -> List[str]:
    """
    Extracts tokens from pool symbol.
    
    Args:
        symbol: Pool symbol (e.g., "DAI-USDC-USDT" or "GTUSDC")
        
    Returns:
        List of token symbols in uppercase
    """
    return [t.upper().strip() for t in symbol.split('-')]


def calculate_aum(warm_wallet: Dict[str, float], 
                  current_allocations: Dict[Tuple[str, str], float],
                  token_prices: Dict[str, float]) -> float:
    """
    Calculates total Assets Under Management in USD.
    
    Args:
        warm_wallet: Unallocated token balances
        current_allocations: Allocated positions
        token_prices: Token prices in USD
        
    Returns:
        Total AUM in USD
    """
    total_usd = 0.0
    
    try:
        # Warm wallet value
        for token, amount in warm_wallet.items():
            price = token_prices.get(token, 1.0)  # Default to $1 for stablecoins
            total_usd += amount * price
        
        # Allocated positions value
        for (pool_id, token), amount in current_allocations.items():
            price = token_prices.get(token, 1.0)
            total_usd += amount * price
        
        logger.info(f"Total AUM: ${total_usd:,.2f}")
        return total_usd
    except Exception as e:
        logger.error(f"Error calculating AUM: {e}")
        return 0.0


def build_token_universe(pools_df: pd.DataFrame, 
                         warm_wallet: Dict[str, float],
                         current_allocations: Dict[Tuple[str, str], float]) -> List[str]:
    """
    Builds the complete set of tokens needed for optimization.
    
    Args:
        pools_df: DataFrame of pool data
        warm_wallet: Current warm wallet balances
        current_allocations: Current pool allocations
        
    Returns:
        Sorted list of unique token symbols
    """
    try:
        tokens = set()
        
        # Tokens from pools
        for symbol in pools_df['symbol']:
            tokens.update(parse_pool_tokens(symbol))
        
        # Tokens in warm wallet
        tokens.update(warm_wallet.keys())
        
        # Tokens in current allocations
        for (pool_id, token) in current_allocations.keys():
            tokens.add(token)
        
        token_list = sorted(list(tokens))
        logger.info(f"Token universe: {len(token_list)} tokens - {token_list}")
        return token_list
    except Exception as e:
        logger.error(f"Error building token universe: {e}")
        return []