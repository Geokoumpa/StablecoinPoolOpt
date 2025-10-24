"""
Data Quality Report Module for Asset Allocation Optimization

This module provides comprehensive data quality analysis for the optimization process,
inspired by test_data_preparation_quality.py and analyze_model_failure_factors.py.
It generates detailed reports on data completeness, accuracy, and potential issues
that could affect optimization results.
"""

import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime, date, timezone
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_utils import get_db_connection
# Import functions directly to avoid circular imports
from database.db_utils import get_db_connection
import json as json_module

# Configure module logger
logger = logging.getLogger(__name__)


# ============================================================================
# DATA FETCHING FUNCTIONS (copied from optimize_allocations.py to avoid circular imports)
# ============================================================================

def fetch_pool_data(engine) -> pd.DataFrame:
    """Fetches approved pools with forecasted APY and metadata."""
    query = """
    SELECT
        pdm.pool_id,
        p.symbol,
        p.chain,
        p.protocol,
        pdm.forecasted_apy,
        pdm.forecasted_tvl,
        pdm.normalized_tokens
    FROM pool_daily_metrics pdm
    JOIN pools p ON pdm.pool_id = p.pool_id
    WHERE pdm.date = CURRENT_DATE 
      AND pdm.is_filtered_out = FALSE
      AND pdm.forecasted_apy IS NOT NULL
      AND pdm.forecasted_apy > 0
      AND pdm.forecasted_tvl IS NOT NULL
      AND pdm.forecasted_tvl > 0;
    """
    df = pd.read_sql(query, engine)
    logger.info(f"Loaded {len(df)} approved pools")
    return df


def fetch_token_prices(engine, tokens: List[str]) -> Dict[str, float]:
    """Fetches latest closing prices for given tokens."""
    if not tokens:
        return {}
    
    token_mapping = {token.lower(): token for token in tokens}
    tokens_lower = list(token_mapping.keys())
    tokens_str = "','".join(tokens_lower)
    query = f"""
    WITH ranked_ohlcv AS (
        SELECT
            LOWER(symbol) as symbol_lower,
            CASE 
                WHEN raw_json_data ? 'USD' THEN (raw_json_data->'USD'->>'close')::float
                WHEN raw_json_data ? 'USDT' THEN (raw_json_data->'USDT'->>'close')::float
                WHEN raw_json_data ? 'BTC' THEN (raw_json_data->'BTC'->>'close')::float
                WHEN raw_json_data ? 'ETH' THEN (raw_json_data->'ETH'->>'close')::float
                ELSE NULL
            END as close_price,
            data_timestamp as ts,
            ROW_NUMBER() OVER(
                PARTITION BY LOWER(symbol) 
                ORDER BY data_timestamp DESC
            ) as rn
        FROM raw_coinmarketcap_ohlcv
        WHERE LOWER(symbol) IN ('{tokens_str}')
    )
    SELECT symbol_lower, close_price
    FROM ranked_ohlcv
    WHERE rn = 1;
    """
    df = pd.read_sql(query, engine)
    prices = {}
    for _, row in df.iterrows():
        if pd.notna(row['close_price']):
            original_token = token_mapping.get(row['symbol_lower'])
            if original_token:
                prices[original_token] = row['close_price']
    logger.info(f"Loaded prices for {len(prices)} tokens")
    return prices


def fetch_gas_fee_data(engine) -> Tuple[float, float, float, float, float]:
    """
    Fetches forecasted gas fee components and ETH price.
    
    Returns:
        Tuple of (eth_price_usd, base_fee_transfer_gwei, base_fee_swap_gwei, priority_fee_gwei, min_gas_units)
    """
    # Fetch ETH price
    eth_price_query = """
    WITH ranked_eth AS (
        SELECT
            (raw_json_data->'USD'->>'close')::float as close_price,
            ROW_NUMBER() OVER(ORDER BY (raw_json_data->'USD'->>'timestamp')::timestamp DESC) as rn
        FROM raw_coinmarketcap_ohlcv
        WHERE symbol = 'ETH'
    )
    SELECT close_price
    FROM ranked_eth
    WHERE rn = 1;
    """
    eth_df = pd.read_sql(eth_price_query, engine)
    eth_price = eth_df['close_price'].iloc[0] if not eth_df.empty and pd.notna(eth_df['close_price'].iloc[0]) else 3000.0
    
    # Gas fee components based on requirements
    base_fee_transfer_gwei = 10.0  # Base fee for transfer/deposit
    base_fee_swap_gwei = 30.0       # Base fee for swap
    priority_fee_gwei = 10.0        # Priority fee
    min_gas_units = 21000          # Minimum gas units
    
    logger.info(f"ETH price: ${eth_price:.2f}")
    logger.info(f"Gas fee components - Base transfer: {base_fee_transfer_gwei} Gwei, Base swap: {base_fee_swap_gwei} Gwei, Priority: {priority_fee_gwei} Gwei, Min gas units: {min_gas_units}")
    
    return eth_price, base_fee_transfer_gwei, base_fee_swap_gwei, priority_fee_gwei, min_gas_units

def calculate_gas_fee_usd(gas_units: float, base_fee_gwei: float, priority_fee_gwei: float, eth_price_usd: float) -> float:
    """
    Calculate gas fee in USD based on the formula: Gas fee = Gas units * (base fee + priority fee)
    
    Args:
        gas_units: Gas units (limit) for the transaction
        base_fee_gwei: Base fee in Gwei
        priority_fee_gwei: Priority fee in Gwei
        eth_price_usd: ETH price in USD
        
    Returns:
        Gas fee in USD
    """
    total_fee_gwei = gas_units * (base_fee_gwei + priority_fee_gwei)
    gas_fee_usd = total_fee_gwei * 1e-9 * eth_price_usd
    return gas_fee_usd


def calculate_transaction_gas_fees(eth_price_usd: float, base_fee_transfer_gwei: float, 
                                   base_fee_swap_gwei: float, priority_fee_gwei: float, 
                                   min_gas_units: float) -> Dict[str, float]:
    """
    Calculate gas fees for different transaction types.
    
    Args:
        eth_price_usd: ETH price in USD
        base_fee_transfer_gwei: Base fee for transfer/deposit in Gwei
        base_fee_swap_gwei: Base fee for swap in Gwei
        priority_fee_gwei: Priority fee in Gwei
        min_gas_units: Minimum gas units
        
    Returns:
        Dictionary with gas fees for different transaction types in USD
    """
    # Pool allocation/withdrawal gas fee (using transfer base fee)
    pool_transaction_gas_fee_usd = calculate_gas_fee_usd(
        min_gas_units, base_fee_transfer_gwei, priority_fee_gwei, eth_price_usd
    )
    
    # Token swap gas fee (using swap base fee)
    token_swap_gas_fee_usd = calculate_gas_fee_usd(
        min_gas_units, base_fee_swap_gwei, priority_fee_gwei, eth_price_usd
    )
    
    gas_fees = {
        'allocation': pool_transaction_gas_fee_usd,      # Allocating to pools
        'withdrawal': pool_transaction_gas_fee_usd,      # Withdrawing from pools
        'conversion': token_swap_gas_fee_usd,            # Token swaps/conversions
        'transfer': pool_transaction_gas_fee_usd,        # General transfers
        'deposit': pool_transaction_gas_fee_usd          # Deposits to pools
    }
    
    logger.info(f"Transaction gas fees - Pool Allocation/Withdrawal: ${pool_transaction_gas_fee_usd:.6f}, Token Swap/Conversion: ${token_swap_gas_fee_usd:.6f}")
    
    return gas_fees
    
    return eth_price, base_fee_transfer_gwei, base_fee_swap_gwei, priority_fee_gwei, min_gas_units


def fetch_current_balances(engine) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
    """Fetches current token balances from warm wallet and allocated positions."""
    try:
        from config import MAIN_ASSET_HOLDING_ADDRESS
    except ImportError:
        logger.warning("Could not import MAIN_ASSET_HOLDING_ADDRESS, using empty balances")
        return {}, {}
    
    if not MAIN_ASSET_HOLDING_ADDRESS:
        logger.error("MAIN_ASSET_HOLDING_ADDRESS not configured")
        return {}, {}
    
    warm_wallet = {}
    allocations = {}
    
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
    
    for _, row in df.iterrows():
        token = row['token_symbol']
        
        if pd.notna(row['unallocated_balance']) and row['unallocated_balance'] > 0:
            warm_wallet[token] = warm_wallet.get(token, 0) + float(row['unallocated_balance'])
        
        if pd.notna(row['allocated_balance']) and row['allocated_balance'] > 0 and pd.notna(row['pool_id']):
            key = (row['pool_id'], token)
            allocations[key] = allocations.get(key, 0) + float(row['allocated_balance'])
    
    logger.info(f"Warm wallet: {len(warm_wallet)} tokens, Total allocated positions: {len(allocations)}")
    return warm_wallet, allocations


def fetch_allocation_parameters(engine) -> Dict:
    """Fetches the latest allocation parameters."""
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


def parse_pool_tokens_with_mapping(symbol: str, normalized_tokens_json: str = None) -> List[str]:
    """Extracts tokens from pool symbol and applies normalized mappings if available."""
    tokens = [t.upper().strip() for t in symbol.split('-')]
    
    if normalized_tokens_json:
        try:
            token_mappings = json_module.loads(normalized_tokens_json)
            normalized_tokens = []
            for token in tokens:
                mapped_token = None
                for original, approved in token_mappings.items():
                    if token.lower() == original.lower():
                        mapped_token = approved.upper()
                        break
                normalized_tokens.append(mapped_token if mapped_token else token)
            return normalized_tokens
        except (json_module.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse normalized_tokens_json: {e}")
    
    return tokens


def build_token_universe(pools_df: pd.DataFrame, 
                         warm_wallet: Dict[str, float],
                         current_allocations: Dict[Tuple[str, str], float]) -> List[str]:
    """Builds the complete set of tokens needed for optimization."""
    tokens = set()
    
    for _, row in pools_df.iterrows():
        symbol = row['symbol']
        normalized_tokens_json = row.get('normalized_tokens')
        pool_tokens = parse_pool_tokens_with_mapping(symbol, normalized_tokens_json)
        tokens.update(pool_tokens)
    
    tokens.update(warm_wallet.keys())
    
    for (pool_id, token) in current_allocations.keys():
        tokens.add(token)
    
    token_list = sorted(list(tokens))
    logger.info(f"Token universe: {len(token_list)} tokens - {token_list}")
    return token_list


def calculate_aum(warm_wallet: Dict[str, float], 
                  current_allocations: Dict[Tuple[str, str], float],
                  token_prices: Dict[str, float]) -> float:
    """Calculates total Assets Under Management in USD."""
    total_usd = 0.0
    
    for token, amount in warm_wallet.items():
        price = token_prices.get(token, 1.0)
        total_usd += amount * price
    
    for (pool_id, token), amount in current_allocations.items():
        price = token_prices.get(token, 1.0)
        total_usd += amount * price
    
    logger.info(f"Total AUM: ${total_usd:,.2f}")
    return total_usd

class DataQualityReporter:
    """
    Comprehensive data quality reporter for asset allocation optimization.
    
    This class analyzes all data inputs to the optimization process and provides
    detailed reporting on:
    - Data completeness and coverage
    - Value ranges and anomalies
    - Fallback values and defaults
    - Potential model failure factors
    - Recommendations for improvement
    """
    
    def __init__(self):
        """Initialize the data quality reporter."""
        self.engine = get_db_connection()
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'pool_data_quality': {},
            'token_price_quality': {},
            'gas_fee_quality': {},
            'balance_quality': {},
            'parameter_quality': {},
            'model_feasibility': {},
            'abnormal_values': {},
            'fallback_values': {},
            'recommendations': [],
            'data_freshness': {}
        }
        
        # Define expected value ranges and thresholds
        self.expected_ranges = {
            'apy': {'min': 0, 'max': 100, 'unit': '%', 'description': 'Annual Percentage Yield'},
            'tvl': {'min': 1000, 'max': 1e12, 'unit': 'USD', 'description': 'Total Value Locked'},
            'gas_gwei': {'min': 0.01, 'max': 1000, 'unit': 'Gwei', 'description': 'Gas price in Gwei'},
            'eth_price': {'min': 100, 'max': 10000, 'unit': 'USD', 'description': 'ETH price in USD'},
            'token_price': {'min': 0.0001, 'max': 100000, 'unit': 'USD', 'description': 'Token price in USD'},
            'balance': {'min': 0, 'max': 1e12, 'unit': 'tokens', 'description': 'Token balance'},
            'max_alloc_pct': {'min': 0.01, 'max': 0.5, 'unit': '%', 'description': 'Maximum allocation percentage'},
            'tvl_limit_pct': {'min': 0.001, 'max': 0.2, 'unit': '%', 'description': 'TVL limit percentage'},
            'conversion_rate': {'min': 0.0001, 'max': 0.01, 'unit': '%', 'description': 'Token conversion rate'}
        }
        
        # Known fallback/default values
        self.fallback_values = {
            'gas_gwei': 50.0,
            'eth_price': 3000.0,
            'token_price': 1.0,
            'max_alloc_percentage': 0.20,
            'conversion_rate': 0.0004,
            'min_transaction_value': 50.0
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            'min_pool_count': 10,
            'min_token_coverage': 0.8,
            'max_abnormal_apy_pct': 10,
            'max_missing_prices': 5,
            'min_balance_data': True
        }
    
    def _add_summary(self, key: str, value: Any):
        """Add a summary statistic to the report."""
        self.report['summary'][key] = value
    
    def _add_fallback(self, category: str, item: str, value: Any, expected_source: str):
        """Record a fallback value usage."""
        if category not in self.report['fallback_values']:
            self.report['fallback_values'][category] = []
        
        self.report['fallback_values'][category].append({
            'item': item,
            'value': value,
            'expected_source': expected_source,
            'detected_at': datetime.now().isoformat()
        })
        
        logger.warning(f"Fallback value used for {category}.{item}: {value} (expected from {expected_source})")
    
    def _add_abnormal_value(self, category: str, item: str, value: Any, 
                           reason: str, severity: str = 'warning'):
        """Record an abnormal value."""
        if category not in self.report['abnormal_values']:
            self.report['abnormal_values'][category] = []
        
        self.report['abnormal_values'][category].append({
            'item': item,
            'value': value,
            'reason': reason,
            'severity': severity,
            'detected_at': datetime.now().isoformat()
        })
        
        log_level = logging.ERROR if severity == 'critical' else logging.WARNING
        logger.log(log_level, f"Abnormal value in {category}.{item}: {value} - {reason}")
    
    def _add_recommendation(self, recommendation: str, priority: str = 'medium', details: List[str] = None):
        """Add a recommendation to the report."""
        self.report['recommendations'].append({
            'recommendation': recommendation,
            'priority': priority,
            'details': details or [],
            'created_at': datetime.now().isoformat()
        })
    
    def analyze_pool_data(self) -> pd.DataFrame:
        """Analyze pool data quality."""
        logger.info("Analyzing pool data quality...")
        
        try:
            pools_df = fetch_pool_data(self.engine)
            
            if pools_df.empty:
                self._add_abnormal_value(
                    'pool_data',
                    'no_data',
                    'None',
                    "No approved pools found in database",
                    'critical'
                )
                self._add_recommendation(
                    "Check pool filtering logic and data pipeline",
                    'critical',
                    ["Verify pool_daily_metrics table has data", "Check is_filtered_out flag", "Verify forecasted_apy is not NULL"]
                )
                return pools_df
            
            # Basic statistics
            self._add_summary('total_pools', len(pools_df))
            self._add_summary('unique_protocols', pools_df['protocol'].nunique())
            self._add_summary('unique_chains', pools_df['chain'].nunique())
            
            # APY analysis
            apy_stats = pools_df['forecasted_apy'].describe()
            self._add_summary('apy_min', apy_stats['min'])
            self._add_summary('apy_max', apy_stats['max'])
            self._add_summary('apy_mean', apy_stats['mean'])
            self._add_summary('apy_median', apy_stats['50%'])
            
            # Flag abnormal APY values
            high_apy_pools = pools_df[pools_df['forecasted_apy'] > self.expected_ranges['apy']['max']]
            for _, pool in high_apy_pools.iterrows():
                self._add_abnormal_value(
                    'pool_apy',
                    f"{pool['pool_id']} ({pool['symbol']})",
                    pool['forecasted_apy'],
                    f"APY exceeds {self.expected_ranges['apy']['max']}%",
                    'warning' if pool['forecasted_apy'] < 100 else 'critical'
                )
            
            # TVL analysis
            tvl_stats = pools_df['forecasted_tvl'].describe()
            self._add_summary('tvl_min', tvl_stats['min'])
            self._add_summary('tvl_max', tvl_stats['max'])
            self._add_summary('tvl_mean', tvl_stats['mean'])
            self._add_summary('tvl_median', tvl_stats['50%'])
            self._add_summary('total_tvl', pools_df['forecasted_tvl'].sum())
            
            # Flag abnormal TVL values
            low_tvl_pools = pools_df[pools_df['forecasted_tvl'] < self.expected_ranges['tvl']['min']]
            for _, pool in low_tvl_pools.iterrows():
                self._add_abnormal_value(
                    'pool_tvl',
                    f"{pool['pool_id']} ({pool['symbol']})",
                    pool['forecasted_tvl'],
                    f"TVL below ${self.expected_ranges['tvl']['min']:,.0f}",
                    'warning'
                )
            
            # Normalized tokens analysis
            pools_with_mappings = pools_df[pools_df['normalized_tokens'].notna()]
            self._add_summary('pools_with_token_mappings', len(pools_with_mappings))
            
            # Analyze token mapping quality
            mapping_issues = 0
            invalid_json_count = 0
            
            for _, pool in pools_with_mappings.iterrows():
                try:
                    mappings = json.loads(pool['normalized_tokens'])
                    original_tokens = parse_pool_tokens_with_mapping(pool['symbol'], None)
                    mapped_tokens = parse_pool_tokens_with_mapping(pool['symbol'], pool['normalized_tokens'])
                    
                    if original_tokens != mapped_tokens:
                        mapping_issues += 1
                        
                except (json.JSONDecodeError, TypeError):
                    invalid_json_count += 1
                    self._add_abnormal_value(
                        'token_mapping',
                        f"{pool['pool_id']} ({pool['symbol']})",
                        pool['normalized_tokens'],
                        "Invalid JSON in normalized_tokens",
                        'critical'
                    )
            
            self._add_summary('pools_with_token_changes', mapping_issues)
            self._add_summary('invalid_token_mappings', invalid_json_count)
            
            # Data completeness checks
            null_counts = pools_df.isnull().sum()
            for col, count in null_counts.items():
                if count > 0:
                    self._add_abnormal_value(
                        'pool_data_completeness',
                        col,
                        count,
                        f"{count} NULL values in {col}",
                        'warning' if count < len(pools_df) * 0.1 else 'critical'
                    )
            
            # Store detailed pool data quality info
            self.report['pool_data_quality'] = {
                'total_records': len(pools_df),
                'columns': list(pools_df.columns),
                'data_types': pools_df.dtypes.to_dict(),
                'null_counts': null_counts.to_dict(),
                'extreme_apy_pools': len(high_apy_pools),
                'low_tvl_pools': len(low_tvl_pools),
                'mapping_issues': mapping_issues,
                'invalid_mappings': invalid_json_count
            }
            
            # Check if we have enough pools for meaningful optimization
            if len(pools_df) < self.quality_thresholds['min_pool_count']:
                self._add_recommendation(
                    f"Consider adding more pools (currently {len(pools_df)} below threshold of {self.quality_thresholds['min_pool_count']})",
                    'medium',
                    ["Review pool filtering criteria", "Check data pipeline for missing pools"]
                )
            
            logger.info(f"Pool data analysis complete: {len(pools_df)} pools, {len(high_apy_pools)} high APY, {len(low_tvl_pools)} low TVL")
            return pools_df
            
        except Exception as e:
            logger.error(f"Error analyzing pool data: {e}")
            self._add_abnormal_value(
                'pool_data',
                'analysis_error',
                str(e),
                "Failed to analyze pool data",
                'critical'
            )
            return pd.DataFrame()
    
    def analyze_token_prices(self, tokens: List[str]) -> Dict[str, float]:
        """Analyze token price data quality."""
        logger.info("Analyzing token price data quality...")
        
        try:
            prices = fetch_token_prices(self.engine, tokens)
            
            self._add_summary('total_tokens_requested', len(tokens))
            self._add_summary('tokens_with_prices', len(prices))
            coverage_pct = (len(prices) / len(tokens) * 100) if tokens else 0
            self._add_summary('price_coverage_pct', coverage_pct)
            
            # Check for missing prices
            tokens_without_prices = set(tokens) - set(prices.keys())
            for token in tokens_without_prices:
                self._add_fallback(
                    'token_prices',
                    token,
                    self.fallback_values['token_price'],
                    'CoinMarketCap OHLCV data'
                )
            
            # Check price coverage quality
            if coverage_pct < self.quality_thresholds['min_token_coverage'] * 100:
                self._add_recommendation(
                    f"Improve token price coverage (currently {coverage_pct:.1f}%)",
                    'high',
                    [f"Add price data for {len(tokens_without_prices)} missing tokens", 
                     "Check CoinMarketCap API configuration",
                     "Consider alternative price sources"]
                )
            
            # Analyze price values for abnormalities
            if prices:
                price_values = list(prices.values())
                self._add_summary('price_min', min(price_values))
                self._add_summary('price_max', max(price_values))
                self._add_summary('price_mean', np.mean(price_values))
                self._add_summary('price_median', np.median(price_values))
                
                # Flag abnormal prices
                for token, price in prices.items():
                    if price < self.expected_ranges['token_price']['min']:
                        self._add_abnormal_value(
                            'token_price',
                            token,
                            price,
                            f"Price below ${self.expected_ranges['token_price']['min']}",
                            'warning'
                        )
                    elif price > self.expected_ranges['token_price']['max']:
                        self._add_abnormal_value(
                            'token_price',
                            token,
                            price,
                            f"Price above ${self.expected_ranges['token_price']['max']}",
                            'warning'
                        )
            
            # Store detailed token price quality info
            self.report['token_price_quality'] = {
                'requested_tokens': tokens,
                'fetched_prices': prices,
                'missing_tokens': list(tokens_without_prices),
                'coverage_percentage': coverage_pct,
                'price_stats': {
                    'count': len(prices),
                    'min': min(prices.values()) if prices else None,
                    'max': max(prices.values()) if prices else None,
                    'mean': np.mean(list(prices.values())) if prices else None,
                    'median': np.median(list(prices.values())) if prices else None
                }
            }
            
            logger.info(f"Token price analysis complete: {len(prices)}/{len(tokens)} tokens have prices ({coverage_pct:.1f}% coverage)")
            return prices
            
        except Exception as e:
            logger.error(f"Error analyzing token prices: {e}")
            self._add_abnormal_value(
                'token_prices',
                'analysis_error',
                str(e),
                "Failed to analyze token prices",
                'critical'
            )
            return {}
    
    def analyze_gas_fee_data(self) -> Tuple[float, float, float, float, float]:
        """Analyze gas fee data quality."""
        logger.info("Analyzing gas fee data quality...")
        
        try:
            eth_price, base_fee_transfer_gwei, base_fee_swap_gwei, priority_fee_gwei, min_gas_units = fetch_gas_fee_data(self.engine)
            
            # Calculate transaction gas fees
            gas_fees = calculate_transaction_gas_fees(
                eth_price, base_fee_transfer_gwei, base_fee_swap_gwei, priority_fee_gwei, min_gas_units
            )
            
            # Check for fallback values
            if eth_price == self.fallback_values['eth_price']:
                self._add_fallback(
                    'gas_fees',
                    'eth_price',
                    eth_price,
                    'CoinMarketCap OHLCV data'
                )
            
            # Check for abnormal values
            if eth_price < self.expected_ranges['eth_price']['min']:
                self._add_abnormal_value(
                    'gas_fees',
                    'eth_price',
                    eth_price,
                    f"ETH price below ${self.expected_ranges['eth_price']['min']}",
                    'warning'
                )
            elif eth_price > self.expected_ranges['eth_price']['max']:
                self._add_abnormal_value(
                    'gas_fees',
                    'eth_price',
                    eth_price,
                    f"ETH price above ${self.expected_ranges['eth_price']['max']}",
                    'warning'
                )
            
            # Add summary values
            self._add_summary('eth_price_usd', eth_price)
            self._add_summary('base_fee_transfer_gwei', base_fee_transfer_gwei)
            self._add_summary('base_fee_swap_gwei', base_fee_swap_gwei)
            self._add_summary('priority_fee_gwei', priority_fee_gwei)
            self._add_summary('min_gas_units', min_gas_units)
            self._add_summary('pool_allocation_gas_fee_usd', gas_fees['allocation'])
            self._add_summary('pool_withdrawal_gas_fee_usd', gas_fees['withdrawal'])
            self._add_summary('token_swap_gas_fee_usd', gas_fees['conversion'])
            
            # Store detailed gas fee quality info
            self.report['gas_fee_quality'] = {
                'eth_price_usd': eth_price,
                'base_fee_transfer_gwei': base_fee_transfer_gwei,
                'base_fee_swap_gwei': base_fee_swap_gwei,
                'priority_fee_gwei': priority_fee_gwei,
                'min_gas_units': min_gas_units,
                'pool_allocation_gas_fee_usd': gas_fees['allocation'],
                'pool_withdrawal_gas_fee_usd': gas_fees['withdrawal'],
                'token_swap_gas_fee_usd': gas_fees['conversion'],
                'is_eth_fallback': eth_price == self.fallback_values['eth_price'],
                'eth_abnormal': eth_price < self.expected_ranges['eth_price']['min'] or eth_price > self.expected_ranges['eth_price']['max']
            }
            
            logger.info(f"Gas fee analysis complete: {base_fee_transfer_gwei:.2f} Gwei base transfer, {base_fee_swap_gwei:.2f} Gwei base swap, {priority_fee_gwei:.2f} Gwei priority, ${eth_price:.2f} ETH")
            return eth_price, base_fee_transfer_gwei, base_fee_swap_gwei, priority_fee_gwei, min_gas_units
            
        except Exception as e:
            logger.error(f"Error analyzing gas fee data: {e}")
            self._add_abnormal_value(
                'gas_fees',
                'analysis_error',
                str(e),
                "Failed to analyze gas fee data",
                'critical'
            )
            return self.fallback_values['eth_price'], 10.0, 30.0, 10.0, 21000
    
    def analyze_balance_data(self) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
        """Analyze balance data quality."""
        logger.info("Analyzing balance data quality...")
        
        try:
            warm_wallet, allocations = fetch_current_balances(self.engine)
            
            # Check if we have any balance data
            if not warm_wallet and not allocations:
                self._add_abnormal_value(
                    'balances',
                    'no_data',
                    'None',
                    "No balance data found for configured wallet",
                    'critical'
                )
                self._add_recommendation(
                    "Check wallet configuration and balance data pipeline",
                    'critical',
                    ["Verify MAIN_ASSET_HOLDING_ADDRESS configuration", 
                     "Check daily_balances table for today's data",
                     "Verify balance collection process"]
                )
            
            # Analyze warm wallet balances
            total_warm_value = 0
            negative_balances = 0
            
            for token, amount in warm_wallet.items():
                if amount < 0:
                    negative_balances += 1
                    self._add_abnormal_value(
                        'balances',
                        f'warm_wallet_{token}',
                        amount,
                        "Negative balance in warm wallet",
                        'critical'
                    )
                total_warm_value += amount
            
            # Analyze allocated balances
            total_allocated_value = 0
            for (pool_id, token), amount in allocations.items():
                if amount < 0:
                    self._add_abnormal_value(
                        'balances',
                        f'allocated_{pool_id}_{token}',
                        amount,
                        "Negative allocated balance",
                        'critical'
                    )
                total_allocated_value += amount
            
            self._add_summary('warm_wallet_tokens', len(warm_wallet))
            self._add_summary('allocated_positions', len(allocations))
            self._add_summary('total_warm_balance', total_warm_value)
            self._add_summary('total_allocated_balance', total_allocated_value)
            self._add_summary('negative_balances', negative_balances)
            
            # Store detailed balance quality info
            self.report['balance_quality'] = {
                'warm_wallet': warm_wallet,
                'allocations': {f"{k[0]}_{k[1]}": v for k, v in allocations.items()},
                'warm_wallet_total': total_warm_value,
                'allocated_total': total_allocated_value,
                'has_data': bool(warm_wallet or allocations),
                'negative_balance_count': negative_balances
            }
            
            logger.info(f"Balance analysis complete: {len(warm_wallet)} warm wallet tokens, {len(allocations)} allocated positions")
            return warm_wallet, allocations
            
        except Exception as e:
            logger.error(f"Error analyzing balance data: {e}")
            self._add_abnormal_value(
                'balances',
                'analysis_error',
                str(e),
                "Failed to analyze balance data",
                'critical'
            )
            return {}, {}
    
    def analyze_allocation_parameters(self) -> Dict:
        """Analyze allocation parameters quality."""
        logger.info("Analyzing allocation parameters...")
        
        try:
            params = fetch_allocation_parameters(self.engine)
            
            # Check for fallback values - only flag if the parameter was actually NULL in the database
            # We need to check if the value came from the database or from the fallback logic in fetch_allocation_parameters
            # Since we can't easily detect this from the returned dictionary, we'll only flag obvious cases
            
            # Note: The fetch_allocation_parameters function already handles NULL values by providing defaults
            # So if we're getting a value that matches the default, it might be legitimate or it might be a fallback
            # This is a limitation of the current implementation
            
            # Check parameter ranges
            for param_name, param_value in params.items():
                if param_name in self.expected_ranges:
                    range_info = self.expected_ranges[param_name]
                    if param_value < range_info['min'] or param_value > range_info['max']:
                        self._add_abnormal_value(
                            'allocation_parameters',
                            param_name,
                            param_value,
                            f"Value {param_value} outside expected range [{range_info['min']}, {range_info['max']}]",
                            'warning'
                        )
            
            # Store summary stats
            for key, value in params.items():
                if isinstance(value, (int, float)):
                    self._add_summary(f'param_{key}', value)
            
            # Store detailed parameter quality info
            # Note: We're not detecting fallback values for allocation_parameters since it's difficult to distinguish
            # between legitimate values that happen to match defaults and actual fallback usage
            self.report['parameter_quality'] = {
                'parameters': params,
                'has_fallback_values': False,  # Not detecting for allocation_parameters
                'abnormal_parameters': len([
                    k for k, v in params.items() 
                    if k in self.expected_ranges and 
                    (v < self.expected_ranges[k]['min'] or v > self.expected_ranges[k]['max'])
                ])
            }
            
            logger.info(f"Allocation parameters analysis complete: {len(params)} parameters")
            return params
            
        except Exception as e:
            logger.error(f"Error analyzing allocation parameters: {e}")
            self._add_abnormal_value(
                'allocation_parameters',
                'analysis_error',
                str(e),
                "Failed to analyze allocation parameters",
                'critical'
            )
            return {}
    
    def analyze_model_feasibility(self, pools_df: pd.DataFrame, token_prices: Dict[str, float], 
                                alloc_params: Dict, warm_wallet: Dict[str, float], 
                                current_allocations: Dict[Tuple[str, str], float]):
        """Analyze model feasibility and potential failure factors."""
        logger.info("Analyzing model feasibility...")
        
        try:
            # Calculate AUM
            total_aum = calculate_aum(warm_wallet, current_allocations, token_prices)
            self._add_summary('total_aum', total_aum)
            
            # Analyze pool diversity
            token_universe = build_token_universe(pools_df, warm_wallet, current_allocations)
            self._add_summary('token_universe_size', len(token_universe))
            
            # Check token coverage
            tokens_without_prices = [t for t in token_universe if t not in token_prices]
            self._add_summary('tokens_without_prices', len(tokens_without_prices))
            
            if len(tokens_without_prices) > self.quality_thresholds['max_missing_prices']:
                self._add_recommendation(
                    f"Too many tokens without prices ({len(tokens_without_prices)})",
                    'high',
                    ["Add price sources for missing tokens", "Consider excluding pools with unpriced tokens"]
                )
            
            # Analyze gas fee impact
            eth_price, base_fee_transfer_gwei, base_fee_swap_gwei, priority_fee_gwei, min_gas_units = fetch_gas_fee_data(self.engine)
            gas_fees = calculate_transaction_gas_fees(
                eth_price, base_fee_transfer_gwei, base_fee_swap_gwei, priority_fee_gwei, min_gas_units
            )
            
            # Estimate transaction costs for all pools
            estimated_transactions = len(pools_df) * 3  # Assume 3 transactions per pool on average
            # Use average gas fee (mix of allocation/withdrawal and conversion fees)
            avg_gas_fee = (gas_fees['allocation'] + gas_fees['conversion']) / 2
            total_gas_cost = estimated_transactions * avg_gas_fee
            
            self._add_summary('estimated_transactions', estimated_transactions)
            self._add_summary('total_gas_cost_estimate', total_gas_cost)
            self._add_summary('gas_cost_as_pct_of_aum', (total_gas_cost / total_aum * 100) if total_aum > 0 else 0)
            
            # Check if gas costs are prohibitive
            if total_gas_cost > total_aum * 0.01:  # Gas costs > 1% of AUM
                self._add_recommendation(
                    "High gas costs relative to AUM",
                    'medium',
                    [f"Gas costs: ${total_gas_cost:.2f} ({total_gas_cost/total_aum*100:.2f}% of AUM)",
                     "Consider reducing pool count or waiting for lower gas prices"]
                )
            
            # Analyze constraint feasibility
            max_alloc_pct = alloc_params.get('max_alloc_percentage', 0.20)
            tvl_limit_pct = alloc_params.get('tvl_limit_percentage', 0.05)
            
            # Calculate total capacity
            total_capacity = sum(pools_df['forecasted_tvl'] * tvl_limit_pct)
            self._add_summary('total_pool_capacity', total_capacity)
            
            # Check if AUM exceeds capacity
            if total_aum > total_capacity:
                self._add_abnormal_value(
                    'model_feasibility',
                    'aum_exceeds_capacity',
                    f"AUM: ${total_aum:,.0f}, Capacity: ${total_capacity:,.0f}",
                    "Total AUM exceeds pool capacity",
                    'critical'
                )
                self._add_recommendation(
                    "AUM exceeds total pool capacity",
                    'critical',
                    ["Increase TVL limits", "Add more pools", "Reduce allocation limits"]
                )
            
            # Store detailed feasibility analysis
            self.report['model_feasibility'] = {
                'total_aum': total_aum,
                'token_universe_size': len(token_universe),
                'tokens_without_prices': len(tokens_without_prices),
                'estimated_transactions': estimated_transactions,
                'total_gas_cost': total_gas_cost,
                'gas_cost_pct_of_aum': (total_gas_cost / total_aum * 100) if total_aum > 0 else 0,
                'total_pool_capacity': total_capacity,
                'aum_exceeds_capacity': total_aum > total_capacity,
                'constraint_feasible': total_aum <= total_capacity and len(tokens_without_prices) <= self.quality_thresholds['max_missing_prices']
            }
            
            logger.info(f"Model feasibility analysis complete: AUM=${total_aum:,.0f}, Capacity=${total_capacity:,.0f}, Gas cost=${total_gas_cost:.2f}")
            
        except Exception as e:
            logger.error(f"Error analyzing model feasibility: {e}")
            self._add_abnormal_value(
                'model_feasibility',
                'analysis_error',
                str(e),
                "Failed to analyze model feasibility",
                'critical'
            )
    
    def analyze_data_freshness(self):
        """Analyze the freshness of data sources."""
        logger.info("Analyzing data freshness...")
        
        try:
            freshness_info = {}
            
            # Check pool data freshness
            pool_query = """
            SELECT 
                date,
                COUNT(*) as pool_count
            FROM pool_daily_metrics
            WHERE date = CURRENT_DATE
            GROUP BY date
            """
            pool_df = pd.read_sql(pool_query, self.engine)
            
            if not pool_df.empty:
                freshness_info['pool_data'] = {
                    'date': str(date.today()),
                    'record_count': pool_df['pool_count'].iloc[0],
                    'has_current_data': True
                }
            else:
                freshness_info['pool_data'] = {
                    'date': str(date.today()),
                    'record_count': 0,
                    'has_current_data': False
                }
            
            # Check price data freshness
            price_query = """
            SELECT 
                symbol,
                data_timestamp,
                AGE(NOW(), data_timestamp) as age
            FROM raw_coinmarketcap_ohlcv
            WHERE symbol IN ('ETH', 'USDC', 'USDT')
            ORDER BY data_timestamp DESC
            LIMIT 10
            """
            price_df = pd.read_sql(price_query, self.engine)
            
            if not price_df.empty:
                avg_age_hours = price_df['age'].apply(lambda x: x.total_seconds() / 3600).mean()
                freshness_info['price_data'] = {
                    'sample_count': len(price_df),
                    'avg_age_hours': avg_age_hours,
                    'latest_timestamp': str(price_df['data_timestamp'].iloc[0])
                }
            
            # Check gas fee data freshness
            gas_query = """
            SELECT 
                date,
                forecasted_max_gas_gwei
            FROM gas_fees_daily
            WHERE date = CURRENT_DATE
            LIMIT 1
            """
            gas_df = pd.read_sql(gas_query, self.engine)
            
            if not gas_df.empty:
                freshness_info['gas_data'] = {
                    'date': str(date.today()),
                    'gas_gwei': gas_df['forecasted_max_gas_gwei'].iloc[0],
                    'has_current_data': True
                }
            else:
                freshness_info['gas_data'] = {
                    'date': str(date.today()),
                    'gas_gwei': None,
                    'has_current_data': False
                }
            
            # Check balance data freshness
            balance_query = """
            SELECT 
                date,
                COUNT(*) as record_count
            FROM daily_balances
            WHERE date = CURRENT_DATE
            GROUP BY date
            """
            balance_df = pd.read_sql(balance_query, self.engine)
            
            if not balance_df.empty:
                freshness_info['balance_data'] = {
                    'date': str(date.today()),
                    'record_count': balance_df['record_count'].iloc[0],
                    'has_current_data': True
                }
            else:
                freshness_info['balance_data'] = {
                    'date': str(date.today()),
                    'record_count': 0,
                    'has_current_data': False
                }
            
            self.report['data_freshness'] = freshness_info
            
            # Check for stale data
            for data_type, info in freshness_info.items():
                if data_type == 'pool_data' and not info.get('has_current_data', False):
                    self._add_abnormal_value(
                        'data_freshness',
                        data_type,
                        'No current data',
                        f"No data available for today",
                        'warning'
                    )
            
            logger.info(f"Data freshness analysis complete: {len(freshness_info)} data sources checked")
            
        except Exception as e:
            logger.error(f"Error analyzing data freshness: {e}")
            self._add_abnormal_value(
                'data_freshness',
                'analysis_error',
                str(e),
                "Failed to analyze data freshness",
                'warning'
            )
    
    def generate_recommendations(self):
        """Generate comprehensive recommendations based on analysis."""
        logger.info("Generating recommendations...")
        
        # Critical issues first
        critical_issues = []
        for category, items in self.report['abnormal_values'].items():
            critical_items = [i for i in items if i['severity'] == 'critical']
            if critical_items:
                critical_issues.extend([f"{category}: {i['reason']}" for i in critical_items])
        
        if critical_issues:
            self._add_recommendation(
                "Address critical data quality issues immediately",
                'critical',
                critical_issues
            )
        
        # Fallback values
        total_fallbacks = sum(len(items) for items in self.report['fallback_values'].values())
        if total_fallbacks > 0:
            self._add_recommendation(
                f"Reduce reliance on fallback values ({total_fallbacks} detected)",
                'high',
                ["Improve data pipeline reliability", "Add redundant data sources", "Implement better error handling"]
            )
        
        # Data coverage
        price_coverage = self.report['summary'].get('price_coverage_pct', 0)
        if price_coverage < 95:
            self._add_recommendation(
                f"Improve token price coverage (currently {price_coverage:.1f}%)",
                'high',
                ["Add more tokens to price tracking", "Check API rate limits", "Consider alternative price sources"]
            )
        
        # Model complexity
        total_pools = self.report['summary'].get('total_pools', 0)
        if total_pools > 100:
            self._add_recommendation(
                f"Consider reducing model complexity ({total_pools} pools may cause solver issues)",
                'medium',
                ["Select top N pools by quality metrics", "Implement pool clustering", "Use hierarchical optimization"]
            )
        
        # Performance optimization
        gas_cost_pct = self.report['summary'].get('gas_cost_as_pct_of_aum', 0)
        if gas_cost_pct > 0.5:
            self._add_recommendation(
                f"High gas costs relative to AUM ({gas_cost_pct:.2f}%)",
                'medium',
                ["Optimize transaction batching", "Consider layer-2 solutions", "Wait for lower gas periods"]
            )
    
    def run_comprehensive_analysis(self) -> Dict:
        """Run comprehensive data quality analysis."""
        logger.info("Starting comprehensive data quality analysis...")
        
        # Analyze all data components
        pools_df = self.analyze_pool_data()
        
        # Get token universe for price analysis
        if not pools_df.empty:
            # Get sample balances to build token universe
            try:
                warm_wallet, current_allocations = fetch_current_balances(self.engine)
                tokens = build_token_universe(pools_df, warm_wallet, current_allocations)
                token_prices = self.analyze_token_prices(tokens)
            except:
                token_prices = self.analyze_token_prices(['USDC', 'USDT', 'ETH', 'WBTC'])
        else:
            token_prices = self.analyze_token_prices(['USDC', 'USDT', 'ETH', 'WBTC'])
        
        eth_price, base_fee_transfer_gwei, base_fee_swap_gwei, priority_fee_gwei, min_gas_units = self.analyze_gas_fee_data()
        warm_wallet, current_allocations = self.analyze_balance_data()
        alloc_params = self.analyze_allocation_parameters()
        
        # Analyze model feasibility if we have sufficient data
        if not pools_df.empty and token_prices:
            self.analyze_model_feasibility(pools_df, token_prices, alloc_params, warm_wallet, current_allocations)
        
        # Analyze data freshness
        self.analyze_data_freshness()
        
        # Generate recommendations
        self.generate_recommendations()
        
        # Calculate overall quality score
        self._calculate_quality_score()
        
        logger.info("Comprehensive data quality analysis complete")
        return self.report
    
    def _calculate_quality_score(self):
        """Calculate an overall data quality score."""
        score = 100.0
        
        # Deduct points for critical issues
        for category, items in self.report['abnormal_values'].items():
            for item in items:
                if item['severity'] == 'critical':
                    score -= 20
                elif item['severity'] == 'warning':
                    score -= 5
        
        # Deduct points for fallback values
        total_fallbacks = sum(len(items) for items in self.report['fallback_values'].values())
        score -= min(total_fallbacks * 2, 20)
        
        # Deduct points for low coverage
        price_coverage = self.report['summary'].get('price_coverage_pct', 100)
        if price_coverage < 100:
            score -= (100 - price_coverage) * 0.3
        
        # Ensure score doesn't go below 0
        score = max(0, score)
        
        self._add_summary('overall_quality_score', score)
        
        # Add quality assessment
        if score >= 90:
            assessment = "Excellent"
        elif score >= 75:
            assessment = "Good"
        elif score >= 60:
            assessment = "Fair"
        else:
            assessment = "Poor"
        
        self._add_summary('quality_assessment', assessment)
    
    def print_report_summary(self):
        """Print a comprehensive summary of the data quality report."""
        print("\n" + "="*80)
        print("DATA QUALITY REPORT SUMMARY")
        print("="*80)
        
        # Overall assessment
        score = self.report['summary'].get('overall_quality_score', 0)
        assessment = self.report['summary'].get('quality_assessment', 'Unknown')
        print(f"\n🎯 OVERALL QUALITY: {assessment} (Score: {score:.1f}/100)")
        
        # Summary statistics
        print("\n📊 SUMMARY STATISTICS:")
        for key, value in self.report['summary'].items():
            if key not in ['overall_quality_score', 'quality_assessment']:
                if isinstance(value, float):
                    if 'pct' in key or 'rate' in key:
                        print(f"  {key}: {value:.2f}%")
                    elif 'gas_fee_usd' in key.lower() or 'gas_fee' in key.lower():
                        # Special formatting for very small gas fees
                        if value < 0.01:
                            print(f"  {key}: ${value:.6f}")
                        else:
                            print(f"  {key}: ${value:,.2f}")
                    elif 'usd' in key.lower() or 'aum' in key.lower():
                        print(f"  {key}: ${value:,.2f}")
                    else:
                        print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
        
        # Fallback values
        print("\n⚠️  FALLBACK VALUES:")
        total_fallbacks = sum(len(items) for items in self.report['fallback_values'].values())
        print(f"  Total fallback values: {total_fallbacks}")
        for category, items in self.report['fallback_values'].items():
            if items:
                print(f"  {category}: {len(items)} items")
                for item in items[:2]:  # Show first 2
                    print(f"    - {item['item']}: {item['value']}")
                if len(items) > 2:
                    print(f"    ... and {len(items) - 2} more")
        
        # Abnormal values
        print("\n🚨 ABNORMAL VALUES:")
        total_abnormal = sum(len(items) for items in self.report['abnormal_values'].values())
        critical_count = sum(len([i for i in items if i['severity'] == 'critical']) 
                           for items in self.report['abnormal_values'].values())
        print(f"  Total abnormal values: {total_abnormal} ({critical_count} critical)")
        
        for category, items in self.report['abnormal_values'].items():
            if items:
                critical_items = [i for i in items if i['severity'] == 'critical']
                print(f"  {category}: {len(items)} items ({len(critical_items)} critical)")
        
        # Model feasibility
        if 'model_feasibility' in self.report:
            feasibility = self.report['model_feasibility']
            print(f"\n⚖️  MODEL FEASIBILITY:")
            print(f"  Total AUM: ${feasibility.get('total_aum', 0):,.2f}")
            print(f"  Pool capacity: ${feasibility.get('total_pool_capacity', 0):,.2f}")
            print(f"  Gas cost estimate: ${feasibility.get('total_gas_cost', 0):.2f}")
            print(f"  Feasible: {feasibility.get('constraint_feasible', False)}")
        
        # Top recommendations
        print("\n💡 TOP RECOMMENDATIONS:")
        # Sort by priority
        recs = sorted(self.report['recommendations'], 
                     key=lambda x: {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}[x['priority']])
        
        for rec in recs[:5]:  # Show top 5
            print(f"  [{rec['priority'].upper()}] {rec['recommendation']}")
        
        print("\n" + "="*80)
    
    def save_report(self, filename: str = None) -> str:
        """Save the detailed report to a JSON file (DISABLED)."""
        logger.info("Data quality report saving is disabled")
        return None


def generate_data_quality_report() -> Dict:
    """
    Generate a comprehensive data quality report for the optimization process.
    
    Returns:
        Dictionary containing the complete data quality analysis
    """
    reporter = DataQualityReporter()
    report = reporter.run_comprehensive_analysis()
    
    # Print summary to logs
    reporter.print_report_summary()
    
    # Save detailed report (DISABLED)
    # reporter.save_report()
    
    return report


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Generate and display the report
    print("="*80)
    print("ASSET ALLOCATION DATA QUALITY REPORT")
    print("="*80)
    
    report = generate_data_quality_report()
    
    print(f"\n📄 Detailed report saved with timestamp: {report['timestamp']}")
    print("="*80)