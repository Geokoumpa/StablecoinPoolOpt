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
from typing import Dict, List, Tuple, Optional, Any

import sys
import os

from database.repositories.pool_metrics_repository import PoolMetricsRepository
from database.repositories.raw_data_repository import RawDataRepository
from database.repositories.daily_balance_repository import DailyBalanceRepository
from database.repositories.parameter_repository import ParameterRepository

# Configure module logger
logger = logging.getLogger(__name__)

# ============================================================================
# DATA FETCHING FUNCTIONS (adapted to use Repositories)
# ============================================================================

def fetch_pool_data() -> pd.DataFrame:
    """Fetches approved pools with forecasted APY and metadata."""
    metrics_repo = PoolMetricsRepository()
    
    # We want active pools for optimization candidates
    # The original query filtered by current date, not filtered out, and positive forecasts
    # get_pool_candidates_for_optimization does exactly this (excluding current allocations logic which is handled inside optimize_allocations)
    # However, get_pool_candidates_for_optimization also returns pools with current allocations even if filtered out.
    # For data quality report, we probably want to see availability of pools for optimization.
    
    # Let's use get_pool_candidates_for_optimization with empty allocated_pool_ids list to get "clean" candidates
    # But ideally we want ALL potentially valid pools.
    
    rows = metrics_repo.get_pool_candidates_for_optimization(date.today(), [])
    
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        'pool_id', 'symbol', 'chain', 'protocol', 'forecasted_apy', 'forecasted_tvl', 'underlying_tokens'
    ])
    
    logger.info(f"Loaded {len(df)} approved pools")
    return df


def fetch_token_prices(tokens: List[str]) -> Dict[str, float]:
    """Fetches latest closing prices for given tokens."""
    repo = RawDataRepository()
    prices = repo.get_latest_prices(tokens)
    logger.info(f"Loaded prices for {len(prices)} tokens")
    return prices


def fetch_gas_fee_data() -> Tuple[float, float, float, float, float]:
    """
    Fetches forecasted gas fee components and ETH price.
    """
    repo = RawDataRepository()
    prices = repo.get_latest_prices(['ETH'])
    eth_price = prices.get('ETH', 3000.0)
    
    # Gas fee components based on requirements
    base_fee_transfer_gwei = 10.0
    base_fee_swap_gwei = 30.0
    priority_fee_gwei = 10.0
    min_gas_units = 21000
    
    logger.info(f"ETH price: ${eth_price:.2f}")
    logger.info(f"Gas fee components - Base transfer: {base_fee_transfer_gwei} Gwei, Base swap: {base_fee_swap_gwei} Gwei, Priority: {priority_fee_gwei} Gwei, Min gas units: {min_gas_units}")
    
    return eth_price, base_fee_transfer_gwei, base_fee_swap_gwei, priority_fee_gwei, min_gas_units

def calculate_gas_fee_usd(gas_units: float, base_fee_gwei: float, priority_fee_gwei: float, eth_price_usd: float) -> float:
    """Calculate gas fee in USD."""
    total_fee_gwei = gas_units * (base_fee_gwei + priority_fee_gwei)
    gas_fee_usd = total_fee_gwei * 1e-9 * eth_price_usd
    return gas_fee_usd


def calculate_transaction_gas_fees(eth_price_usd: float, base_fee_transfer_gwei: float, 
                                   base_fee_swap_gwei: float, priority_fee_gwei: float, 
                                   min_gas_units: float) -> Dict[str, float]:
    """Calculate gas fees for different transaction types."""
    pool_transaction_gas_fee_usd = calculate_gas_fee_usd(
        min_gas_units, base_fee_transfer_gwei, priority_fee_gwei, eth_price_usd
    )
    
    token_swap_gas_fee_usd = calculate_gas_fee_usd(
        min_gas_units, base_fee_swap_gwei, priority_fee_gwei, eth_price_usd
    )
    
    gas_fees = {
        'allocation': pool_transaction_gas_fee_usd,
        'withdrawal': pool_transaction_gas_fee_usd,
        'conversion': token_swap_gas_fee_usd,
        'transfer': pool_transaction_gas_fee_usd,
        'deposit': pool_transaction_gas_fee_usd
    }
    
    logger.info(f"Transaction gas fees - Pool Allocation/Withdrawal: ${pool_transaction_gas_fee_usd:.6f}, Token Swap/Conversion: ${token_swap_gas_fee_usd:.6f}")
    
    return gas_fees


def fetch_current_balances() -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
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
    
    repo = DailyBalanceRepository()
    
    try:
        balances = repo.get_current_balances(MAIN_ASSET_HOLDING_ADDRESS, date.today())
        
        if not balances:
            logger.info(f"No balance data found for wallet {MAIN_ASSET_HOLDING_ADDRESS} today")
            return {}, {}
            
        logger.info(f"Using today's data for wallet {MAIN_ASSET_HOLDING_ADDRESS}")
        
    except Exception as e:
        logger.error(f"Error fetching balance data: {e}")
        return {}, {}
    
    for row in balances:
        token = row.token_symbol
        
        if row.unallocated_balance and row.unallocated_balance > 0:
            warm_wallet[token] = warm_wallet.get(token, 0) + float(row.unallocated_balance)
        
        if row.allocated_balance and row.allocated_balance > 0 and row.pool_id:
            key = (row.pool_id, token)
            allocations[key] = allocations.get(key, 0) + float(row.allocated_balance)
    
    logger.info(f"Warm wallet: {len(warm_wallet)} tokens, Total allocated positions: {len(allocations)}")
    return warm_wallet, allocations


def fetch_allocation_parameters() -> Dict:
    """Fetches the latest allocation parameters."""
    repo = ParameterRepository()
    latest_params = repo.get_latest_parameters()
    
    if not latest_params:
        logger.warning("No allocation parameters found, using defaults")
        return {
            'max_alloc_percentage': 0.20,
            'conversion_rate': 0.0004,
            'min_transaction_value': 50.0
        }
    
    # Convert object to dict
    params = {
        'max_alloc_percentage': latest_params.max_alloc_percentage,
        'tvl_limit_percentage': latest_params.tvl_limit_percentage,
        'conversion_rate': latest_params.conversion_rate or 0.0004,
        'min_pools': latest_params.min_pools
    }
    
    logger.info(f"Loaded allocation parameters: max_alloc={params.get('max_alloc_percentage')}, tvl_limit={params.get('tvl_limit_percentage')}")
    return params


class DataQualityReporter:
    """
    Comprehensive data quality reporter for asset allocation optimization.
    """
    
    def __init__(self):
        """Initialize the data quality reporter."""
        # Removed self.engine = get_db_connection() as we use repositories now
        
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
            pools_df = fetch_pool_data()
            
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
            
            # No normalized tokens analysis needed - using underlying_tokens directly
            self._add_summary('pools_with_underlying_tokens', len(pools_df))
            
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
                'data_types': pools_df.dtypes.to_dict(), # Note: types might be issues with JSON serialization
                'null_counts': null_counts.to_dict(),
                'extreme_apy_pools': len(high_apy_pools),
                'low_tvl_pools': len(low_tvl_pools)
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
            prices = fetch_token_prices(tokens)
            
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
            eth_price, base_fee_transfer_gwei, base_fee_swap_gwei, priority_fee_gwei, min_gas_units = fetch_gas_fee_data()
            
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
            warm_wallet, allocations = fetch_current_balances()
            
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
        """Analyze allocation parameter quality."""
        logger.info("Analyzing allocation parameters...")
        
        try:
            params = fetch_allocation_parameters()
            
            self._add_summary('max_alloc_percentage', params.get('max_alloc_percentage'))
            self._add_summary('tvl_limit_percentage', params.get('tvl_limit_percentage'))
            self._add_summary('conversion_rate', params.get('conversion_rate'))
            
            # Check constraints
            if params.get('max_alloc_percentage', 0) > self.expected_ranges['max_alloc_pct']['max']:
                 self._add_abnormal_value(
                     'parameters',
                     'max_alloc_percentage',
                     params['max_alloc_percentage'],
                     f"High allocation limit (> {self.expected_ranges['max_alloc_pct']['max']})",
                     'warning'
                 )
            
            self.report['parameter_quality'] = params
            return params
            
        except Exception as e:
            logger.error(f"Error analyzing parameters: {e}")
            return {}

    def generate_report(self) -> Dict:
        """Executes all analyses and generates the final report."""
        logger.info("Generating data quality report...")
        
        try:
             # Run analyses
             pools_df = self.analyze_pool_data()
             
             # Extract tokens from pools for price checking
             tokens = set()
             if not pools_df.empty:
                 for _, row in pools_df.iterrows():
                     ut = row.get('underlying_tokens')
                     if isinstance(ut, list):
                         tokens.update(ut)
                     elif isinstance(ut, str):
                         try:
                             tokens.update(json.loads(ut))
                         except: pass
             
             warm_wallet, allocations = self.analyze_balance_data()
             tokens.update(warm_wallet.keys())
             for _, token in allocations.keys():
                 tokens.add(token)
             
             self.analyze_token_prices(list(tokens))
             self.analyze_gas_fee_data()
             self.analyze_allocation_parameters()
             
             # Calculate overall score
             score = 100
             critical_errors = len([a for cat in self.report['abnormal_values'].values() for a in cat if a['severity'] == 'critical'])
             warnings = len([a for cat in self.report['abnormal_values'].values() for a in cat if a['severity'] == 'warning'])
             
             score -= (critical_errors * 20)
             score -= (warnings * 2)
             score = max(0, score)
             
             self._add_summary('overall_quality_score', score)
             self._add_summary('critical_issues_count', critical_errors)
             self._add_summary('warning_issues_count', warnings)
             
             logger.info(f"Report generated. Quality Score: {score}/100")
             return self.report
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return self.report

def generate_data_quality_report() -> Dict:
    """Convenience function to generate the report."""
    reporter = DataQualityReporter()
    return reporter.generate_report()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    report = generate_data_quality_report()
    print(json.dumps(report, indent=2, default=str))         