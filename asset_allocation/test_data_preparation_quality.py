"""
Comprehensive Data Quality Test for Asset Allocation Optimization

This test analyzes the data preparation phase of optimize_allocations.py to:
1. Identify real vs. mocked/fallback data
2. Detect abnormal value ranges that might cause model failures
3. Provide detailed reporting on data quality issues
"""

import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime, date, timezone
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_utils import get_db_connection
from asset_allocation.optimize_allocations import (
    fetch_pool_data,
    fetch_token_prices,
    fetch_gas_fee_data,
    fetch_current_balances,
    fetch_allocation_parameters,
    parse_pool_tokens_with_mapping,
    build_token_universe,
    calculate_aum
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataQualityReport:
    """Collects and reports data quality metrics."""
    
    def __init__(self):
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'pool_data': {},
            'token_prices': {},
            'gas_fees': {},
            'balances': {},
            'allocation_parameters': {},
            'abnormal_values': {},
            'fallback_values': {},
            'recommendations': []
        }
    
    def add_summary(self, key: str, value: Any):
        self.report['summary'][key] = value
    
    def add_fallback(self, category: str, item: str, value: Any, expected_source: str):
        if category not in self.report['fallback_values']:
            self.report['fallback_values'][category] = []
        self.report['fallback_values'][category].append({
            'item': item,
            'value': value,
            'expected_source': expected_source,
            'is_fallback': True
        })
    
    def add_abnormal_value(self, category: str, item: str, value: Any, 
                          reason: str, severity: str = 'warning'):
        if category not in self.report['abnormal_values']:
            self.report['abnormal_values'][category] = []
        self.report['abnormal_values'][category].append({
            'item': item,
            'value': value,
            'reason': reason,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        })
    
    def add_recommendation(self, recommendation: str, priority: str = 'medium'):
        self.report['recommendations'].append({
            'recommendation': recommendation,
            'priority': priority
        })
    
    def save_report(self, filename: str = None):
        if filename is None:
            filename = f"data_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.report, f, indent=2, default=str)
        
        logger.info(f"Data quality report saved to {filename}")
        return filename
    
    def print_summary(self):
        """Print a concise summary of key findings."""
        print("\n" + "="*80)
        print("DATA QUALITY REPORT SUMMARY")
        print("="*80)
        
        # Summary stats
        print("\n📊 SUMMARY STATISTICS:")
        for key, value in self.report['summary'].items():
            print(f"  {key}: {value}")
        
        # Fallback values
        print("\n⚠️  FALLBACK VALUES DETECTED:")
        total_fallbacks = sum(len(items) for items in self.report['fallback_values'].values())
        print(f"  Total fallback values: {total_fallbacks}")
        for category, items in self.report['fallback_values'].items():
            if items:
                print(f"  {category}: {len(items)} items")
                for item in items[:3]:  # Show first 3
                    print(f"    - {item['item']}: {item['value']} (expected from {item['expected_source']})")
                if len(items) > 3:
                    print(f"    ... and {len(items) - 3} more")
        
        # Abnormal values
        print("\n🚨 ABNORMAL VALUES:")
        total_abnormal = sum(len(items) for items in self.report['abnormal_values'].values())
        print(f"  Total abnormal values: {total_abnormal}")
        for category, items in self.report['abnormal_values'].items():
            if items:
                high_severity = [i for i in items if i['severity'] == 'critical']
                print(f"  {category}: {len(items)} items ({len(high_severity)} critical)")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        for rec in self.report['recommendations'][:5]:  # Show top 5
            print(f"  [{rec['priority'].upper()}] {rec['recommendation']}")
        
        print("\n" + "="*80)


class DataQualityTester:
    """Tests data quality for asset allocation optimization."""
    
    def __init__(self):
        self.engine = get_db_connection()
        self.report = DataQualityReport()
        
        # Define expected value ranges
        self.expected_ranges = {
            'apy': {'min': 0, 'max': 50, 'unit': '%'},  # 0-50% APY
            'tvl': {'min': 1000, 'max': 1e12, 'unit': 'USD'},  # $1K - $1T TVL
            'gas_gwei': {'min': 1, 'max': 1000, 'unit': 'Gwei'},  # 1-1000 Gwei
            'eth_price': {'min': 100, 'max': 10000, 'unit': 'USD'},  # $100-$10K ETH
            'token_price': {'min': 0.0001, 'max': 100000, 'unit': 'USD'},  # Wide range for tokens
            'balance': {'min': 0, 'max': 1e12, 'unit': 'tokens'}  # 0 - 1T tokens
        }
        
        # Known fallback values
        self.fallback_values = {
            'gas_gwei': 50.0,
            'eth_price': 3000.0,
            'token_price': 1.0,  # Default for stablecoins
            'max_alloc_percentage': 0.20,
            'conversion_rate': 0.0004,
            'min_transaction_value': 50.0
        }
    
    def test_pool_data(self) -> pd.DataFrame:
        """Test pool data quality."""
        logger.info("Testing pool data quality...")
        
        try:
            pools_df = fetch_pool_data(self.engine)
            
            # Basic stats
            self.report.add_summary('total_pools', len(pools_df))
            self.report.add_summary('unique_protocols', pools_df['protocol'].nunique())
            self.report.add_summary('unique_chains', pools_df['chain'].nunique())
            
            # Check for abnormal APY values
            apy_stats = pools_df['forecasted_apy'].describe()
            self.report.add_summary('apy_min', apy_stats['min'])
            self.report.add_summary('apy_max', apy_stats['max'])
            self.report.add_summary('apy_mean', apy_stats['mean'])
            
            # Flag abnormal APYs
            high_apy_pools = pools_df[pools_df['forecasted_apy'] > self.expected_ranges['apy']['max']]
            for _, pool in high_apy_pools.iterrows():
                self.report.add_abnormal_value(
                    'pool_apy',
                    f"{pool['pool_id']} ({pool['symbol']})",
                    pool['forecasted_apy'],
                    f"APY exceeds {self.expected_ranges['apy']['max']}%",
                    'warning' if pool['forecasted_apy'] < 100 else 'critical'
                )
            
            # Check for abnormal TVL values
            tvl_stats = pools_df['forecasted_tvl'].describe()
            self.report.add_summary('tvl_min', tvl_stats['min'])
            self.report.add_summary('tvl_max', tvl_stats['max'])
            self.report.add_summary('tvl_mean', tvl_stats['mean'])
            
            # Flag abnormal TVLs
            low_tvl_pools = pools_df[pools_df['forecasted_tvl'] < self.expected_ranges['tvl']['min']]
            for _, pool in low_tvl_pools.iterrows():
                self.report.add_abnormal_value(
                    'pool_tvl',
                    f"{pool['pool_id']} ({pool['symbol']})",
                    pool['forecasted_tvl'],
                    f"TVL below ${self.expected_ranges['tvl']['min']:,.0f}",
                    'warning'
                )
            
            # Check normalized_tokens
            pools_with_mappings = pools_df[pools_df['normalized_tokens'].notna()]
            self.report.add_summary('pools_with_token_mappings', len(pools_with_mappings))
            
            # Analyze token mappings
            mapping_issues = 0
            for _, pool in pools_with_mappings.iterrows():
                try:
                    mappings = json.loads(pool['normalized_tokens'])
                    original_tokens = parse_pool_tokens_with_mapping(pool['symbol'], None)
                    mapped_tokens = parse_pool_tokens_with_mapping(pool['symbol'], pool['normalized_tokens'])
                    
                    if original_tokens != mapped_tokens:
                        mapping_issues += 1
                        
                        # Check if mapped tokens exist in price data
                        for token in mapped_tokens:
                            if token not in original_tokens:
                                # This is a mapped token, check if we have price data
                                pass
                                
                except (json.JSONDecodeError, TypeError):
                    self.report.add_abnormal_value(
                        'token_mapping',
                        f"{pool['pool_id']} ({pool['symbol']})",
                        pool['normalized_tokens'],
                        "Invalid JSON in normalized_tokens",
                        'critical'
                    )
            
            self.report.add_summary('pools_with_token_changes', mapping_issues)
            
            self.report.report['pool_data'] = {
                'total_records': len(pools_df),
                'columns': list(pools_df.columns),
                'sample_data': pools_df.head(3).to_dict('records'),
                'null_counts': pools_df.isnull().to_dict(),
                'data_types': pools_df.dtypes.to_dict()
            }
            
            return pools_df
            
        except Exception as e:
            logger.error(f"Error testing pool data: {e}")
            self.report.add_abnormal_value(
                'pool_data',
                'fetch_error',
                str(e),
                "Failed to fetch pool data",
                'critical'
            )
            return pd.DataFrame()
    
    def test_token_prices(self, tokens: List[str]) -> Dict[str, float]:
        """Test token price data quality."""
        logger.info("Testing token price data quality...")
        
        try:
            prices = fetch_token_prices(self.engine, tokens)
            
            self.report.add_summary('total_tokens_requested', len(tokens))
            self.report.add_summary('tokens_with_prices', len(prices))
            self.report.add_summary('price_coverage', f"{len(prices)/len(tokens)*100:.1f}%" if tokens else "0%")
            
            # Check for fallback prices
            tokens_without_prices = set(tokens) - set(prices.keys())
            for token in tokens_without_prices:
                self.report.add_fallback(
                    'token_prices',
                    token,
                    self.fallback_values['token_price'],
                    'CoinMarketCap OHLCV data'
                )
            
            # Check for abnormal prices
            for token, price in prices.items():
                if price < self.expected_ranges['token_price']['min']:
                    self.report.add_abnormal_value(
                        'token_price',
                        token,
                        price,
                        f"Price below ${self.expected_ranges['token_price']['min']}",
                        'warning'
                    )
                elif price > self.expected_ranges['token_price']['max']:
                    self.report.add_abnormal_value(
                        'token_price',
                        token,
                        price,
                        f"Price above ${self.expected_ranges['token_price']['max']}",
                        'warning'
                    )
            
            # Store price statistics
            if prices:
                price_values = list(prices.values())
                self.report.add_summary('price_min', min(price_values))
                self.report.add_summary('price_max', max(price_values))
                self.report.add_summary('price_mean', np.mean(price_values))
            
            self.report.report['token_prices'] = {
                'fetched_prices': prices,
                'missing_tokens': list(tokens_without_prices),
                'price_stats': {
                    'count': len(prices),
                    'min': min(prices.values()) if prices else None,
                    'max': max(prices.values()) if prices else None,
                    'mean': np.mean(list(prices.values())) if prices else None
                }
            }
            
            return prices
            
        except Exception as e:
            logger.error(f"Error testing token prices: {e}")
            self.report.add_abnormal_value(
                'token_prices',
                'fetch_error',
                str(e),
                "Failed to fetch token prices",
                'critical'
            )
            return {}
    
    def test_gas_fee_data(self) -> Tuple[float, float]:
        """Test gas fee data quality."""
        logger.info("Testing gas fee data quality...")
        
        try:
            gas_gwei, eth_price = fetch_gas_fee_data(self.engine)
            
            # Check for fallback values
            if gas_gwei == self.fallback_values['gas_gwei']:
                self.report.add_fallback(
                    'gas_fees',
                    'gas_gwei',
                    gas_gwei,
                    'gas_fees_daily table'
                )
            
            if eth_price == self.fallback_values['eth_price']:
                self.report.add_fallback(
                    'gas_fees',
                    'eth_price',
                    eth_price,
                    'CoinMarketCap OHLCV data'
                )
            
            # Check for abnormal values
            if gas_gwei < self.expected_ranges['gas_gwei']['min']:
                self.report.add_abnormal_value(
                    'gas_fees',
                    'gas_gwei',
                    gas_gwei,
                    f"Gas price below {self.expected_ranges['gas_gwei']['min']} Gwei",
                    'warning'
                )
            elif gas_gwei > self.expected_ranges['gas_gwei']['max']:
                self.report.add_abnormal_value(
                    'gas_fees',
                    'gas_gwei',
                    gas_gwei,
                    f"Gas price above {self.expected_ranges['gas_gwei']['max']} Gwei",
                    'warning'
                )
            
            if eth_price < self.expected_ranges['eth_price']['min']:
                self.report.add_abnormal_value(
                    'gas_fees',
                    'eth_price',
                    eth_price,
                    f"ETH price below ${self.expected_ranges['eth_price']['min']}",
                    'warning'
                )
            elif eth_price > self.expected_ranges['eth_price']['max']:
                self.report.add_abnormal_value(
                    'gas_fees',
                    'eth_price',
                    eth_price,
                    f"ETH price above ${self.expected_ranges['eth_price']['max']}",
                    'warning'
                )
            
            self.report.add_summary('gas_gwei', gas_gwei)
            self.report.add_summary('eth_price_usd', eth_price)
            self.report.add_summary('gas_fee_usd', gas_gwei * 1e-9 * eth_price)
            
            self.report.report['gas_fees'] = {
                'gas_gwei': gas_gwei,
                'eth_price_usd': eth_price,
                'gas_fee_usd': gas_gwei * 1e-9 * eth_price,
                'is_gas_fallback': gas_gwei == self.fallback_values['gas_gwei'],
                'is_eth_fallback': eth_price == self.fallback_values['eth_price']
            }
            
            return gas_gwei, eth_price
            
        except Exception as e:
            logger.error(f"Error testing gas fee data: {e}")
            self.report.add_abnormal_value(
                'gas_fees',
                'fetch_error',
                str(e),
                "Failed to fetch gas fee data",
                'critical'
            )
            return self.fallback_values['gas_gwei'], self.fallback_values['eth_price']
    
    def test_balance_data(self) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
        """Test balance data quality."""
        logger.info("Testing balance data quality...")
        
        try:
            warm_wallet, allocations = fetch_current_balances(self.engine)
            
            # Check if we have any balance data
            if not warm_wallet and not allocations:
                self.report.add_abnormal_value(
                    'balances',
                    'no_data',
                    'None',
                    "No balance data found for configured wallet",
                    'critical'
                )
                self.report.add_recommendation(
                    "Check MAIN_ASSET_HOLDING_ADDRESS configuration and daily_balances table",
                    'high'
                )
            
            # Analyze warm wallet balances
            total_warm_value = 0
            for token, amount in warm_wallet.items():
                if amount < 0:
                    self.report.add_abnormal_value(
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
                    self.report.add_abnormal_value(
                        'balances',
                        f'allocated_{pool_id}_{token}',
                        amount,
                        "Negative allocated balance",
                        'critical'
                    )
                total_allocated_value += amount
            
            self.report.add_summary('warm_wallet_tokens', len(warm_wallet))
            self.report.add_summary('allocated_positions', len(allocations))
            self.report.add_summary('total_warm_balance', total_warm_value)
            self.report.add_summary('total_allocated_balance', total_allocated_value)
            
            self.report.report['balances'] = {
                'warm_wallet': warm_wallet,
                'allocations': {f"{k[0]}_{k[1]}": v for k, v in allocations.items()},
                'warm_wallet_total': total_warm_value,
                'allocated_total': total_allocated_value,
                'has_data': bool(warm_wallet or allocations)
            }
            
            return warm_wallet, allocations
            
        except Exception as e:
            logger.error(f"Error testing balance data: {e}")
            self.report.add_abnormal_value(
                'balances',
                'fetch_error',
                str(e),
                "Failed to fetch balance data",
                'critical'
            )
            return {}, {}
    
    def test_allocation_parameters(self) -> Dict:
        """Test allocation parameters quality."""
        logger.info("Testing allocation parameters...")
        
        try:
            params = fetch_allocation_parameters(self.engine)
            
            # Check for default/fallback values
            # Note: These values are legitimately stored in allocation_parameters table,
            # not fallbacks. Only flag if they're missing entirely.
            if 'max_alloc_percentage' not in params:
                self.report.add_fallback(
                    'allocation_parameters',
                    'max_alloc_percentage',
                    self.fallback_values['max_alloc_percentage'],
                    'allocation_parameters table'
                )
            
            if 'conversion_rate' not in params:
                self.report.add_fallback(
                    'allocation_parameters',
                    'conversion_rate',
                    self.fallback_values['conversion_rate'],
                    'allocation_parameters table'
                )
            
            if 'min_transaction_value' not in params:
                self.report.add_fallback(
                    'allocation_parameters',
                    'min_transaction_value',
                    self.fallback_values['min_transaction_value'],
                    'allocation_parameters table'
                )
            
            # Check parameter ranges
            if params.get('max_alloc_percentage', 0) > 0.5:
                self.report.add_abnormal_value(
                    'allocation_parameters',
                    'max_alloc_percentage',
                    params['max_alloc_percentage'],
                    "Maximum allocation percentage exceeds 50%",
                    'warning'
                )
            
            if params.get('conversion_rate', 0) > 0.01:
                self.report.add_abnormal_value(
                    'allocation_parameters',
                    'conversion_rate',
                    params['conversion_rate'],
                    "Conversion rate exceeds 1%",
                    'warning'
                )
            
            self.report.add_summary('allocation_params_count', len(params))
            self.report.report['allocation_parameters'] = params
            
            return params
            
        except Exception as e:
            logger.error(f"Error testing allocation parameters: {e}")
            self.report.add_abnormal_value(
                'allocation_parameters',
                'fetch_error',
                str(e),
                "Failed to fetch allocation parameters",
                'critical'
            )
            return {}
    
    def test_aum_calculation(self, warm_wallet: Dict[str, float], 
                           allocations: Dict[Tuple[str, str], float],
                           token_prices: Dict[str, float]) -> float:
        """Test AUM calculation and identify price dependencies."""
        logger.info("Testing AUM calculation...")
        
        try:
            aum = calculate_aum(warm_wallet, allocations, token_prices)
            
            # Check how much AUM depends on fallback prices
            fallback_dependent_aum = 0
            total_aum = 0
            
            # Warm wallet
            for token, amount in warm_wallet.items():
                price = token_prices.get(token, 1.0)
                value = amount * price
                total_aum += value
                if token not in token_prices:
                    fallback_dependent_aum += value
            
            # Allocations
            for (pool_id, token), amount in allocations.items():
                price = token_prices.get(token, 1.0)
                value = amount * price
                total_aum += value
                if token not in token_prices:
                    fallback_dependent_aum += value
            
            if total_aum > 0:
                fallback_percentage = (fallback_dependent_aum / total_aum) * 100
                self.report.add_summary('aum_total', total_aum)
                self.report.add_summary('aum_fallback_dependent', fallback_percentage)
                
                if fallback_percentage > 20:
                    self.report.add_abnormal_value(
                        'aum_calculation',
                        'fallback_dependency',
                        f"{fallback_percentage:.1f}%",
                        "High percentage of AUM depends on fallback prices",
                        'warning'
                    )
                    self.report.add_recommendation(
                        f"Improve price data coverage - {fallback_percentage:.1f}% of AUM uses fallback prices",
                        'high'
                    )
            
            return aum
            
        except Exception as e:
            logger.error(f"Error testing AUM calculation: {e}")
            self.report.add_abnormal_value(
                'aum_calculation',
                'calculation_error',
                str(e),
                "Failed to calculate AUM",
                'critical'
            )
            return 0
    
    def run_comprehensive_test(self):
        """Run all data quality tests."""
        logger.info("Starting comprehensive data quality test...")
        
        # Test 1: Pool Data
        pools_df = self.test_pool_data()
        
        # Test 2: Build token universe
        warm_wallet, allocations = self.test_balance_data()
        tokens = build_token_universe(pools_df, warm_wallet, allocations)
        
        # Test 3: Token Prices
        token_prices = self.test_token_prices(tokens)
        
        # Test 4: Gas Fees
        gas_gwei, eth_price = self.test_gas_fee_data()
        
        # Test 5: Allocation Parameters
        alloc_params = self.test_allocation_parameters()
        
        # Test 6: AUM Calculation
        aum = self.test_aum_calculation(warm_wallet, allocations, token_prices)
        
        # Generate recommendations based on findings
        self._generate_recommendations()
        
        # Print summary
        self.report.print_summary()
        
        # Save detailed report
        report_file = self.report.save_report()
        
        return report_file
    
    def _generate_recommendations(self):
        """Generate recommendations based on test results."""
        total_fallbacks = sum(len(items) for items in self.report.report['fallback_values'].values())
        total_abnormal = sum(len(items) for items in self.report.report['abnormal_values'].values())
        
        if total_fallbacks > 5:
            self.report.add_recommendation(
                f"High number of fallback values ({total_fallbacks}) - investigate data pipeline issues",
                'high'
            )
        
        if total_abnormal > 10:
            self.report.add_recommendation(
                f"Many abnormal values detected ({total_abnormal}) - review data quality controls",
                'medium'
            )
        
        # Check for critical issues
        critical_issues = 0
        for category, items in self.report.report['abnormal_values'].items():
            critical_issues += len([i for i in items if i['severity'] == 'critical'])
        
        if critical_issues > 0:
            self.report.add_recommendation(
                f"{critical_issues} critical issues found - address before running optimization",
                'high'
            )


def main():
    """Main function to run the data quality test."""
    print("="*80)
    print("ASSET ALLOCATION DATA QUALITY TEST")
    print("="*80)
    print(f"Started at: {datetime.now().isoformat()}")
    
    tester = DataQualityTester()
    report_file = tester.run_comprehensive_test()
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    print("="*80)
    
    return report_file


if __name__ == "__main__":
    main()