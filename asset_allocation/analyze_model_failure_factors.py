"""
Analyze factors that might cause the optimization model to fail when including all filtered pools.

This script investigates:
1. Pool combinations that create infeasible constraints
2. Token mapping issues that break price lookups
3. Extreme APY/TVL values that cause numerical instability
4. Gas fee and transaction cost implications
"""

import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple, Set
import itertools
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


class ModelFailureAnalyzer:
    """Analyzes potential causes of optimization model failures."""
    
    def __init__(self):
        self.engine = get_db_connection()
        self.analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'pool_analysis': {},
            'token_analysis': {},
            'constraint_analysis': {},
            'numerical_stability': {},
            'failure_scenarios': [],
            'recommendations': []
        }
    
    def analyze_pool_combinations(self, pools_df: pd.DataFrame):
        """Analyze how pool combinations affect model feasibility."""
        logger.info("Analyzing pool combinations...")
        
        # Sort pools by APY to identify high-APY outliers
        pools_sorted = pools_df.sort_values('forecasted_apy', ascending=False)
        
        # Identify pools with extreme values
        extreme_apy_pools = pools_sorted[pools_sorted['forecasted_apy'] > 30]
        low_tvl_pools = pools_df[pools_df['forecasted_tvl'] < 1000000]  # < $1M
        
        self.analysis_results['pool_analysis']['extreme_apy_pools'] = {
            'count': len(extreme_apy_pools),
            'pools': extreme_apy_pools[['pool_id', 'symbol', 'forecasted_apy', 'forecasted_tvl']].to_dict('records')
        }
        
        self.analysis_results['pool_analysis']['low_tvl_pools'] = {
            'count': len(low_tvl_pools),
            'pools': low_tvl_pools[['pool_id', 'symbol', 'forecasted_apy', 'forecasted_tvl']].to_dict('records')
        }
        
        # Analyze token distribution across pools
        token_pool_count = defaultdict(int)
        token_apy_ranges = defaultdict(list)
        
        for _, pool in pools_df.iterrows():
            tokens = parse_pool_tokens_with_mapping(pool['symbol'], pool.get('normalized_tokens'))
            for token in tokens:
                token_pool_count[token] += 1
                token_apy_ranges[token].append(pool['forecasted_apy'])
        
        # Identify tokens that appear in many pools (potential constraint conflicts)
        high_frequency_tokens = {t: c for t, c in token_pool_count.items() if c > 10}
        
        self.analysis_results['token_analysis']['token_frequency'] = dict(token_pool_count)
        self.analysis_results['token_analysis']['high_frequency_tokens'] = high_frequency_tokens
        
        # Calculate APY variance for tokens appearing in multiple pools
        token_apy_variance = {}
        for token, apys in token_apy_ranges.items():
            if len(apys) > 1:
                token_apy_variance[token] = {
                    'mean': np.mean(apys),
                    'std': np.std(apys),
                    'min': min(apys),
                    'max': max(apys),
                    'pool_count': len(apys)
                }
        
        # Identify tokens with high APY variance (potential arbitrage opportunities)
        high_variance_tokens = {t: v for t, v in token_apy_variance.items() if v['std'] > 5}
        
        self.analysis_results['token_analysis']['apy_variance'] = token_apy_variance
        self.analysis_results['token_analysis']['high_variance_tokens'] = high_variance_tokens
        
        logger.info(f"Found {len(extreme_apy_pools)} pools with APY > 30%")
        logger.info(f"Found {len(low_tvl_pools)} pools with TVL < $1M")
        logger.info(f"Found {len(high_variance_tokens)} tokens with high APY variance")
    
    def analyze_gas_fee_impact(self, gas_gwei: float, eth_price: float):
        """Analyze how gas fees affect model feasibility with many pools."""
        logger.info("Analyzing gas fee impact...")
        
        gas_fee_usd = gas_gwei * 1e-9 * eth_price
        
        # Calculate transaction costs for different pool counts
        pool_counts = [10, 25, 50, 97]  # Different pool set sizes
        cost_analysis = {}
        
        for pool_count in pool_counts:
            # Assume average of 2 transactions per pool (withdraw + allocate)
            transactions = pool_count * 2
            total_gas_cost = transactions * gas_fee_usd
            
            # Calculate minimum AUM needed to justify gas costs
            # Assuming 1% daily yield, gas cost should be < 10% of daily yield
            min_aum_for_gas = total_gas_cost * 10 / 0.01
            
            cost_analysis[pool_count] = {
                'transactions': transactions,
                'total_gas_cost_usd': total_gas_cost,
                'min_aum_for_gas_usd': min_aum_for_gas,
                'gas_as_pct_of_10k': (total_gas_cost / 10000) * 100
            }
        
        self.analysis_results['constraint_analysis']['gas_costs'] = cost_analysis
        
        # Check if current gas price is abnormal
        if gas_gwei < 1:
            self.analysis_results['failure_scenarios'].append({
                'scenario': 'extremely_low_gas',
                'description': f"Gas price {gas_gwei} Gwei is unusually low, may indicate data issues",
                'impact': 'medium'
            })
        
        logger.info(f"Gas cost for 97 pools: ${cost_analysis[97]['total_gas_cost_usd']:.6f}")
        logger.info(f"Minimum AUM to justify gas costs: ${cost_analysis[97]['min_aum_for_gas_usd']:,.2f}")
    
    def analyze_numerical_stability(self, pools_df: pd.DataFrame, token_prices: Dict[str, float]):
        """Analyze numerical stability factors that could cause solver failures."""
        logger.info("Analyzing numerical stability...")
        
        # Check for extreme value ratios that can cause numerical issues
        apy_ratio = pools_df['forecasted_apy'].max() / pools_df['forecasted_apy'].min()
        tvl_ratio = pools_df['forecasted_tvl'].max() / pools_df['forecasted_tvl'].min()
        
        self.analysis_results['numerical_stability']['apy_ratio'] = apy_ratio
        self.analysis_results['numerical_stability']['tvl_ratio'] = tvl_ratio
        
        # Check for price dependencies
        tokens_without_prices = []
        for _, pool in pools_df.iterrows():
            tokens = parse_pool_tokens_with_mapping(pool['symbol'], pool.get('normalized_tokens'))
            for token in tokens:
                if token not in token_prices:
                    tokens_without_prices.append(token)
        
        unique_missing_tokens = list(set(tokens_without_prices))
        
        self.analysis_results['numerical_stability']['tokens_without_prices'] = unique_missing_tokens
        self.analysis_results['numerical_stability']['missing_token_count'] = len(unique_missing_tokens)
        
        # Identify potential numerical issues
        if apy_ratio > 1000:
            self.analysis_results['failure_scenarios'].append({
                'scenario': 'high_apy_variance',
                'description': f"APY ratio {apy_ratio:.1f} may cause numerical instability",
                'impact': 'high'
            })
        
        if tvl_ratio > 10000:
            self.analysis_results['failure_scenarios'].append({
                'scenario': 'high_tvl_variance',
                'description': f"TVL ratio {tvl_ratio:.1f} may cause scaling issues",
                'impact': 'medium'
            })
        
        if len(unique_missing_tokens) > 0:
            self.analysis_results['failure_scenarios'].append({
                'scenario': 'missing_token_prices',
                'description': f"{len(unique_missing_tokens)} tokens lack price data",
                'impact': 'high'
            })
        
        # Check pool token count complexity
        pool_token_counts = []
        for _, pool in pools_df.iterrows():
            tokens = parse_pool_tokens_with_mapping(pool['symbol'], pool.get('normalized_tokens'))
            pool_token_counts.append(len(tokens))
        
        max_tokens_per_pool = max(pool_token_counts)
        avg_tokens_per_pool = np.mean(pool_token_counts)
        
        self.analysis_results['numerical_stability']['max_tokens_per_pool'] = max_tokens_per_pool
        self.analysis_results['numerical_stability']['avg_tokens_per_pool'] = avg_tokens_per_pool
        
        if max_tokens_per_pool > 4:
            self.analysis_results['failure_scenarios'].append({
                'scenario': 'complex_pools',
                'description': f"Pools with up to {max_tokens_per_pool} tokens increase model complexity",
                'impact': 'medium'
            })
        
        logger.info(f"APY ratio: {apy_ratio:.1f}, TVL ratio: {tvl_ratio:.1f}")
        logger.info(f"Tokens without prices: {len(unique_missing_tokens)}")
        logger.info(f"Max tokens per pool: {max_tokens_per_pool}")
    
    def analyze_constraint_feasibility(self, pools_df: pd.DataFrame, alloc_params: Dict):
        """Analyze if constraints might be infeasible with all pools."""
        logger.info("Analyzing constraint feasibility...")
        
        max_alloc_pct = alloc_params.get('max_alloc_percentage', 0.20)
        tvl_limit_pct = alloc_params.get('tvl_limit_percentage', 0.05)
        
        # Check TVL constraints
        tvl_constraints = []
        for _, pool in pools_df.iterrows():
            max_allocation = pool['forecasted_tvl'] * tvl_limit_pct
            tvl_constraints.append(max_allocation)
        
        total_tvl_capacity = sum(tvl_constraints)
        avg_pool_capacity = np.mean(tvl_constraints)
        
        self.analysis_results['constraint_analysis']['tvl_constraints'] = {
            'total_capacity': total_tvl_capacity,
            'avg_capacity_per_pool': avg_pool_capacity,
            'min_capacity': min(tvl_constraints),
            'max_capacity': max(tvl_constraints)
        }
        
        # Check if max allocation percentage is too restrictive
        if max_alloc_pct < 0.1 and len(pools_df) > 50:
            self.analysis_results['failure_scenarios'].append({
                'scenario': 'restrictive_max_allocation',
                'description': f"Max allocation {max_alloc_pct*100:.1f}% may be too low for {len(pools_df)} pools",
                'impact': 'high'
            })
        
        # Check if TVL constraints are too tight
        if avg_pool_capacity < 1000:  # Less than $1000 per pool
            self.analysis_results['failure_scenarios'].append({
                'scenario': 'tight_tvl_constraints',
                'description': f"Average TVL capacity ${avg_pool_capacity:,.0f} may be too restrictive",
                'impact': 'medium'
            })
        
        logger.info(f"Total TVL capacity across all pools: ${total_tvl_capacity:,.0f}")
        logger.info(f"Average capacity per pool: ${avg_pool_capacity:,.0f}")
    
    def generate_recommendations(self):
        """Generate recommendations based on analysis."""
        logger.info("Generating recommendations...")
        
        # High-impact recommendations
        critical_scenarios = [s for s in self.analysis_results['failure_scenarios'] if s['impact'] == 'high']
        
        if critical_scenarios:
            self.analysis_results['recommendations'].append({
                'priority': 'critical',
                'action': 'Address critical issues before running full optimization',
                'details': [s['description'] for s in critical_scenarios]
            })
        
        # Data quality recommendations
        if self.analysis_results['numerical_stability']['missing_token_count'] > 0:
            self.analysis_results['recommendations'].append({
                'priority': 'high',
                'action': 'Improve token price coverage',
                'details': [
                    f"Add price data for {self.analysis_results['numerical_stability']['missing_token_count']} tokens",
                    "Consider using fallback prices only for stablecoins"
                ]
            })
        
        # Pool filtering recommendations
        extreme_apy_count = self.analysis_results['pool_analysis']['extreme_apy_pools']['count']
        if extreme_apy_count > 0:
            self.analysis_results['recommendations'].append({
                'priority': 'medium',
                'action': 'Review extreme APY pools',
                'details': [
                    f"Verify {extreme_apy_count} pools with APY > 30%",
                    "Consider capping APY at reasonable levels (e.g., 50%)"
                ]
            })
        
        # Model complexity recommendations
        if len(self.analysis_results['pool_analysis']) > 80:
            self.analysis_results['recommendations'].append({
                'priority': 'medium',
                'action': 'Consider pool subset selection',
                'details': [
                    f"Current pool count ({len(self.analysis_results['pool_analysis'])}) may cause solver issues",
                    "Select top N pools by APY/TVL ratio or use clustering"
                ]
            })
        
        # Constraint tuning recommendations
        self.analysis_results['recommendations'].append({
            'priority': 'low',
            'action': 'Optimize solver parameters',
            'details': [
                "Consider scaling numerical values for better stability",
                "Adjust solver tolerance settings if needed"
            ]
        })
    
    def run_analysis(self):
        """Run complete analysis of model failure factors."""
        logger.info("Starting model failure analysis...")
        
        # Fetch all data
        pools_df = fetch_pool_data(self.engine)
        warm_wallet, allocations = fetch_current_balances(self.engine)
        tokens = build_token_universe(pools_df, warm_wallet, allocations)
        token_prices = fetch_token_prices(self.engine, tokens)
        gas_gwei, eth_price = fetch_gas_fee_data(self.engine)
        alloc_params = fetch_allocation_parameters(self.engine)
        
        # Run all analyses
        self.analyze_pool_combinations(pools_df)
        self.analyze_gas_fee_impact(gas_gwei, eth_price)
        self.analyze_numerical_stability(pools_df, token_prices)
        self.analyze_constraint_feasibility(pools_df, alloc_params)
        
        # Generate recommendations
        self.generate_recommendations()
        
        # Save results
        results_file = f"model_failure_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        # Print summary
        self.print_summary()
        
        logger.info(f"Analysis results saved to {results_file}")
        return results_file
    
    def print_summary(self):
        """Print analysis summary."""
        print("\n" + "="*80)
        print("MODEL FAILURE ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\n📊 POOL ANALYSIS:")
        print(f"  Total pools: {len(self.analysis_results.get('pool_analysis', {}).get('extreme_apy_pools', {}).get('pools', []))}")
        print(f"  Extreme APY pools (>30%): {self.analysis_results['pool_analysis']['extreme_apy_pools']['count']}")
        print(f"  Low TVL pools (<$1M): {self.analysis_results['pool_analysis']['low_tvl_pools']['count']}")
        
        print(f"\n🔤 TOKEN ANALYSIS:")
        print(f"  High-frequency tokens: {len(self.analysis_results['token_analysis']['high_frequency_tokens'])}")
        print(f"  High variance tokens: {len(self.analysis_results['token_analysis']['high_variance_tokens'])}")
        
        print(f"\n⚖️  CONSTRAINT ANALYSIS:")
        gas_costs = self.analysis_results['constraint_analysis']['gas_costs']
        print(f"  Gas cost for all pools: ${gas_costs[97]['total_gas_cost_usd']:.6f}")
        print(f"  Min AUM for gas feasibility: ${gas_costs[97]['min_aum_for_gas_usd']:,.2f}")
        
        print(f"\n🔢 NUMERICAL STABILITY:")
        print(f"  APY ratio (max/min): {self.analysis_results['numerical_stability']['apy_ratio']:.1f}")
        print(f"  TVL ratio (max/min): {self.analysis_results['numerical_stability']['tvl_ratio']:.1f}")
        print(f"  Tokens without prices: {self.analysis_results['numerical_stability']['missing_token_count']}")
        
        print(f"\n🚨 FAILURE SCENARIOS:")
        for scenario in self.analysis_results['failure_scenarios']:
            print(f"  [{scenario['impact'].upper()}] {scenario['scenario']}: {scenario['description']}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in self.analysis_results['recommendations']:
            print(f"  [{rec['priority'].upper()}] {rec['action']}")
            for detail in rec.get('details', []):
                print(f"    - {detail}")
        
        print("\n" + "="*80)


def main():
    """Main function to run model failure analysis."""
    print("="*80)
    print("ASSET ALLOCATION MODEL FAILURE ANALYSIS")
    print("="*80)
    print(f"Started at: {datetime.now().isoformat()}")
    
    analyzer = ModelFailureAnalyzer()
    results_file = analyzer.run_analysis()
    
    print(f"\n📄 Detailed analysis saved to: {results_file}")
    print("="*80)
    
    return results_file


if __name__ == "__main__":
    main()