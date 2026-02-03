"""
Backtesting Script for Asset Allocation Optimization

This script simulates the optimization algorithm over historical data to evaluate performance.
It fetches pool metrics, token prices, and other data from the production database for each day
in the specified period, runs the optimization, and tracks the portfolio performance.

Key features:
- Uses historical forecasted APY and TVL from pool_metrics
- Simulates with starting AUM (default: 1M USDC)
- Tracks daily portfolio value and allocations
- Captures optimization failures with detailed reasons
- Generates comprehensive performance report
"""

import logging
import argparse
import pandas as pd
import numpy as np
import json
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.repositories.pool_metrics_repository import PoolMetricsRepository
from database.repositories.raw_data_repository import RawDataRepository
from database.repositories.token_repository import TokenRepository
from asset_allocation.optimize_allocations import (
    AllocationOptimizer,
    build_token_universe,
    calculate_aum,
    calculate_gas_fee_usd,
    calculate_transaction_gas_fees,
    fetch_gas_fee_data
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BacktestOptimizer:
    """
    Backtesting engine for the allocation optimization algorithm.
    """

    def __init__(self, start_date: date, end_date: date, initial_aum: float = 1_000_000):
        """
        Initialize backtesting optimizer.

        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            initial_aum: Initial Assets Under Management in USD (default: 1M)
        """
        self.start_date = start_date
        self.end_date = end_date
        self.initial_aum = initial_aum

        # Initialize repositories
        self.metrics_repo = PoolMetricsRepository()
        self.raw_data_repo = RawDataRepository()
        self.token_repo = TokenRepository()

        # Backtracking state
        self.current_allocations = {}  # Dict[(pool_id, token): amount]
        self.warm_wallet = {}  # Dict[token: amount]
        self.current_aum = initial_aum

        # Yield tracking
        self.cumulative_forecasted_yield = 0.0  # Track total forecasted yield
        self.cumulative_actual_yield = 0.0  # Track total actual yield
        self.previous_day_pools_df = None  # Store pool data for yield calculation
        self.previous_day_token_prices = {}  # Store prices for yield calculation

        # Results storage
        self.daily_results = []
        self.failed_days = []

        # Load approved tokens mapping once
        self._load_approved_tokens()

        logger.info(f"Initialized BacktestOptimizer:")
        logger.info(f"  Period: {start_date} to {end_date}")
        logger.info(f"  Initial AUM: ${initial_aum:,.2f}")

    def _load_approved_tokens(self):
        """Load approved tokens mapping for normalization."""
        approved_tokens = self.token_repo.get_approved_tokens()
        self.addr_to_symbol = {}
        for t in approved_tokens:
            if t.token_address:
                self.addr_to_symbol[t.token_address.lower()] = t.token_symbol

    def fetch_pool_data_for_date(self, target_date: date) -> pd.DataFrame:
        """
        Fetch approved pools with forecasted APY and actual APY for backtesting.

        Args:
            target_date: Date to fetch pool data for

        Returns:
            DataFrame with pool data including both forecasted and actual APY
        """
        from sqlalchemy import text

        # Get currently allocated pool IDs
        allocated_pool_ids = list(set([pool_id for (pool_id, _) in self.current_allocations.keys()]))

        # Fetch pool candidates with BOTH forecasted and actual APY
        sql = text("""
            SELECT
                p.pool_id,
                p.symbol,
                p.chain,
                p.protocol,
                COALESCE(pdm.forecasted_apy, 0) as forecasted_apy,
                COALESCE(pdm.forecasted_tvl, 0) as forecasted_tvl,
                COALESCE(pdm.actual_apy, 0) as actual_apy,
                p.underlying_tokens
            FROM pools p
            LEFT JOIN pool_daily_metrics pdm ON p.pool_id = pdm.pool_id AND pdm.date = :date
            WHERE (
                pdm.pool_id IS NOT NULL
                AND pdm.is_filtered_out = FALSE
                AND pdm.forecasted_apy > 0
                AND pdm.forecasted_tvl > 0
                AND p.is_active = TRUE
            )
            OR (
                p.pool_id = ANY(:allocated_ids)
            )
        """)

        ids_param = allocated_pool_ids if allocated_pool_ids else []

        with self.metrics_repo.session() as session:
            rows = session.execute(sql, {'date': target_date, 'allocated_ids': ids_param}).fetchall()

        if not rows:
            logger.warning(f"No pools found for {target_date} (Allocated pools: {len(allocated_pool_ids)})")
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=[
            'pool_id', 'symbol', 'chain', 'protocol', 'forecasted_apy', 'forecasted_tvl', 'actual_apy', 'underlying_tokens'
        ])

        logger.info(f"Loaded {len(df)} pools for {target_date} ({len(allocated_pool_ids)} with current allocations)")

        # Normalize underlying tokens using approved tokens mapping
        normalized_tokens_list = []
        for _, row in df.iterrows():
            raw_tokens = row.get('underlying_tokens')
            pool_tokens = []

            # Parse if string
            if isinstance(raw_tokens, str):
                try:
                    pool_tokens = json.loads(raw_tokens)
                except:
                    pool_tokens = []
            elif isinstance(raw_tokens, list):
                pool_tokens = raw_tokens

            # Normalize
            normalized_pool_tokens = []
            if pool_tokens:
                for t in pool_tokens:
                    t_lower = t.lower() if isinstance(t, str) else str(t).lower()
                    if t_lower in self.addr_to_symbol:
                        normalized_pool_tokens.append(self.addr_to_symbol[t_lower])
                    else:
                        normalized_pool_tokens.append(t)

            normalized_tokens_list.append(normalized_pool_tokens)

        df['underlying_tokens'] = normalized_tokens_list
        return df

    def fetch_token_prices_for_date(self, tokens: List[str], target_date: date) -> Dict[str, float]:
        """
        Fetch token prices for a specific date.

        Note: Currently uses latest prices as historical daily prices aren't readily available.
        This can be enhanced when historical price data is stored.

        Args:
            tokens: List of token symbols
            target_date: Date to fetch prices for (currently uses latest)

        Returns:
            Dictionary mapping token symbol to price
        """
        # For now, use latest prices
        # TODO: Implement historical price fetching when data is available
        prices = self.raw_data_repo.get_latest_prices(tokens)

        # Set default price for stablecoins at 1.0 if not found
        stablecoins = ['USDC', 'USDT', 'DAI', 'USDD', 'FRAX', 'TUSD', 'USDP', 'BUSD']
        for token in tokens:
            if token not in prices and token in stablecoins:
                prices[token] = 1.0

        logger.info(f"Loaded prices for {len(prices)} tokens for {target_date}")
        return prices

    def fetch_gas_fees_for_date(self, target_date: date) -> Tuple[float, float, float, float, float]:
        """
        Fetch gas fee data for a specific date.

        Note: Currently uses default values. Can be enhanced with historical gas data.

        Args:
            target_date: Date to fetch gas fees for

        Returns:
            Tuple of (eth_price, base_fee_transfer_gwei, base_fee_swap_gwei, priority_fee_gwei, min_gas_units)
        """
        # For now, use the default implementation
        return fetch_gas_fee_data()

    def compound_yields_in_pools(self, target_date: date) -> Tuple[float, float]:
        """
        Compound yields directly into pool allocations using ACTUAL APY.
        
        Portfolio simulation uses actual APY (what really happened).
        Forecasted APY is tracked separately for analysis purposes only.
        
        Args:
            target_date: Today's date (yield is from previous day)
            
        Returns:
            Tuple of (forecasted_yield_usd, actual_yield_usd)
        """
        if not self.current_allocations:
            return 0.0, 0.0
        
        if self.previous_day_pools_df is None or self.previous_day_pools_df.empty:
            return 0.0, 0.0
        
        total_forecasted_yield = 0.0
        total_actual_yield = 0.0
        
        # Create a copy to iterate while modifying
        allocations_to_update = {}
        
        for (pool_id, token), amount in self.current_allocations.items():
            # Get pool APY from previous day's data
            pool_info = self.previous_day_pools_df[
                self.previous_day_pools_df['pool_id'] == pool_id
            ]
            
            if pool_info.empty:
                # Keep allocation unchanged if no pool info
                allocations_to_update[(pool_id, token)] = amount
                continue
            
            forecasted_apy = float(pool_info['forecasted_apy'].iloc[0])
            actual_apy = float(pool_info['actual_apy'].iloc[0])
            price = self.previous_day_token_prices.get(token, 1.0)
            
            # Track forecasted yield (for analysis only - what we expected)
            if forecasted_apy > 0:
                forecasted_daily_rate = forecasted_apy / 100.0 / 365.0
                forecasted_position_yield = amount * forecasted_daily_rate
                total_forecasted_yield += forecasted_position_yield * price
            
            # Use ACTUAL APY for compounding (what really happened)
            if actual_apy > 0:
                actual_daily_rate = actual_apy / 100.0 / 365.0
                actual_position_yield = amount * actual_daily_rate
                new_amount = amount + actual_position_yield
                allocations_to_update[(pool_id, token)] = new_amount
                total_actual_yield += actual_position_yield * price
            else:
                # No actual yield, keep allocation unchanged
                allocations_to_update[(pool_id, token)] = amount
        
        # Update allocations with compounded amounts (using actual yield)
        self.current_allocations = allocations_to_update
        
        # Update AUM tracking using ACTUAL yield
        self.current_aum += total_actual_yield
        self.cumulative_actual_yield += total_actual_yield
        self.cumulative_forecasted_yield += total_forecasted_yield
        
        logger.info(f"  Forecasted yield: ${total_forecasted_yield:,.2f}, Actual yield: ${total_actual_yield:,.2f}")
        if total_forecasted_yield > 0:
            logger.info(f"  Forecast accuracy: {(total_actual_yield / total_forecasted_yield * 100):.1f}% of forecast")
        
        return total_forecasted_yield, total_actual_yield

    def optimize_for_date(self, target_date: date, alloc_params: Dict) -> Optional[Dict]:
        """
        Run optimization for a specific date.

        Args:
            target_date: Date to optimize for
            alloc_params: Allocation parameters

        Returns:
            Dictionary with optimization results or None if failed
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"OPTIMIZING FOR {target_date}")
        logger.info(f"{'='*80}")

        # 1. Fetch data for this date
        pools_df = self.fetch_pool_data_for_date(target_date)

        if pools_df.empty:
            error_msg = "No pools available for optimization"
            logger.error(error_msg)
            return {
                'date': target_date,
                'success': False,
                'error': error_msg,
                'error_category': 'no_pools'
            }

        # Build token universe
        tokens = build_token_universe(pools_df, self.warm_wallet, self.current_allocations)
        token_prices = self.fetch_token_prices_for_date(tokens + ['ETH'], target_date)

        # Fetch gas fees
        eth_price, base_fee_transfer_gwei, base_fee_swap_gwei, priority_fee_gwei, min_gas_units = \
            self.fetch_gas_fees_for_date(target_date)
        gas_fees = calculate_transaction_gas_fees(
            eth_price, base_fee_transfer_gwei, base_fee_swap_gwei, priority_fee_gwei, min_gas_units
        )

        # 2. Initialize optimizer
        try:
            optimizer = AllocationOptimizer(
                pools_df, token_prices, self.warm_wallet, self.current_allocations,
                gas_fees, alloc_params
            )
        except Exception as e:
            error_msg = f"Failed to initialize optimizer: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                'date': target_date,
                'success': False,
                'error': error_msg,
                'error_category': 'optimizer_init'
            }

        # 3. Solve
        try:
            success = optimizer.solve()
        except Exception as e:
            error_msg = f"Solver error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                'date': target_date,
                'success': False,
                'error': error_msg,
                'error_category': 'solver_error'
            }

        if not success:
            error_msg = "Optimization failed to find a solution"
            logger.error(error_msg)

            # Determine specific failure reason
            min_pools = alloc_params.get('min_pools', 0)
            pool_tvl_limit = alloc_params.get('pool_tvl_limit', 0)

            eligible_pools = 0
            for pool_id, tvl in optimizer.pool_tvl.items():
                if tvl >= pool_tvl_limit:
                    eligible_pools += 1

            failure_reason = f"Solver failed. Eligible pools (TVL >= ${pool_tvl_limit:,}): {eligible_pools}, Required: {min_pools}"

            return {
                'date': target_date,
                'success': False,
                'error': failure_reason,
                'error_category': 'solver_infeasible',
                'eligible_pools': eligible_pools,
                'required_pools': min_pools,
                'pool_tvl_limit': pool_tvl_limit
            }

        # 4. Extract results
        try:
            allocations_df, transactions = optimizer.extract_results()
            formatted_results = optimizer.format_results()
        except Exception as e:
            error_msg = f"Failed to extract results: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                'date': target_date,
                'success': False,
                'error': error_msg,
                'error_category': 'result_extraction'
            }

        # 5. Calculate metrics
        total_costs = sum(float(txn.get('total_cost_usd', 0)) for txn in transactions)
        allocated_value = sum(row['amount_usd'] for _, row in allocations_df.iterrows())
        wallet_value = sum(
            amt * token_prices.get(tok, 1.0)
            for tok, amt in formatted_results.get('unallocated_tokens', {}).items()
            for amt in [formatted_results['unallocated_tokens'][tok].get('amount', 0)]
        )

        # Calculate projected APY
        if not allocations_df.empty:
            merged = allocations_df.merge(pools_df[['pool_id', 'forecasted_apy']], on='pool_id', how='left')
            total_amt = merged['amount_usd'].sum()
            if total_amt > 0:
                projected_apy = (merged['amount_usd'] * merged['forecasted_apy']).sum() / total_amt
            else:
                projected_apy = 0.0
        else:
            projected_apy = 0.0

        new_aum = allocated_value + wallet_value

        return {
            'date': target_date,
            'success': True,
            'allocations': formatted_results['final_allocations'],
            'unallocated_tokens': formatted_results['unallocated_tokens'],
            'transactions': transactions,
            'metrics': {
                'total_aum': new_aum,
                'allocated_value': allocated_value,
                'wallet_value': wallet_value,
                'transaction_costs': total_costs,
                'projected_apy': projected_apy,
                'num_pools': len(formatted_results['final_allocations']),
                'num_transactions': len(transactions)
            }
        }

    def update_state(self, result: Dict, target_date: date):
        """
        Update the backtesting state with optimization results.

        Args:
            result: Optimization result dictionary
            target_date: Current optimization date
        """
        if not result['success']:
            # Keep previous state on failure
            logger.warning(f"Keeping previous state for {target_date} due to optimization failure")
            return

        # Update current allocations
        self.current_allocations = {}
        for pool_id, pool_data in result['allocations'].items():
            for token, token_data in pool_data['tokens'].items():
                self.current_allocations[(pool_id, token)] = token_data['amount']

        # Update warm wallet
        self.warm_wallet = {}
        for token, token_data in result['unallocated_tokens'].items():
            self.warm_wallet[token] = token_data['amount']

        # Update AUM
        self.current_aum = result['metrics']['total_aum']

        logger.info(f"Updated state for {target_date}:")
        logger.info(f"  AUM: ${self.current_aum:,.2f}")
        logger.info(f"  Allocated: ${result['metrics']['allocated_value']:,.2f}")
        logger.info(f"  Wallet: ${result['metrics']['wallet_value']:,.2f}")
        logger.info(f"  Pools: {result['metrics']['num_pools']}")

    def run_backtest(self, alloc_params: Optional[Dict] = None) -> Dict:
        """
        Run the full backtest over the specified date range.

        Args:
            alloc_params: Allocation parameters to use (optional)

        Returns:
            Dictionary with backtest summary and results
        """
        if alloc_params is None:
            alloc_params = {}

        # Set default parameters
        default_params = {
            'max_alloc_percentage': 0.25,
            'conversion_rate': 0.0004,
            'tvl_limit_percentage': 0.05,
            'min_pools': 4,
            'optimization_horizon_days': 1,
            'pool_tvl_limit': 0
        }

        # Merge with user-provided params
        final_params = {**default_params, **alloc_params}

        # Initialize with starting USDC in warm wallet
        self.warm_wallet = {'USDC': self.initial_aum}
        self.current_allocations = {}
        self.current_aum = self.initial_aum

        logger.info(f"\nStarting backtest with parameters:")
        for key, value in final_params.items():
            logger.info(f"  {key}: {value}")

        # Generate list of dates
        dates = []
        current = self.start_date
        while current <= self.end_date:
            dates.append(current)
            current += timedelta(days=1)

        logger.info(f"\nRunning backtest for {len(dates)} days...")

        # Run optimization for each date
        for i, target_date in enumerate(dates, 1):
            logger.info(f"\n[{i}/{len(dates)}] Processing {target_date}")

            # Step 1: Compound yield from previous day's allocations directly in pools
            if i > 1:  # Skip first day (no previous allocations to earn from)
                forecasted_yield, actual_yield = self.compound_yields_in_pools(target_date)
            else:
                forecasted_yield, actual_yield = 0.0, 0.0

            # Step 2: Run optimization for today
            result = self.optimize_for_date(target_date, final_params)

            if result['success']:
                # Add yield info to result (both forecasted and actual)
                result['metrics']['forecasted_yield_earned'] = forecasted_yield
                result['metrics']['actual_yield_earned'] = actual_yield
                result['metrics']['cumulative_forecasted_yield'] = self.cumulative_forecasted_yield
                result['metrics']['cumulative_actual_yield'] = self.cumulative_actual_yield
                
                self.daily_results.append(result)
                self.update_state(result, target_date)
                
                # Store today's pool data for tomorrow's yield calculation
                self.previous_day_pools_df = self.fetch_pool_data_for_date(target_date)
                # Refresh token prices for yield calc
                if not self.previous_day_pools_df.empty:
                    tokens = build_token_universe(
                        self.previous_day_pools_df, 
                        self.warm_wallet, 
                        self.current_allocations
                    )
                    self.previous_day_token_prices = self.fetch_token_prices_for_date(
                        tokens + ['ETH'], target_date
                    )
            else:
                # Add yield info even on failure
                result['forecasted_yield_earned'] = forecasted_yield
                result['actual_yield_earned'] = actual_yield
                result['cumulative_forecasted_yield'] = self.cumulative_forecasted_yield
                result['cumulative_actual_yield'] = self.cumulative_actual_yield
                self.failed_days.append(result)

        # Generate summary
        return self._generate_summary()

    def _generate_summary(self) -> Dict:
        """Generate backtest summary."""
        total_days = len(self.daily_results) + len(self.failed_days)
        success_days = len(self.daily_results)
        failed_days_count = len(self.failed_days)
        success_rate = (success_days / total_days * 100) if total_days > 0 else 0

        # Calculate returns
        if self.daily_results:
            first_aum = self.daily_results[0]['metrics']['total_aum']
            last_aum = self.daily_results[-1]['metrics']['total_aum']
            
            # Get all APYs
            apys = [r['metrics']['projected_apy'] for r in self.daily_results]
            avg_apy = np.mean(apys) if apys else 0

            # Get all transaction costs
            costs = [r['metrics']['transaction_costs'] for r in self.daily_results]
            total_costs = sum(costs)

            # Get all yield earned (both forecasted and actual)
            forecasted_yields = [r['metrics'].get('forecasted_yield_earned', 0) for r in self.daily_results]
            actual_yields = [r['metrics'].get('actual_yield_earned', 0) for r in self.daily_results]
            total_forecasted_yield = sum(forecasted_yields)
            total_actual_yield = sum(actual_yields)
            
            # Net profit = actual yield - transaction costs
            net_profit = total_actual_yield - total_costs
            
            # Total return based on net profit (most accurate)
            total_return = (net_profit / self.initial_aum * 100) if self.initial_aum > 0 else 0
            
            # Annualized APY based on actual net performance
            # Formula: ((1 + period_return) ^ (365/days)) - 1
            num_days = len(self.daily_results)
            if num_days > 0 and self.initial_aum > 0:
                period_return = net_profit / self.initial_aum
                net_annualized_apy = ((1 + period_return) ** (365 / num_days) - 1) * 100
            else:
                net_annualized_apy = 0

            # Get pool counts
            pool_counts = [r['metrics']['num_pools'] for r in self.daily_results]
            avg_pools = np.mean(pool_counts) if pool_counts else 0
        else:
            first_aum = self.initial_aum
            last_aum = self.current_aum
            total_return = 0
            avg_apy = 0
            total_costs = 0
            total_forecasted_yield = 0
            total_actual_yield = 0
            net_profit = 0
            net_annualized_apy = 0
            avg_pools = 0

        # Analyze failures
        failure_categories = {}
        for failure in self.failed_days:
            category = failure.get('error_category', 'unknown')
            failure_categories[category] = failure_categories.get(category, 0) + 1

        summary = {
            'period': {
                'start_date': self.start_date.isoformat(),
                'end_date': self.end_date.isoformat(),
                'total_days': total_days
            },
            'initial_aum': self.initial_aum,
            'final_aum': last_aum,
            'performance': {
                'total_return_pct': total_return,
                'net_annualized_apy': net_annualized_apy,
                'total_forecasted_yield_earned': total_forecasted_yield,
                'total_actual_yield_earned': total_actual_yield,
                'forecast_accuracy_pct': (total_actual_yield / total_forecasted_yield * 100) if total_forecasted_yield > 0 else 0,
                'total_transaction_costs': total_costs,
                'net_profit': net_profit,
                'avg_projected_apy': avg_apy,
                'avg_num_pools': avg_pools
            },
            'success_rate': {
                'successful_days': success_days,
                'failed_days': failed_days_count,
                'success_rate_pct': success_rate
            },
            'failures': {
                'by_category': failure_categories,
                'details': self.failed_days
            }
        }

        return summary

    def save_results(self, output_path: str):
        """
        Save backtest results to file.

        Args:
            output_path: Path to save results JSON
        """
        summary = self._generate_summary()

        # Prepare serializable output
        output = {
            'summary': {
                'period': summary['period'],
                'initial_aum': summary['initial_aum'],
                'final_aum': summary['final_aum'],
                'performance': summary['performance'],
                'success_rate': summary['success_rate']
            },
            'failures': summary['failures'],
            'daily_results': []
        }

        # Add daily results
        for result in self.daily_results:
            daily = {
                'date': result['date'].isoformat(),
                'metrics': result['metrics'],
                'allocations': result['allocations'],  # Full allocation details
                'transactions': result['transactions'],  # Full transaction details
                'num_allocations': len(result['allocations']),
                'num_transactions': len(result['transactions'])
            }
            output['daily_results'].append(daily)

        # Write to file
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"\nResults saved to {output_path}")

    def print_summary(self):
        """Print backtest summary to console."""
        summary = self._generate_summary()

        print(f"\n{'='*80}")
        print("BACKTEST SUMMARY")
        print(f"{'='*80}")
        print(f"\nPeriod:")
        print(f"  Start: {summary['period']['start_date']}")
        print(f"  End: {summary['period']['end_date']}")
        print(f"  Total Days: {summary['period']['total_days']}")

        print(f"\nAUM:")
        print(f"  Initial: ${summary['initial_aum']:,.2f}")
        print(f"  Final: ${summary['final_aum']:,.2f}")

        print(f"\nPerformance:")
        print(f"  Total Return: {summary['performance']['total_return_pct']:.4f}%")
        print(f"  Net Annualized APY: {summary['performance']['net_annualized_apy']:.2f}%")
        print(f"  Avg Projected APY: {summary['performance']['avg_projected_apy']:.2f}%")
        print(f"  Avg Num Pools: {summary['performance']['avg_num_pools']:.1f}")

        print(f"\nYield Analysis (Forecast vs Actual):")
        print(f"  Total Forecasted Yield: ${summary['performance']['total_forecasted_yield_earned']:,.2f}")
        print(f"  Total Actual Yield: ${summary['performance']['total_actual_yield_earned']:,.2f}")
        print(f"  Forecast Accuracy: {summary['performance']['forecast_accuracy_pct']:.2f}%")
        print(f"  Total Transaction Costs: ${summary['performance']['total_transaction_costs']:,.2f}")
        print(f"  Net Profit (Actual): ${summary['performance']['net_profit']:,.2f}")

        print(f"\nSuccess Rate:")
        print(f"  Successful Days: {summary['success_rate']['successful_days']}")
        print(f"  Failed Days: {summary['success_rate']['failed_days']}")
        print(f"  Success Rate: {summary['success_rate']['success_rate_pct']:.1f}%")

        if summary['failures']['by_category']:
            print(f"\nFailures by Category:")
            for category, count in summary['failures']['by_category'].items():
                print(f"  {category}: {count}")

            print(f"\nFailure Details:")
            for failure in summary['failures']['details']:
                print(f"  {failure['date']}: {failure['error']}")

        print(f"\n{'='*80}\n")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Backtest asset allocation optimization algorithm'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        default='2026-01-01',
        help='Start date (YYYY-MM-DD format, default: 2026-01-01)'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        default='2026-01-31',
        help='End date (YYYY-MM-DD format, default: 2026-01-31)'
    )

    parser.add_argument(
        '--initial-aum',
        type=float,
        default=1_000_000,
        help='Initial AUM in USD (default: 1,000,000)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='backtest_results.json',
        help='Output file path for results (default: backtest_results.json)'
    )

    parser.add_argument(
        '--min-pools',
        type=int,
        default=4,
        help='Minimum number of pools required (default: 4)'
    )

    parser.add_argument(
        '--max-alloc-pct',
        type=float,
        default=0.25,
        help='Maximum allocation percentage per pool (default: 0.25)'
    )

    parser.add_argument(
        '--tvl-limit-pct',
        type=float,
        default=0.05,
        help='TVL limit percentage per pool (default: 0.05)'
    )

    parser.add_argument(
        '--pool-tvl-limit',
        type=float,
        default=0,
        help='Minimum pool TVL in USD (default: 0)'
    )

    parser.add_argument(
        '--optimization-horizon',
        type=int,
        default=30,
        help='Optimization horizon in days (default: 30)'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    # Parse dates
    try:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        logger.error("Please use YYYY-MM-DD format")
        sys.exit(1)

    # Validate dates
    if end_date < start_date:
        logger.error("End date must be after start date")
        sys.exit(1)

    # Setup allocation parameters
    alloc_params = {
        'min_pools': args.min_pools,
        'max_alloc_percentage': args.max_alloc_pct,
        'tvl_limit_percentage': args.tvl_limit_pct,
        'pool_tvl_limit': args.pool_tvl_limit,
        'optimization_horizon_days': args.optimization_horizon
    }

    # Initialize backtester
    backtester = BacktestOptimizer(
        start_date=start_date,
        end_date=end_date,
        initial_aum=args.initial_aum
    )

    # Run backtest
    try:
        backtester.run_backtest(alloc_params=alloc_params)
    except Exception as e:
        logger.error(f"Backtest failed with error: {e}", exc_info=True)
        sys.exit(1)

    # Print summary
    backtester.print_summary()

    # Save results
    backtester.save_results(args.output)

    logger.info("Backtest completed successfully!")


if __name__ == "__main__":
    main()
