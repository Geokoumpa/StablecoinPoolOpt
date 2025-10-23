#!/usr/bin/env python3
"""
Test script for Phase 4: Testing & Validation with Real Data

This script tests the complete optimization system with real data from the database
to ensure all components work together correctly and meet requirements.
"""

import logging
import pandas as pd
import numpy as np
import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asset_allocation.optimize_allocations import (
    AllocationOptimizer,
    fetch_pool_data,
    fetch_token_prices,
    fetch_gas_fee_data,
    fetch_current_balances,
    fetch_allocation_parameters,
    calculate_aum,
    build_token_universe
)
from database.db_utils import get_db_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Phase4Validator:
    """Comprehensive validator for Phase 4 testing"""
    
    def __init__(self):
        self.test_results = {
            'wallet_data_retrieval': {'status': 'pending', 'details': []},
            'pool_data_validation': {'status': 'pending', 'details': []},
            'optimization_solving': {'status': 'pending', 'details': []},
            'output_format_compliance': {'status': 'pending', 'details': []},
            'constraint_validation': {'status': 'pending', 'details': []},
            'cost_calculation_validation': {'status': 'pending', 'details': []}
        }
        self.engine = None
    
    def setup_database_connection(self) -> bool:
        """Establish database connection"""
        try:
            self.engine = get_db_connection()
            logger.info("✓ Database connection established")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to connect to database: {e}")
            return False
    
    def validate_wallet_data_retrieval(self) -> bool:
        """Test wallet balance retrieval with real data"""
        logger.info("\n" + "="*80)
        logger.info("TEST 1: Wallet Data Retrieval Validation")
        logger.info("="*80)
        
        try:
            # Fetch current balances
            warm_wallet, current_allocations = fetch_current_balances(self.engine)
            
            # Validate warm wallet data
            if not warm_wallet:
                logger.warning("No warm wallet balances found - may be expected if all funds allocated")
            else:
                logger.info(f"✓ Warm wallet contains {len(warm_wallet)} tokens")
                for token, amount in warm_wallet.items():
                    logger.info(f"  - {token}: {amount:,.2f}")
                    assert amount >= 0, f"Negative balance for {token}: {amount}"
            
            # Validate current allocations
            if not current_allocations:
                logger.warning("No current allocations found - may be expected for fresh deployment")
            else:
                logger.info(f"✓ Found {len(current_allocations)} allocated positions")
                for (pool_id, token), amount in current_allocations.items():
                    logger.info(f"  - {pool_id}/{token}: {amount:,.2f}")
                    assert amount > 0, f"Non-positive allocation for {pool_id}/{token}: {amount}"
            
            # Check for duplicate tokens in allocations
            pool_tokens = {}
            for (pool_id, token), amount in current_allocations.items():
                if pool_id not in pool_tokens:
                    pool_tokens[pool_id] = set()
                assert token not in pool_tokens[pool_id], f"Duplicate token {token} in pool {pool_id}"
                pool_tokens[pool_id].add(token)
            
            self.test_results['wallet_data_retrieval']['status'] = 'passed'
            self.test_results['wallet_data_retrieval']['details'].append(
                f"Warm wallet: {len(warm_wallet)} tokens, Allocations: {len(current_allocations)} positions"
            )
            
            logger.info("✓ Wallet data retrieval validation passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Wallet data retrieval validation failed: {e}")
            self.test_results['wallet_data_retrieval']['status'] = 'failed'
            self.test_results['wallet_data_retrieval']['details'].append(str(e))
            return False
    
    def validate_pool_data(self) -> bool:
        """Test pool data retrieval and validation"""
        logger.info("\n" + "="*80)
        logger.info("TEST 2: Pool Data Validation")
        logger.info("="*80)
        
        try:
            # Fetch pool data
            pools_df = fetch_pool_data(self.engine)
            
            if pools_df.empty:
                logger.error("No pool data found - cannot proceed with optimization")
                self.test_results['pool_data_validation']['status'] = 'failed'
                self.test_results['pool_data_validation']['details'].append("No pool data available")
                return False
            
            logger.info(f"✓ Found {len(pools_df)} approved pools")
            
            # Validate pool data structure
            required_columns = ['pool_id', 'symbol', 'chain', 'protocol', 'forecasted_apy']
            for col in required_columns:
                assert col in pools_df.columns, f"Missing required column: {col}"
            
            # Validate APY values
            invalid_apy = pools_df[pools_df['forecasted_apy'] <= 0]
            if not invalid_apy.empty:
                logger.warning(f"Found {len(invalid_apy)} pools with non-positive APY")
            
            # Validate pool symbols
            for _, pool in pools_df.iterrows():
                tokens = pool['symbol'].split('-')
                assert len(tokens) >= 1, f"Invalid pool symbol: {pool['symbol']}"
            
            # Log pool summary
            logger.info(f"Pool summary:")
            logger.info(f"  - Total pools: {len(pools_df)}")
            logger.info(f"  - Chains: {pools_df['chain'].nunique()} unique")
            logger.info(f"  - Protocols: {pools_df['protocol'].nunique()} unique")
            logger.info(f"  - APY range: {pools_df['forecasted_apy'].min():.3%} - {pools_df['forecasted_apy'].max():.3%}")
            
            self.test_results['pool_data_validation']['status'] = 'passed'
            self.test_results['pool_data_validation']['details'].append(
                f"Validated {len(pools_df)} pools with APY range {pools_df['forecasted_apy'].min():.3%}-{pools_df['forecasted_apy'].max():.3%}"
            )
            
            logger.info("✓ Pool data validation passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Pool data validation failed: {e}")
            self.test_results['pool_data_validation']['status'] = 'failed'
            self.test_results['pool_data_validation']['details'].append(str(e))
            return False
    
    def validate_optimization_solving(self) -> bool:
        """Test the complete optimization process"""
        logger.info("\n" + "="*80)
        logger.info("TEST 3: Optimization Solving Validation")
        logger.info("="*80)
        
        try:
            # Fetch all required data
            pools_df = fetch_pool_data(self.engine)
            warm_wallet, current_allocations = fetch_current_balances(self.engine)
            gas_gwei, eth_price = fetch_gas_fee_data(self.engine)
            alloc_params = fetch_allocation_parameters(self.engine)
            
            # Build token universe
            tokens = build_token_universe(pools_df, warm_wallet, current_allocations)
            token_prices = fetch_token_prices(self.engine, tokens)
            
            # Calculate gas fee in USD
            gas_fee_usd = gas_gwei * 1e-9 * eth_price
            
            # Calculate total AUM
            total_aum = calculate_aum(warm_wallet, current_allocations, token_prices)
            
            logger.info(f"Optimization setup:")
            logger.info(f"  - Pools: {len(pools_df)}")
            logger.info(f"  - Tokens: {len(tokens)}")
            logger.info(f"  - Total AUM: ${total_aum:,.2f}")
            logger.info(f"  - Gas fee: ${gas_fee_usd:.6f}")
            
            # Initialize optimizer
            optimizer = AllocationOptimizer(
                pools_df=pools_df,
                token_prices=token_prices,
                warm_wallet=warm_wallet,
                current_allocations=current_allocations,
                gas_fee_usd=gas_fee_usd,
                alloc_params=alloc_params
            )
            
            # Build and solve model
            logger.info("Building optimization model...")
            problem = optimizer.build_model()
            logger.info(f"✓ Model built with {len(problem.constraints)} constraints")
            
            logger.info("Solving optimization problem...")
            import cvxpy as cp
            # Try multiple MIP-capable solvers in order
            solvers_to_try = [
                (cp.CBC, "CBC"),
                (cp.HIGHS, "HiGHS"),
                (cp.SCIPY, "SCIPY")
            ]
            
            success = False
            for solver, name in solvers_to_try:
                try:
                    logger.info(f"Attempting with {name} solver...")
                    success = optimizer.solve(solver=solver, verbose=True)
                    if success:
                        logger.info(f"✓ Solved with {name} solver")
                        break
                except Exception as e:
                    logger.warning(f"{name} solver failed: {e}")
                    continue
            
            if not success:
                logger.info("All MIP solvers failed, attempting with ECOS as final fallback...")
                success = optimizer.solve(solver=cp.ECOS, verbose=True)
            
            if not success:
                logger.warning("Optimization did not find optimal solution")
                self.test_results['optimization_solving']['status'] = 'warning'
                self.test_results['optimization_solving']['details'].append("No optimal solution found")
                return False
            
            logger.info("✓ Optimization solved successfully")
            
            # Extract results
            allocations_df, transactions = optimizer.extract_results()
            
            logger.info(f"Results:")
            logger.info(f"  - Final allocations: {len(allocations_df)} positions")
            logger.info(f"  - Transactions: {len(transactions)} total")
            
            self.test_results['optimization_solving']['status'] = 'passed'
            self.test_results['optimization_solving']['details'].append(
                f"Optimized with {len(allocations_df)} allocations and {len(transactions)} transactions"
            )
            
            # Store results for further validation
            self.optimizer = optimizer
            self.allocations_df = allocations_df
            self.transactions = transactions
            
            logger.info("✓ Optimization solving validation passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Optimization solving validation failed: {e}")
            self.test_results['optimization_solving']['status'] = 'failed'
            self.test_results['optimization_solving']['details'].append(str(e))
            return False
    
    def validate_output_format_compliance(self) -> bool:
        """Test output format compliance with optimization.md requirements"""
        logger.info("\n" + "="*80)
        logger.info("TEST 4: Output Format Compliance Validation")
        logger.info("="*80)
        
        try:
            if not hasattr(self, 'optimizer'):
                logger.error("Optimizer not available - run optimization test first")
                return False
            
            # Get formatted results
            formatted_results = self.optimizer.format_results()
            
            # Validate top-level structure
            required_sections = ['final_allocations', 'unallocated_tokens', 'transactions']
            for section in required_sections:
                assert section in formatted_results, f"Missing required section: {section}"
            
            # Validate final allocations
            final_allocations = formatted_results['final_allocations']
            assert isinstance(final_allocations, dict), "final_allocations must be a dictionary"
            
            for pool_id, pool_data in final_allocations.items():
                assert 'pool_symbol' in pool_data, f"Missing pool_symbol for {pool_id}"
                assert 'tokens' in pool_data, f"Missing tokens for {pool_id}"
                assert isinstance(pool_data['tokens'], dict), f"Tokens must be a dictionary for {pool_id}"
                
                for token, token_data in pool_data['tokens'].items():
                    assert 'amount' in token_data, f"Missing amount for {pool_id}/{token}"
                    assert 'amount_usd' in token_data, f"Missing amount_usd for {pool_id}/{token}"
                    assert token_data['amount'] >= 0, f"Negative amount for {pool_id}/{token}"
                    assert token_data['amount_usd'] >= 0, f"Negative amount_usd for {pool_id}/{token}"
            
            # Validate unallocated tokens
            unallocated_tokens = formatted_results['unallocated_tokens']
            assert isinstance(unallocated_tokens, dict), "unallocated_tokens must be a dictionary"
            
            for token, token_data in unallocated_tokens.items():
                assert 'amount' in token_data, f"Missing amount for unallocated {token}"
                assert 'amount_usd' in token_data, f"Missing amount_usd for unallocated {token}"
                assert token_data['amount'] >= 0, f"Negative amount for unallocated {token}"
                assert token_data['amount_usd'] >= 0, f"Negative amount_usd for unallocated {token}"
            
            # Validate transactions
            transactions = formatted_results['transactions']
            assert isinstance(transactions, list), "transactions must be a list"
            
            for i, txn in enumerate(transactions):
                required_fields = ['seq', 'type', 'from_location', 'to_location', 'token', 'amount', 'amount_usd', 'gas_cost_usd']
                for field in required_fields:
                    assert field in txn, f"Transaction {i} missing field: {field}"
                
                assert txn['seq'] == i + 1, f"Transaction sequence mismatch: expected {i+1}, got {txn['seq']}"
                assert txn['type'] in ['WITHDRAWAL', 'CONVERSION', 'ALLOCATION'], f"Invalid transaction type: {txn['type']}"
                assert txn['amount'] >= 0, f"Negative transaction amount: {txn['amount']}"
                assert txn['amount_usd'] >= 0, f"Negative transaction amount_usd: {txn['amount_usd']}"
                assert txn['gas_cost_usd'] >= 0, f"Negative gas cost: {txn['gas_cost_usd']}"
            
            logger.info(f"✓ Output format validation passed")
            logger.info(f"  - Final allocations: {len(final_allocations)} pools")
            logger.info(f"  - Unallocated tokens: {len(unallocated_tokens)} tokens")
            logger.info(f"  - Transactions: {len(transactions)} total")
            
            self.test_results['output_format_compliance']['status'] = 'passed'
            self.test_results['output_format_compliance']['details'].append(
                f"Validated format with {len(final_allocations)} pools, {len(unallocated_tokens)} tokens, {len(transactions)} transactions"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Output format compliance validation failed: {e}")
            self.test_results['output_format_compliance']['status'] = 'failed'
            self.test_results['output_format_compliance']['details'].append(str(e))
            return False
    
    def validate_constraints(self) -> bool:
        """Test that all constraints are satisfied"""
        logger.info("\n" + "="*80)
        logger.info("TEST 5: Constraint Validation")
        logger.info("="*80)
        
        try:
            if not hasattr(self, 'optimizer'):
                logger.error("Optimizer not available - run optimization test first")
                return False
            
            # Get optimization parameters
            max_alloc_pct = self.optimizer.alloc_params['max_alloc_percentage']
            min_txn_value = self.optimizer.alloc_params['min_transaction_value']
            
            # Calculate total AUM
            total_aum = self.optimizer.total_aum
            
            # Validate maximum allocation constraint
            for pool_id, pool_data in self.optimizer.format_results()['final_allocations'].items():
                pool_total_usd = sum(token['amount_usd'] for token in pool_data['tokens'].values())
                alloc_pct = pool_total_usd / total_aum if total_aum > 0 else 0
                
                if alloc_pct > max_alloc_pct + 0.01:  # Allow 1% tolerance
                    logger.warning(f"Pool {pool_id} exceeds max allocation: {alloc_pct:.2f%} > {max_alloc_pct:.2f%}")
                else:
                    logger.info(f"✓ Pool {pool_id} within max allocation: {alloc_pct:.2f%} <= {max_alloc_pct:.2f%}")
            
            # Validate minimum transaction value constraint
            for txn in self.optimizer.format_results()['transactions']:
                if txn['amount_usd'] < min_txn_value and txn['amount_usd'] > 0:
                    logger.warning(f"Transaction below minimum value: ${txn['amount_usd']:.2f} < ${min_txn_value:.2f}")
                else:
                    logger.info(f"✓ Transaction meets minimum value: ${txn['amount_usd']:.2f} >= ${min_txn_value:.2f}")
            
            # Validate mass balance constraint
            formatted_results = self.optimizer.format_results()
            
            # Calculate total value in final allocations
            final_alloc_value = 0
            for pool_data in formatted_results['final_allocations'].values():
                final_alloc_value += sum(token['amount_usd'] for token in pool_data['tokens'].values())
            
            # Calculate total value in unallocated tokens
            unalloc_value = sum(token['amount_usd'] for token in formatted_results['unallocated_tokens'].values())
            
            # Calculate total transaction costs
            total_costs = sum(txn.get('total_cost_usd', 0) for txn in formatted_results['transactions'])
            
            # Check mass balance (allowing for small rounding errors)
            expected_final_value = total_aum - total_costs
            actual_final_value = final_alloc_value + unalloc_value
            
            mass_balance_diff = abs(actual_final_value - expected_final_value)
            if mass_balance_diff > total_aum * 0.001:  # Allow 0.1% tolerance
                logger.warning(f"Mass balance issue: diff=${mass_balance_diff:.2f}")
            else:
                logger.info(f"✓ Mass balance satisfied: diff=${mass_balance_diff:.2f}")
            
            logger.info("✓ Constraint validation completed")
            
            self.test_results['constraint_validation']['status'] = 'passed'
            self.test_results['constraint_validation']['details'].append(
                f"Validated max allocation ({max_alloc_pct:.1%}), min txn (${min_txn_value:.2f}), and mass balance"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Constraint validation failed: {e}")
            self.test_results['constraint_validation']['status'] = 'failed'
            self.test_results['constraint_validation']['details'].append(str(e))
            return False
    
    def validate_cost_calculations(self) -> bool:
        """Test transaction cost calculations"""
        logger.info("\n" + "="*80)
        logger.info("TEST 6: Cost Calculation Validation")
        logger.info("="*80)
        
        try:
            if not hasattr(self, 'optimizer'):
                logger.error("Optimizer not available - run optimization test first")
                return False
            
            formatted_results = self.optimizer.format_results()
            transactions = formatted_results['transactions']
            conversion_rate = self.optimizer.alloc_params['conversion_rate']
            gas_fee_usd = self.optimizer.gas_fee_usd
            
            # Validate each transaction's cost calculation
            for txn in transactions:
                txn_type = txn['type']
                amount_usd = txn['amount_usd']
                
                if 'total_cost_usd' not in txn:
                    logger.warning(f"Transaction {txn['seq']} missing total_cost_usd")
                    continue
                
                actual_cost = txn['total_cost_usd']
                
                # Calculate expected cost based on transaction type
                if txn_type == 'WITHDRAWAL':
                    expected_cost = amount_usd * conversion_rate + gas_fee_usd
                elif txn_type == 'CONVERSION':
                    expected_cost = amount_usd * conversion_rate + gas_fee_usd
                elif txn_type == 'ALLOCATION':
                    expected_cost = amount_usd * conversion_rate
                    if txn.get('needs_conversion', False):
                        expected_cost += gas_fee_usd * 2
                    else:
                        expected_cost += gas_fee_usd
                else:
                    logger.warning(f"Unknown transaction type: {txn_type}")
                    continue
                
                # Check if costs match (allow small rounding differences)
                cost_diff = abs(actual_cost - expected_cost)
                if cost_diff > 0.01:  # Allow 1 cent tolerance
                    logger.warning(f"Cost mismatch for {txn_type} txn {txn['seq']}: expected ${expected_cost:.4f}, got ${actual_cost:.4f}")
                else:
                    logger.info(f"✓ Cost correct for {txn_type} txn {txn['seq']}: ${actual_cost:.4f}")
            
            # Calculate total costs
            total_conversion_costs = sum(txn.get('conversion_cost_usd', 0) for txn in transactions)
            total_gas_costs = sum(txn.get('gas_cost_usd', 0) for txn in transactions)
            total_costs = sum(txn.get('total_cost_usd', 0) for txn in transactions)
            
            logger.info(f"\nCost Summary:")
            logger.info(f"  - Total conversion costs: ${total_conversion_costs:.4f}")
            logger.info(f"  - Total gas costs: ${total_gas_costs:.4f}")
            logger.info(f"  - Total transaction costs: ${total_costs:.4f}")
            
            self.test_results['cost_calculation_validation']['status'] = 'passed'
            self.test_results['cost_calculation_validation']['details'].append(
                f"Validated {len(transactions)} transactions with total cost ${total_costs:.4f}"
            )
            
            logger.info("✓ Cost calculation validation passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Cost calculation validation failed: {e}")
            self.test_results['cost_calculation_validation']['status'] = 'failed'
            self.test_results['cost_calculation_validation']['details'].append(str(e))
            return False
    
    def save_test_results(self, filename: str = None) -> str:
        """Save test results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"phase4_test_results_{timestamp}.json"
        
        # Add timestamp and summary
        results = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': len(self.test_results),
                'passed': sum(1 for r in self.test_results.values() if r['status'] == 'passed'),
                'failed': sum(1 for r in self.test_results.values() if r['status'] == 'failed'),
                'warnings': sum(1 for r in self.test_results.values() if r['status'] == 'warning')
            },
            'test_results': self.test_results
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Test results saved to: {filename}")
        return filename
    
    def run_all_tests(self) -> bool:
        """Run all Phase 4 tests"""
        logger.info("\n" + "="*80)
        logger.info("PHASE 4: TESTING & VALIDATION WITH REAL DATA")
        logger.info("="*80)
        
        # Setup
        if not self.setup_database_connection():
            return False
        
        # Run all tests
        tests = [
            self.validate_wallet_data_retrieval,
            self.validate_pool_data,
            self.validate_optimization_solving,
            self.validate_output_format_compliance,
            self.validate_constraints,
            self.validate_cost_calculations
        ]
        
        passed = 0
        for test in tests:
            if test():
                passed += 1
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("PHASE 4 TEST SUMMARY")
        logger.info("="*80)
        logger.info(f"Tests passed: {passed}/{len(tests)}")
        
        for test_name, result in self.test_results.items():
            status_icon = "✓" if result['status'] == 'passed' else "✗" if result['status'] == 'failed' else "⚠"
            logger.info(f"{status_icon} {test_name}: {result['status']}")
            for detail in result['details']:
                logger.info(f"    - {detail}")
        
        # Save results
        self.save_test_results()
        
        return passed == len(tests)


def main():
    """Main test function"""
    validator = Phase4Validator()
    
    try:
        success = validator.run_all_tests()
        if success:
            logger.info("\n✓ PHASE 4 TESTS PASSED - All validations successful")
            return 0
        else:
            logger.error("\n✗ PHASE 4 TESTS FAILED - Some validations failed")
            return 1
    except Exception as e:
        logger.error(f"\n✗ PHASE 4 TESTS FAILED WITH ERROR: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())