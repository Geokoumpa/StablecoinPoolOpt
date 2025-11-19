import unittest
import pandas as pd
import numpy as np
from datetime import timedelta
from unittest.mock import patch, MagicMock, Mock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from forecasting.global_forecasting import (
    train_and_forecast_global, 
    get_filtered_pool_ids,
    persist_global_forecasts
)
from forecasting.panel_data_utils import fetch_panel_history, build_pool_feature_row
from forecasting.neighbor_features import add_neighbor_features
from forecasting.model_utils import fit_global_panel_model, make_tvl_oof

class TestGlobalForecasting(unittest.TestCase):
    """Test suite for global LightGBM forecasting functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample pool data with 10 rows
        self.sample_pool_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10, freq='D'),
            'pool_id': ['TEST_POOL_1'] * 5 + ['TEST_POOL_2'] * 5,
            'apy_7d': [0.05, 0.06, 0.04, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13],
            'actual_apy': [0.05, 0.06, 0.04, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13],
            'tvl_usd': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
            'actual_tvl': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
            'eth_open': [2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900],
            'btc_open': [40000, 41000, 42000, 43000, 44000, 45000, 46000, 47000, 48000, 49000],
            'gas_price_gwei': [20, 22, 24, 26, 28, 30, 32, 34, 36, 38],
            'pool_group': [1, 1, 2, 2, 1, 2, 2, 2, 1, 2]
        })
        
        # Create sample panel data with 20 rows
        self.sample_panel_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=20, freq='D'),
            'pool_id': ['TEST_POOL_1'] * 10 + ['TEST_POOL_2'] * 10,
            'apy_7d': [0.05, 0.06, 0.04, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13] * 2,
            'actual_apy': [0.05, 0.06, 0.04, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13] * 2,
            'tvl_usd': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900] * 2,
            'actual_tvl': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900] * 2,
            'eth_open': [2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900] * 2,
            'btc_open': [40000, 41000, 42000, 43000, 44000, 45000, 46000, 47000, 48000, 49000] * 2,
            'gas_price_gwei': [20, 22, 24, 26, 28, 30, 32, 34, 36, 38] * 2,
            'pool_group': [1] * 20
        })

    def test_get_filtered_pool_ids(self):
        """Test get_filtered_pool_ids function."""
        mock_df = pd.DataFrame({'pool_id': ['POOL_1', 'POOL_2']})
        
        with patch('forecasting.global_forecasting.get_db_connection') as mock_get_conn:
            mock_conn = MagicMock()
            mock_get_conn.return_value = mock_conn
            
            # Mock pd.read_sql to return our dataframe
            with patch('pandas.read_sql', return_value=mock_df):
                result = get_filtered_pool_ids()
            
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertIn('POOL_1', result)
        self.assertIn('POOL_2', result)

    def test_fetch_panel_history(self):
        """Test fetch_panel_history function."""
        with patch('forecasting.panel_data_utils.get_db_connection') as mock_get_conn:
            mock_conn = MagicMock()
            mock_get_conn.return_value = mock_conn
            
            # Mock pd.read_sql to return our sample panel data
            with patch('pandas.read_sql', return_value=self.sample_panel_data):
                result = fetch_panel_history(
                    pd.Timestamp('2023-01-10'), 
                    ['TEST_POOL_1', 'TEST_POOL_2'], 
                    days=10,
                    group_col='pool_group'
                )
            
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)
        self.assertIn('date', result.columns)
        self.assertIn('pool_id', result.columns)
        self.assertIn('apy_7d', result.columns)

    def test_add_neighbor_features(self):
        """Test add_neighbor_features function."""
        # The function requires columns to be added via concat, need to check the actual implementation
        result = add_neighbor_features(self.sample_panel_data.copy(), group_col='pool_group')
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreaterEqual(len(result), len(self.sample_panel_data))
        
        # Check that at least some neighbor columns are added
        # The function concatenates Series, so columns should be present
        neighbor_cols = [
            'group_tvl_sum_t_nbr', 'group_apy_mean_t_nbr', 'group_apy_median_t_nbr',
            'group_apy_std_t_nbr', 'tvl_share_nbr', 'apy_rank_nbr',
            'grp_ex_mean_t_nbr'
        ]
        
        for col in neighbor_cols:
            self.assertIn(col, result.columns, f"Missing column: {col}")

    def test_build_pool_feature_row(self):
        """Test build_pool_feature_row function."""
        # This test requires mocking preprocess_data since it needs EXOG_BASE constant
        # Skip this test for now as it requires internal implementation details
        panel_df = self.sample_panel_data.copy()
        
        # Test with missing data - this should return empty dict
        empty_panel = pd.DataFrame({'date': [], 'pool_id': []})
        result = build_pool_feature_row(empty_panel, 'TEST_POOL_1', pd.Timestamp('2023-01-10'))
        self.assertEqual(result, {})
        
        # For valid data test, we'd need to mock preprocess_data and its dependencies
        # which makes the test too complex. Skip for now.

    def test_fit_global_panel_model(self):
        """Test fit_global_panel_model function."""
        # Create test panel with targets
        test_panel = self.sample_panel_data.copy()
        test_panel['target_apy_t1'] = [0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15] * 2
        test_panel = test_panel.dropna(subset=['target_apy_t1'])
        
        model, feat_cols = fit_global_panel_model(test_panel, target_col='target_apy_t1')
        
        self.assertIsNotNone(model)
        self.assertIsInstance(feat_cols, list)
        self.assertGreater(len(feat_cols), 0)

    def test_make_tvl_oof(self):
        """Test make_tvl_oof function."""
        # Create test panel with TVL targets and 'asof' column (required by make_tvl_oof)
        test_panel = self.sample_panel_data.copy()
        test_panel['target_tvl_t1'] = [1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000] * 2
        test_panel['asof'] = test_panel['date']  # Add asof column
        
        panel_with_oof, tvl_model, oof_mae = make_tvl_oof(test_panel, n_splits=3)
        
        self.assertIsInstance(panel_with_oof, pd.DataFrame)
        self.assertIsNotNone(tvl_model)
        self.assertIsInstance(oof_mae, float)
        self.assertIn('tvl_hat_t1_oof', panel_with_oof.columns)

    def test_train_and_forecast_global(self):
        """Test train_and_forecast_global function."""
        with patch('forecasting.global_forecasting.get_filtered_pool_ids') as mock_get_ids:
            mock_get_ids.return_value = ['TEST_POOL_1', 'TEST_POOL_2']
            
            with patch('forecasting.global_forecasting.persist_global_forecasts') as mock_persist:
                with patch('forecasting.global_forecasting.train_global_models') as mock_train:
                    # Mock return value from train_global_models
                    mock_train.return_value = (
                        MagicMock(),  # apy_model
                        MagicMock(),  # tvl_model
                        ['feature1', 'feature2'],  # apy_features
                        ['tvl_feature1', 'tvl_feature2']  # tvl_features
                    )
                    
                    # Mock predict_global_lgbm to return proper dataframe
                    with patch('forecasting.global_forecasting.predict_global_lgbm') as mock_predict:
                        mock_predict.return_value = pd.DataFrame({
                            'pool_id': ['TEST_POOL_1', 'TEST_POOL_2'],
                            'target_date': [pd.Timestamp('2023-01-11')] * 2,
                            'pred_global_apy': [0.08, 0.09],
                            'pred_tvl_t1': [1500, 1600],
                            'cold_start_flag': [False, False]
                        })
                        
                        result = train_and_forecast_global(
                            pool_ids=None,
                            train_days=10,
                            forecast_ahead=1,
                            use_tvl_stacking=True
                        )
                        
                        # Verify function calls
                        mock_get_ids.assert_called_once()
                        
                        # Check result structure
                        self.assertIsInstance(result, dict)
                        self.assertIn('total_pools', result)
                        self.assertIn('model_pools', result)
                        self.assertIn('cold_start_pools', result)

    def test_persist_global_forecasts(self):
        """Test persist_global_forecasts function."""
        predictions_df = pd.DataFrame({
            'pool_id': ['TEST_POOL_1', 'TEST_POOL_2'],
            'target_date': [pd.Timestamp('2023-01-11'), pd.Timestamp('2023-01-11')],
            'pred_global_apy': [0.08, 0.09],
            'pred_tvl_t1': [1500, 1600],
            'cold_start_flag': [False, False]
        })
        
        with patch('forecasting.global_forecasting.get_db_connection') as mock_get_conn:
            # Create mock engine
            mock_engine = MagicMock()
            
            # Create mock connection (returned by engine.connect())
            mock_conn = MagicMock()
            
            # Create mock result
            mock_result = MagicMock()
            mock_result.fetchone.return_value = (0,)  # Return tuple with 0
            
            # conn.execute() returns mock_result
            mock_conn.execute.return_value = mock_result
            
            # Set up context manager for connection
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = False
            
            # engine.connect() returns mock_conn (as context manager)
            mock_engine.connect.return_value = mock_conn
            
            # get_db_connection() returns mock_engine
            mock_get_conn.return_value = mock_engine
            
            # Call the function
            persist_global_forecasts(predictions_df)
            
            # Verify database connection was made
            mock_get_conn.assert_called_once()
            # Verify engine.connect() was called
            mock_engine.connect.assert_called()

    def test_error_handling(self):
        """Test error handling in global forecasting."""
        with patch('forecasting.global_forecasting.get_filtered_pool_ids') as mock_get_ids:
            mock_get_ids.return_value = []  # No pools available
            
            result = train_and_forecast_global()
            
            # Should handle empty pool list gracefully
            self.assertIsInstance(result, dict)
            self.assertEqual(result.get('total_pools', 0), 0)
            self.assertEqual(result.get('model_pools', 0), 0)

if __name__ == '__main__':
    unittest.main()