import logging

from datetime import datetime
from uuid import uuid4
from typing import Dict, Any, List

from database.repositories.parameter_repository import ParameterRepository
from database.repositories.token_repository import TokenRepository
from database.repositories.protocol_repository import ProtocolRepository
from database.models.allocation_parameters import AllocationParameters

logger = logging.getLogger(__name__)

def fetch_dynamic_lists() -> Dict[str, List[Dict[str, str]]]:
    """Fetches current dynamic lists using repositories."""
    token_repo = TokenRepository()
    protocol_repo = ProtocolRepository()
    
    # Approved tokens
    approved_tokens = token_repo.get_approved_tokens()
    approved_tokens_list = [{"token_symbol": t.token_symbol} for t in approved_tokens]
    
    # Blacklisted tokens
    blacklisted_tokens = token_repo.get_blacklisted_tokens()
    blacklisted_tokens_list = [{"token_symbol": t.token_symbol} for t in blacklisted_tokens]
    
    # Icebox tokens
    icebox_tokens = token_repo.get_icebox_tokens()
    icebox_tokens_list = [{"token_symbol": t.token_symbol} for t in icebox_tokens]
    
    # Approved protocols
    approved_protocols = protocol_repo.get_approved_protocols()
    approved_protocols_list = [{"protocol_name": p.protocol_name} for p in approved_protocols]
    
    return {
        "approved_tokens": approved_tokens_list,
        "blacklisted_tokens": blacklisted_tokens_list,
        "approved_protocols": approved_protocols_list,
        "icebox_tokens": icebox_tokens_list,
    }

def get_base_params(param_repo: ParameterRepository) -> Dict[str, Any]:
    """
    Get base parameters from defaults and latest run.
    """
    # 1. Get defaults from DB
    defaults = param_repo.get_all_default_parameters()
    
    # 2. Get latest params
    latest_params = param_repo.get_latest_parameters()
    
    # Define fallback defaults (hardcoded) in case DB is empty
    fallback_defaults = {
        'tvl_limit_percentage': 0.05,
        'max_alloc_percentage': 0.25,
        'conversion_rate': 0.0004,
        'min_pools': 5,
        'profit_optimization': True,
        'token_marketcap_limit': 1000000000.0,
        'pool_tvl_limit': 100000.0,
        'pool_apy_limit': 0.01,
        'pool_pair_tvl_ratio_min': 0.3,
        'pool_pair_tvl_ratio_max': 0.5,
        'group1_max_pct': 0.35,
        'group2_max_pct': 0.35,
        'group3_max_pct': 0.3,
        'position_max_pct_total_assets': 0.25,
        'position_max_pct_pool_tvl': 0.05,
        'group1_apy_delta_max': 0.01,
        'group1_7d_stddev_max': 0.015,
        'group1_30d_stddev_max': 0.02,
        'group2_apy_delta_max': 0.03,
        'group2_7d_stddev_max': 0.04,
        'group2_30d_stddev_max': 0.05,
        'group3_apy_delta_min': 0.03,
        'group3_7d_stddev_min': 0.04,
        'group3_30d_stddev_min': 0.02,
        'icebox_ohlc_l_threshold_pct': 0.02,
        'icebox_ohlc_l_days_threshold': 2,
        'icebox_ohlc_c_threshold_pct': 0.01,
        'icebox_ohlc_c_days_threshold': 1,
        'icebox_recovery_l_days_threshold': 2,
        'icebox_recovery_c_days_threshold': 3
    }
    
    # Merge strategy:
    # 1. Start with fallbacks
    params = fallback_defaults.copy()
    
    # 2. Update with defaults from table
    for k, v in defaults.items():
        if v is not None:
             # handle profit_optimization bool conversion if needed
             if k == 'profit_optimization' and isinstance(v, str):
                 params[k] = v.lower() == 'true'
             else:
                 params[k] = v
                 
    # 3. Update with latest params
    if latest_params:
        logger.info(f"Using latest parameters from run_id: {latest_params.run_id}")
        # Map object attributes to dict
        attrs = [
            'tvl_limit_percentage', 'max_alloc_percentage', 'conversion_rate', 'min_pools', 'profit_optimization',
            'token_marketcap_limit', 'pool_tvl_limit', 'pool_apy_limit', 'pool_pair_tvl_ratio_min', 'pool_pair_tvl_ratio_max',
            'group1_max_pct', 'group2_max_pct', 'group3_max_pct', 'position_max_pct_total_assets', 'position_max_pct_pool_tvl',
            'group1_apy_delta_max', 'group1_7d_stddev_max', 'group1_30d_stddev_max',
            'group2_apy_delta_max', 'group2_7d_stddev_max', 'group2_30d_stddev_max',
            'group3_apy_delta_min', 'group3_7d_stddev_min', 'group3_30d_stddev_min',
            'other_dynamic_limits',
            'icebox_ohlc_l_threshold_pct', 'icebox_ohlc_l_days_threshold',
            'icebox_ohlc_c_threshold_pct', 'icebox_ohlc_c_days_threshold',
            'icebox_recovery_l_days_threshold', 'icebox_recovery_c_days_threshold'
        ]
        for attr in attrs:
            val = getattr(latest_params, attr, None)
            if val is not None:
                params[attr] = val
    else:
        logger.info("No previous allocation parameters found. Using defaults.")
        
    return params

def create_allocation_snapshots():
    """
    Creates snapshots of current dynamic lists and stores them in allocation_parameters
    for use by subsequent filtering and processing steps.
    """
    logger.info("Creating allocation parameter snapshots...")
    
    try:
        param_repo = ParameterRepository()
        
        # 1. Fetch dynamic lists
        dynamic_lists = fetch_dynamic_lists()
        
        # 2. Get base parameters
        params = get_base_params(param_repo)
        
        # 3. Create new AllocationParameters object
        new_run_id = uuid4()
        
        new_params = AllocationParameters(
            run_id=new_run_id,
            timestamp=datetime.now(),
            
            # Snapshots
            approved_tokens_snapshot=dynamic_lists['approved_tokens'],
            blacklisted_tokens_snapshot=dynamic_lists['blacklisted_tokens'],
            approved_protocols_snapshot=dynamic_lists['approved_protocols'],
            icebox_tokens_snapshot=dynamic_lists['icebox_tokens'],

            # Parameters
            tvl_limit_percentage=params.get('tvl_limit_percentage'),
            max_alloc_percentage=params.get('max_alloc_percentage'),
            conversion_rate=params.get('conversion_rate'),
            min_pools=params.get('min_pools'),
            profit_optimization=params.get('profit_optimization'),
            
            token_marketcap_limit=params.get('token_marketcap_limit'),
            pool_tvl_limit=params.get('pool_tvl_limit'),
            pool_apy_limit=params.get('pool_apy_limit'),
            pool_pair_tvl_ratio_min=params.get('pool_pair_tvl_ratio_min'),
            pool_pair_tvl_ratio_max=params.get('pool_pair_tvl_ratio_max'),
            
            group1_max_pct=params.get('group1_max_pct'),
            group2_max_pct=params.get('group2_max_pct'),
            group3_max_pct=params.get('group3_max_pct'),
            position_max_pct_total_assets=params.get('position_max_pct_total_assets'),
            position_max_pct_pool_tvl=params.get('position_max_pct_pool_tvl'),
            
            group1_apy_delta_max=params.get('group1_apy_delta_max'),
            group1_7d_stddev_max=params.get('group1_7d_stddev_max'),
            group1_30d_stddev_max=params.get('group1_30d_stddev_max'),
            group2_apy_delta_max=params.get('group2_apy_delta_max'),
            group2_7d_stddev_max=params.get('group2_7d_stddev_max'),
            group2_30d_stddev_max=params.get('group2_30d_stddev_max'),
            group3_apy_delta_min=params.get('group3_apy_delta_min'),
            group3_7d_stddev_min=params.get('group3_7d_stddev_min'),
            group3_30d_stddev_min=params.get('group3_30d_stddev_min'),
            
            other_dynamic_limits=params.get('other_dynamic_limits'),
            
            icebox_ohlc_l_threshold_pct=params.get('icebox_ohlc_l_threshold_pct'),
            icebox_ohlc_l_days_threshold=params.get('icebox_ohlc_l_days_threshold'),
            icebox_ohlc_c_threshold_pct=params.get('icebox_ohlc_c_threshold_pct'),
            icebox_ohlc_c_days_threshold=params.get('icebox_ohlc_c_days_threshold'),
            icebox_recovery_l_days_threshold=params.get('icebox_recovery_l_days_threshold'),
            icebox_recovery_c_days_threshold=params.get('icebox_recovery_c_days_threshold')
        )
        
        # 4. Save to DB
        param_repo.save_parameters(new_params)
        
        logger.info(f"Created allocation parameter snapshot with run_id: {new_run_id}")
        logger.info(f"Approved protocols: {len(dynamic_lists['approved_protocols'])}")
        logger.info(f"Approved tokens: {len(dynamic_lists['approved_tokens'])}")
        logger.info(f"Blacklisted tokens: {len(dynamic_lists['blacklisted_tokens'])}")
        logger.info(f"Icebox tokens: {len(dynamic_lists['icebox_tokens'])}")

    except Exception as e:
        logger.error(f"Error creating allocation snapshots: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_allocation_snapshots()