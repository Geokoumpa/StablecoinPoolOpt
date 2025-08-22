import logging
import json
from datetime import datetime
from uuid import uuid4
from database.db_utils import get_db_connection
import pandas as pd
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_dynamic_lists(engine):
    """Fetches current dynamic lists from individual tables."""
    approved_tokens = pd.read_sql("SELECT token_symbol FROM approved_tokens;", engine)['token_symbol'].tolist()
    blacklisted_tokens = pd.read_sql("SELECT token_symbol FROM blacklisted_tokens;", engine)['token_symbol'].tolist()
    approved_protocols = pd.read_sql("SELECT protocol_name FROM approved_protocols;", engine)['protocol_name'].tolist()
    icebox_tokens = pd.read_sql("SELECT token_symbol FROM icebox_tokens WHERE removed_timestamp IS NULL;", engine)['token_symbol'].tolist()
    
    return {
        "approved_tokens": [{"token_symbol": token} for token in approved_tokens],
        "blacklisted_tokens": [{"token_symbol": token} for token in blacklisted_tokens],
        "approved_protocols": [{"protocol_name": protocol} for protocol in approved_protocols],
        "icebox_tokens": [{"token_symbol": token} for token in icebox_tokens],
    }

def create_allocation_snapshots():
    """
    Creates snapshots of current dynamic lists and stores them in allocation_parameters
    for use by subsequent filtering and processing steps.
    """
    logging.info("Creating allocation parameter snapshots...")
    engine = None
    try:
        engine = get_db_connection()
        if not engine:
            raise Exception("Failed to establish database connection")

        # Fetch current allocation parameters (without snapshots)
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT
                    tvl_limit_percentage,
                    max_alloc_percentage,
                    conversion_rate,
                    min_pools,
                    profit_optimization,
                    token_marketcap_limit,
                    pool_tvl_limit,
                    pool_apy_limit,
                    pool_pair_tvl_ratio_min,
                    pool_pair_tvl_ratio_max,
                    group1_max_pct,
                    group2_max_pct,
                    group3_max_pct,
                    position_max_pct_total_assets,
                    position_max_pct_pool_tvl,
                    group1_apy_delta_max,
                    group1_7d_stddev_max,
                    group1_30d_stddev_max,
                    group2_apy_delta_max,
                    group2_7d_stddev_max,
                    group2_30d_stddev_max,
                    group3_apy_delta_min,
                    group3_7d_stddev_min,
                    group3_30d_stddev_min,
                    other_dynamic_limits,
                    icebox_ohlc_l_threshold_pct,
                    icebox_ohlc_l_days_threshold,
                    icebox_ohlc_c_threshold_pct,
                    icebox_ohlc_c_days_threshold,
                    icebox_recovery_l_days_threshold,
                    icebox_recovery_c_days_threshold
                FROM allocation_parameters
                ORDER BY timestamp DESC LIMIT 1;
            """))
            base_params = result.fetchone()

            if not base_params:
                logging.warning("No base allocation parameters found. Using defaults.")
                # Create default parameters
                base_params = (
                    0.05, 0.25, 0.0004, 4, False,  # Basic params
                    35000000.0, 500000.0, 0.06, 0.3, 0.5,  # Limits
                    0.35, 0.35, 0.3, 0.25, 0.05,  # Group allocations
                    0.01, 0.015, 0.02, 0.03, 0.04, 0.05,  # Group 1 & 2 limits
                    0.03, 0.04, 0.02,  # Group 3 limits
                    None,  # other_dynamic_limits
                    0.02, 2, 0.01, 1, 2, 3  # Icebox params
                )

        # Fetch dynamic lists from individual tables
        dynamic_lists = fetch_dynamic_lists(engine)

        # Create new allocation parameters entry with snapshots
        run_id = uuid4()

        # Insert new parameters in a separate connection with its own transaction
        with engine.connect() as conn:
            with conn.begin():
                conn.execute(text("""
                    INSERT INTO allocation_parameters (
                        run_id, timestamp,
                        tvl_limit_percentage, max_alloc_percentage, conversion_rate, min_pools, profit_optimization,
                        approved_tokens_snapshot, blacklisted_tokens_snapshot, approved_protocols_snapshot, icebox_tokens_snapshot,
                        token_marketcap_limit, pool_tvl_limit, pool_apy_limit, pool_pair_tvl_ratio_min, pool_pair_tvl_ratio_max,
                        group1_max_pct, group2_max_pct, group3_max_pct, position_max_pct_total_assets, position_max_pct_pool_tvl,
                        group1_apy_delta_max, group1_7d_stddev_max, group1_30d_stddev_max,
                        group2_apy_delta_max, group2_7d_stddev_max, group2_30d_stddev_max,
                        group3_apy_delta_min, group3_7d_stddev_min, group3_30d_stddev_min,
                        other_dynamic_limits,
                        icebox_ohlc_l_threshold_pct, icebox_ohlc_l_days_threshold,
                        icebox_ohlc_c_threshold_pct, icebox_ohlc_c_days_threshold,
                        icebox_recovery_l_days_threshold, icebox_recovery_c_days_threshold
                    ) VALUES (
                        :run_id, :timestamp, :tvl_limit_pct, :max_alloc_pct, :conv_rate, :min_pools, :profit_opt,
                        :approved_tokens, :blacklisted_tokens, :approved_protocols, :icebox_tokens,
                        :token_mcap_limit, :pool_tvl_limit, :pool_apy_limit, :pair_tvl_ratio_min, :pair_tvl_ratio_max,
                        :group1_max_pct, :group2_max_pct, :group3_max_pct, :pos_max_pct_total, :pos_max_pct_pool,
                        :g1_apy_delta_max, :g1_7d_std_max, :g1_30d_std_max,
                        :g2_apy_delta_max, :g2_7d_std_max, :g2_30d_std_max,
                        :g3_apy_delta_min, :g3_7d_std_min, :g3_30d_std_min,
                        :other_limits,
                        :icebox_l_pct, :icebox_l_days, :icebox_c_pct, :icebox_c_days,
                        :icebox_recovery_l_days, :icebox_recovery_c_days
                    );
                """), {
                    "run_id": str(run_id),
                    "timestamp": datetime.now(),
                    "tvl_limit_pct": base_params[0],
                    "max_alloc_pct": base_params[1],
                    "conv_rate": base_params[2],
                    "min_pools": base_params[3],
                    "profit_opt": base_params[4],
                    "approved_tokens": json.dumps(dynamic_lists['approved_tokens']),
                    "blacklisted_tokens": json.dumps(dynamic_lists['blacklisted_tokens']),
                    "approved_protocols": json.dumps(dynamic_lists['approved_protocols']),
                    "icebox_tokens": json.dumps(dynamic_lists['icebox_tokens']),
                    "token_mcap_limit": base_params[5],
                    "pool_tvl_limit": base_params[6],
                    "pool_apy_limit": base_params[7],
                    "pair_tvl_ratio_min": base_params[8],
                    "pair_tvl_ratio_max": base_params[9],
                    "group1_max_pct": base_params[10],
                    "group2_max_pct": base_params[11],
                    "group3_max_pct": base_params[12],
                    "pos_max_pct_total": base_params[13],
                    "pos_max_pct_pool": base_params[14],
                    "g1_apy_delta_max": base_params[15],
                    "g1_7d_std_max": base_params[16],
                    "g1_30d_std_max": base_params[17],
                    "g2_apy_delta_max": base_params[18],
                    "g2_7d_std_max": base_params[19],
                    "g2_30d_std_max": base_params[20],
                    "g3_apy_delta_min": base_params[21],
                    "g3_7d_std_min": base_params[22],
                    "g3_30d_std_min": base_params[23],
                    "other_limits": json.dumps(base_params[24]) if base_params[24] else None,
                    "icebox_l_pct": base_params[25],
                    "icebox_l_days": base_params[26],
                    "icebox_c_pct": base_params[27],
                    "icebox_c_days": base_params[28],
                    "icebox_recovery_l_days": base_params[29],
                    "icebox_recovery_c_days": base_params[30]
                })

            logging.info(f"Created allocation parameter snapshot with run_id: {run_id}")
            logging.info(f"Approved protocols: {len(dynamic_lists['approved_protocols'])}")
            logging.info(f"Approved tokens: {len(dynamic_lists['approved_tokens'])}")
            logging.info(f"Blacklisted tokens: {len(dynamic_lists['blacklisted_tokens'])}")
            logging.info(f"Icebox tokens: {len(dynamic_lists['icebox_tokens'])}")

    except Exception as e:
        logging.error(f"Error creating allocation snapshots: {e}")
        raise

if __name__ == "__main__":
    create_allocation_snapshots()