#!/usr/bin/env python3
"""
Universal Pipeline Runner
This script can execute any pipeline step by name, handling both regular modules and special functions.
"""

import sys
import os
import logging
import json
from pathlib import Path

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "name": record.name
        }
        return json.dumps(log_record)

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Remove any existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Create a new handler and set the formatter
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)


# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def run_apply_migrations():
    """Special handler for apply_migrations function."""
    from database.db_utils import apply_migrations
    apply_migrations()

def run_module(module_path):
    """Run a regular Python module."""
    import importlib
    module = importlib.import_module(module_path)
    # The module's if __name__ == "__main__": block will execute

def main():
    """Main entry point for pipeline runner."""
    if len(sys.argv) != 2:
        logger.error("Usage: python pipeline_runner.py <script_name>")
        sys.exit(1)
    
    script_name = sys.argv[1]
    logger.info(f"Starting pipeline step: {script_name}")
    
    try:
        # Special case for apply_migrations
        if script_name == "apply_migrations":
            logger.info("Running database migrations...")
            run_apply_migrations()
            logger.info("Database migrations completed successfully")
        else:
            # Regular module execution
            # Convert script name to module path
            module_mapping = {
                # Data ingestion modules
                "fetch_defillama_pools": "data_ingestion.fetch_defillama_pools",
                "fetch_ohlcv_coinmarketcap": "data_ingestion.fetch_ohlcv_coinmarketcap", 
                "fetch_gas_ethgastracker": "data_ingestion.fetch_gas_ethgastracker",
                "fetch_account_data_etherscan": "data_ingestion.fetch_account_data_etherscan",
                "fetch_filtered_pool_histories": "data_ingestion.fetch_filtered_pool_histories",
                
                # Data processing modules
                "create_allocation_snapshots": "data_processing.create_allocation_snapshots",
                "filter_pools_pre": "data_processing.filter_pools_pre",
                "calculate_pool_metrics": "data_processing.calculate_pool_metrics",
                "apply_pool_grouping": "data_processing.apply_pool_grouping",
                "process_icebox_logic": "data_processing.process_icebox_logic",
                "update_allocation_snapshots": "data_processing.update_allocation_snapshots",
                "filter_pools_final": "data_processing.filter_pools_final",
                
                # Forecasting modules  
                "forecast_pools": "forecasting.forecast_pools",
                "forecast_gas_fees": "forecasting.forecast_gas_fees",
                
                # Asset allocation modules
                "optimize_allocations": "asset_allocation.optimize_allocations",
                
                # Reporting modules
                "manage_ledger": "reporting_notification.manage_ledger",
                "post_slack_notification": "reporting_notification.post_slack_notification"
            }
            
            if script_name in module_mapping:
                module_path = module_mapping[script_name]
                logger.info(f"Running module: {module_path}")
                run_module(module_path)
                logger.info(f"Module {script_name} completed successfully")
            else:
                logger.error(f"Unknown script: {script_name}")
                logger.error(f"Available scripts: {', '.join(sorted(module_mapping.keys()))}")
                sys.exit(1)
                
    except Exception as e:
        logger.error(f"Error executing {script_name}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()