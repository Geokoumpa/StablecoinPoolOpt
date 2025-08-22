import subprocess
import sys
import logging
from datetime import datetime
from pathlib import Path

# Configuration Loading
import config

# Import database utilities for migrations
from database.db_utils import apply_migrations

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_script(script_path, description=None):
    """
    Execute a Python script as a subprocess using python -m module syntax.
    """
    try:
        if description:
            logger.info(f"Executing {description}...")
        
        # Convert file path to module path (e.g., data_ingestion/fetch_defillama_pools.py -> data_ingestion.fetch_defillama_pools)
        module_path = script_path.replace('/', '.').replace('.py', '')
        
        result = subprocess.run(
            [sys.executable, '-m', module_path],
            cwd=Path.cwd(),
            capture_output=False,  # Allow output to be shown in real-time
            text=True,
            check=True
        )
        
        if description:
            logger.info(f"{description} completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing {description or script_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error executing {description or script_path}: {e}")
        raise

def run_pipeline(phases=None):
    """
    Main pipeline orchestrator for local testing.
    Executes data ingestion, processing, forecasting, and asset allocation scripts.
    """
    logger.info("Starting DefiYieldOpt local pipeline execution...")

    # Apply database migrations first
    logger.info("Applying database migrations...")
    apply_migrations()

    # Import historical gas data (one-time operation)
    logger.info("Importing historical gas data...")
    try:
        run_script("data_ingestion.import_historical_gas_data", "import_historical_gas_data.py")
    except Exception as e:
        logger.error(f"Error during historical gas data import: {e}")
        # Don't exit - this is a one-time operation that might already be completed

    if phases is None:
        phases = ["all"] # Default to running all phases

    start_time = datetime.now()
    
    # Phase 1: Initial Data Ingestion
    if "all" in phases or "phase1" in phases:
        logger.info("--- Phase 1: Initial Data Ingestion ---")
        try:
            run_script("data_ingestion.fetch_defillama_pools", "fetch_defillama_pools.py")
            run_script("data_ingestion.fetch_ohlcv_coinmarketcap", "fetch_ohlcv_coinmarketcap.py")
            run_script("data_ingestion.fetch_gas_ethgastracker", "fetch_gas_ethgastracker.py")
            run_script("data_ingestion.fetch_account_data_etherscan", "fetch_account_data_etherscan.py")
            logger.info("Phase 1 completed successfully.")
        except Exception as e:
            logger.error(f"Error during Phase 1 (Initial Data Ingestion): {e}")
            sys.exit(1)

    # Phase 2: Pre-Filtering & Pool History Ingestion
    if "all" in phases or "phase2" in phases:
        logger.info("--- Phase 2: Pre-Filtering & Pool History Ingestion ---")
        try:
            run_script("data_processing.filter_pools_pre", "filter_pools_pre.py")
            run_script("data_ingestion.fetch_filtered_pool_histories", "fetch_filtered_pool_histories.py")
            logger.info("Phase 2 completed successfully.")
        except Exception as e:
            logger.error(f"Error during Phase 2 (Pre-Filtering & Pool History Ingestion): {e}")
            sys.exit(1)

    # Phase 3: Pool Analysis & Final Filtering
    if "all" in phases or "phase3" in phases:
        logger.info("--- Phase 3: Pool Analysis & Final Filtering ---")
        try:
            run_script("data_processing.calculate_pool_metrics", "calculate_pool_metrics.py")
            run_script("data_processing.apply_pool_grouping", "apply_pool_grouping.py")
            run_script("data_processing.process_icebox_logic", "process_icebox_logic.py")
            run_script("data_processing.filter_pools_final", "filter_pools_final.py")
            logger.info("Phase 3 completed successfully.")
        except Exception as e:
            logger.error(f"Error during Phase 3 (Pool Analysis & Final Filtering): {e}")
            sys.exit(1)

    # Phase 4: Fresh Data & Snapshots
    if "all" in phases or "phase4" in phases:
        logger.info("--- Phase 4: Fresh Data & Snapshots ---")
        try:
            
            run_script("data_processing.create_allocation_snapshots", "create_allocation_snapshots.py")
            logger.info("Phase 4 completed successfully.")
        except Exception as e:
            logger.error(f"Error during Phase 4 (Fresh Data & Snapshots): {e}")
            sys.exit(1)

    # Phase 5: Forecasting
    if "all" in phases or "phase5" in phases:
        logger.info("--- Phase 5: Forecasting ---")
        try:
            run_script("forecasting.forecast_pools", "forecast_pools.py")
            run_script("forecasting.forecast_gas_fees", "forecast_gas_fees.py")
            logger.info("Phase 5 completed successfully.")
        except Exception as e:
            logger.error(f"Error during Phase 5 (Forecasting): {e}")
            sys.exit(1)

    # Phase 6: Asset Allocation
    if "all" in phases or "phase6" in phases:
        logger.info("--- Phase 6: Asset Allocation ---")
        try:
            run_script("asset_allocation.optimize_allocations", "optimize_allocations.py")
            logger.info("Phase 6 completed successfully.")
        except Exception as e:
            logger.error(f"Error during Phase 6 (Asset Allocation): {e}")
            sys.exit(1)

    # Phase 7: Reporting & Notification
    if "all" in phases or "phase7" in phases:
        logger.info("--- Phase 7: Reporting & Notification ---")
        try:
            run_script("reporting_notification.manage_ledger", "manage_ledger.py")
            run_script("reporting_notification.post_slack_notification", "post_slack_notification.py")
            logger.info("Phase 7 completed successfully.")
        except Exception as e:
            logger.error(f"Error during Phase 7 (Reporting & Notification): {e}")
            sys.exit(1)

    end_time = datetime.now()
    duration = end_time - start_time
    
    # Print comprehensive pipeline summary
    print("\n" + "="*80)
    print("🚀 DEFIYIELDOPT PIPELINE EXECUTION SUMMARY")
    print("="*80)
    print(f"⏰ Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🏁 End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Total duration: {duration}")
    print(f"📋 Phases executed: {', '.join(phases)}")
    print("")
    print("📊 PHASE BREAKDOWN:")
    if "all" in phases or "phase1" in phases:
        print("  ✅ Phase 1: Initial Data Ingestion")
        print("     • DeFiLlama pools data fetched")
        print("     • CoinMarketCap OHLCV data fetched")
        print("     • Gas tracker data fetched")
        print("     • Etherscan account data fetched")
    if "all" in phases or "phase2" in phases:
        print("  ✅ Phase 2: Pre-Filtering & Pool History Ingestion")
        print("     • Pre-filtering applied (no icebox)")
        print("     • Historical data fetched for filtered pools")
    if "all" in phases or "phase3" in phases:
        print("  ✅ Phase 3: Pool Analysis & Final Filtering")
        print("     • Pool metrics calculated")
        print("     • Pool grouping applied")
        print("     • Icebox logic processed")
        print("     • Final filtering completed (with icebox)")
    if "all" in phases or "phase4" in phases:
        print("  ✅ Phase 4: Fresh Data & Snapshots")
        
        print("     • Allocation snapshots created")
    if "all" in phases or "phase5" in phases:
        print("  ✅ Phase 5: Forecasting")
        print("     • Pool forecasts generated")
        print("     • Gas fee forecasts generated")
    if "all" in phases or "phase6" in phases:
        print("  ✅ Phase 6: Asset Allocation")
        print("     • Portfolio optimization completed")
    if "all" in phases or "phase7" in phases:
        print("  ✅ Phase 7: Reporting & Notification")
        print("     • Daily ledger updated")
        print("     • Slack notifications sent")
    print("")
    print("💾 Data stored in PostgreSQL database")
    print("📈 Ready for next pipeline execution")
    print("="*80)
    
    logger.info(f"DefiYieldOpt local pipeline execution finished in {duration}.")
    logger.info("Summary: All selected phases completed without critical errors.")

if __name__ == "__main__":
    # Example of how to run specific phases from command line:
    # python main_pipeline.py --phases phase1 phase3
    import argparse
    parser = argparse.ArgumentParser(description="Run DefiYieldOpt local pipeline.")
    parser.add_argument("--phases", nargs='*', help="Specify phases to run (e.g., phase1 phase3). Use 'all' to run all phases.", default=["all"])
    args = parser.parse_args()
    
    run_pipeline(phases=args.phases)