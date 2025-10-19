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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger()

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

    # Create initial allocation parameter snapshots
    logger.info("Creating initial allocation parameter snapshots...")
    run_script("data_processing.create_allocation_snapshots", "create_allocation_snapshots.py")

    if phases is None:
        phases = ["all"] # Default to running all phases

    start_time = datetime.now()
    
    # Phase 1: Initial Data Ingestion
    if "all" in phases or "phase1" in phases:
        logger.info("--- Phase 1: Initial Data Ingestion ---")
        try:
            run_script("data_ingestion.fetch_ohlcv_coinmarketcap", "fetch_ohlcv_coinmarketcap.py")
            run_script("data_ingestion.fetch_gas_ethgastracker", "fetch_gas_ethgastracker.py")
            run_script("data_ingestion.fetch_defillama_pools", "fetch_defillama_pools.py")
            run_script("data_ingestion.fetch_account_transactions", "fetch_account_transactions.py")
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
            # Update allocation snapshots after icebox logic
            run_script("data_processing.update_allocation_snapshots", "update_allocation_snapshots.py")
            run_script("data_processing.filter_pools_final", "filter_pools_final.py")
            run_script("data_processing.process_account_transactions", "process_account_transactions.py")
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
            run_script("reporting_notification.manage_ledger", "manage_ledger.py")
            run_script("asset_allocation.optimize_allocations", "optimize_allocations.py")
            logger.info("Phase 6 completed successfully.")
        except Exception as e:
            logger.error(f"Error during Phase 6 (Asset Allocation): {e}")
            sys.exit(1)

    # Phase 7: Reporting & Notification
    if "all" in phases or "phase7" in phases:
        logger.info("--- Phase 7: Reporting & Notification ---")
        try:
            run_script("reporting_notification.post_slack_notification", "post_slack_notification.py")
            logger.info("Phase 7 completed successfully.")
        except Exception as e:
            logger.error(f"Error during Phase 7 (Reporting & Notification): {e}")
            sys.exit(1)

    end_time = datetime.now()
    duration = end_time - start_time
    
    # Print comprehensive pipeline summary
    logger.info("\n" + "="*80)
    logger.info("🚀 DEFIYIELDOPT PIPELINE EXECUTION SUMMARY")
    logger.info("="*80)
    logger.info(f"⏰ Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🏁 End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"⏱️  Total duration: {duration}")
    logger.info(f"📋 Phases executed: {', '.join(phases)}")
    logger.info("")
    logger.info("📊 PHASE BREAKDOWN:")
    logger.info("  📸 Initial allocation snapshots created (after migrations)")
    if "all" in phases or "phase1" in phases:
        logger.info("  ✅ Phase 1: Initial Data Ingestion")
        logger.info("     • DeFiLlama pools data fetched")
        logger.info("     • CoinMarketCap OHLCV data fetched")
        logger.info("     • Gas tracker data fetched")
        logger.info("     • Ethplorer account data fetched")
    if "all" in phases or "phase2" in phases:
        logger.info("  ✅ Phase 2: Pre-Filtering & Pool History Ingestion")
        logger.info("     • Pre-filtering applied (no icebox)")
        logger.info("     • Historical data fetched for filtered pools")
    if "all" in phases or "phase3" in phases:
        logger.info("  ✅ Phase 3: Pool Analysis & Final Filtering")
        logger.info("     • Pool metrics calculated")
        logger.info("     • Pool grouping applied")
        logger.info("     • Icebox logic processed")
        logger.info("     • Allocation snapshots updated (post-icebox)")
        logger.info("     • Final filtering completed (with icebox)")
        logger.info("     • Historical metrics calculated for filtered pools (6 months)")
        logger.info("     • Missing historical metrics filled for filtered pools")
    if "all" in phases or "phase4" in phases:
        logger.info("  ✅ Phase 4: Fresh Data & Snapshots")

        logger.info("     • Final allocation snapshots created")
    if "all" in phases or "phase5" in phases:
        logger.info("  ✅ Phase 5: Forecasting")
        logger.info("     • Pool forecasts generated")
        logger.info("     • Gas fee forecasts generated")
    if "all" in phases or "phase6" in phases:
        logger.info("  ✅ Phase 6: Asset Allocation")
        logger.info("     • Daily balances updated")
        logger.info("     • Portfolio optimization completed")
    if "all" in phases or "phase7" in phases:
        logger.info("  ✅ Phase 7: Reporting & Notification")
        logger.info("     • Slack notifications sent")
    logger.info("")
    logger.info("💾 Data stored in PostgreSQL database")
    logger.info("📈 Ready for next pipeline execution")
    logger.info("="*80)
    
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