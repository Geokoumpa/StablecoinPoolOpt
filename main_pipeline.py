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


def run_spark_job(job_name):
    """
    Run a Spark job using the local PySpark runner.
    
    Args:
        job_name: Name of the Spark job to run (calculate_metrics or forecast_pools)
    """
    import os
    
    # Set environment for local Spark mode
    os.environ["ENVIRONMENT"] = "local"
    
    try:
        if job_name == "calculate_metrics":
            logger.info("🔥 Running calculate_pool_metrics using local PySpark...")
            from data_processing.calculate_pool_metrics_spark import calculate_pool_metrics_spark
            result = calculate_pool_metrics_spark()
            logger.info(f"Spark calculate_pool_metrics completed: {result}")
            
        elif job_name == "forecast_pools":
            logger.info("🔥 Running forecast_pools using local PySpark...")
            from forecasting.forecast_pools_spark import forecast_pools_spark
            result = forecast_pools_spark()
            logger.info(f"Spark forecast_pools completed: {result}")
            
        else:
            raise ValueError(f"Unknown Spark job: {job_name}")
            
    except ImportError as e:
        logger.error(f"Failed to import Spark modules. Ensure PySpark is installed: pip install -r requirements-spark.txt")
        logger.error(f"Import error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error running Spark job {job_name}: {e}")
        raise


def run_pipeline(phases=None, use_spark=False):
    """
    Main pipeline orchestrator for local testing.
    Executes data ingestion, processing, forecasting, and asset allocation scripts.
    
    Args:
        phases: List of phases to run (e.g., ["phase1", "phase3"] or ["all"])
        use_spark: If True, use local PySpark for calculate_pool_metrics and forecast_pools
    """
    logger.info("Starting DefiYieldOpt local pipeline execution...")
    
    if use_spark:
        logger.info("🔥 SPARK MODE ENABLED: Using local PySpark for calculate_pool_metrics and forecast_pools")

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
            run_script("data_ingestion.fetch_defillama_pool_addresses", "fetch_defillama_pool_addresses.py")
            run_script("data_ingestion.fetch_macroeconomic_data", "fetch_macroeconomic_data.py")
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

    # Phase 3: Pool Analysis & Metrics Calculation
    if "all" in phases or "phase3" in phases:
        logger.info("--- Phase 3: Pool Analysis & Metrics Calculation ---")
        try:
            # Use Spark or standard implementation based on flag
            if use_spark:
                run_spark_job("calculate_metrics")
            else:
                run_script("data_processing.calculate_pool_metrics", "calculate_pool_metrics.py")
            
            run_script("data_processing.apply_pool_grouping", "apply_pool_grouping.py")
            run_script("data_processing.process_icebox_logic", "process_icebox_logic.py")
            # Update allocation snapshots after icebox logic
            run_script("data_processing.update_allocation_snapshots", "update_allocation_snapshots.py")
            logger.info("Phase 3 completed successfully.")
        except Exception as e:
            logger.error(f"Error during Phase 3 (Pool Analysis & Metrics Calculation): {e}")
            sys.exit(1)

    # Phase 4: Forecasting (moved up from original Phase 4)
    if "all" in phases or "phase4" in phases:
        logger.info("--- Phase 4: Forecasting ---")
        try:
            # Use Spark or standard implementation based on flag
            if use_spark:
                run_spark_job("forecast_pools")
            else:
                run_script("forecasting.forecast_pools", "forecast_pools.py")
            
            run_script("forecasting.forecast_gas_fees", "forecast_gas_fees.py")
            logger.info("Phase 4 completed successfully.")
        except Exception as e:
            logger.error(f"Error during Phase 4 (Forecasting): {e}")
            sys.exit(1)

    # Phase 5: Final Filtering & Transaction Processing (moved down from original Phase 3)
    if "all" in phases or "phase5" in phases:
        logger.info("--- Phase 5: Final Filtering & Transaction Processing ---")
        try:
            run_script("data_processing.filter_pools_final", "filter_pools_final.py")
            run_script("data_processing.process_account_transactions", "process_account_transactions.py")
            logger.info("Phase 5 completed successfully.")
        except Exception as e:
            logger.error(f"Error during Phase 5 (Final Filtering & Transaction Processing): {e}")
            sys.exit(1)

    # Phase 6: Asset Allocation (renumbered from original Phase 5)
    if "all" in phases or "phase6" in phases:
        logger.info("--- Phase 6: Asset Allocation ---")
        try:
            run_script("reporting_notification.manage_ledger", "manage_ledger.py")
            run_script("asset_allocation.optimize_allocations", "optimize_allocations.py")
            logger.info("Phase 6 completed successfully.")
        except Exception as e:
            logger.error(f"Error during Phase 6 (Asset Allocation): {e}")
            sys.exit(1)

    # Phase 7: Reporting & Notification (renumbered from original Phase 6)
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
    if use_spark:
        logger.info(f"🔥 Spark mode: ENABLED (used for calculate_pool_metrics and forecast_pools)")
    else:
        logger.info(f"🐍 Spark mode: DISABLED (using standard Python implementation)")
    logger.info("")
    logger.info("📊 PHASE BREAKDOWN:")
    logger.info("  📸 Initial allocation snapshots created (after migrations)")
    if "all" in phases or "phase1" in phases:
        logger.info("  ✅ Phase 1: Initial Data Ingestion")
        logger.info("     • DeFiLlama pools data fetched")
        logger.info("     • DeFiLlama pool addresses fetched")
        logger.info("     • CoinMarketCap OHLCV data fetched")
        logger.info("     • Gas tracker data fetched")
        logger.info("     • Ethplorer account data fetched")
    if "all" in phases or "phase2" in phases:
        logger.info("  ✅ Phase 2: Pre-Filtering & Pool History Ingestion")
        logger.info("     • Pre-filtering applied (no icebox)")
        logger.info("     • Historical data fetched for filtered pools")
    if "all" in phases or "phase3" in phases:
        logger.info("  ✅ Phase 3: Pool Analysis & Metrics Calculation")
        if use_spark:
            logger.info("     • Pool metrics calculated (🔥 Spark)")
        else:
            logger.info("     • Pool metrics calculated")
        logger.info("     • Pool grouping applied")
        logger.info("     • Icebox logic processed")
        logger.info("     • Allocation snapshots updated (post-icebox)")
    if "all" in phases or "phase4" in phases:
        logger.info("  ✅ Phase 4: Forecasting")
        if use_spark:
            logger.info("     • Pool forecasts generated (🔥 Spark + SynapseML LightGBM)")
        else:
            logger.info("     • Pool forecasts generated using forecasted values")
        logger.info("     • Gas fee forecasts generated")
    if "all" in phases or "phase5" in phases:
        logger.info("  ✅ Phase 5: Final Filtering & Transaction Processing")
        logger.info("     • Final filtering completed using forecasted values")
        logger.info("     • Account transactions processed")
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
    # python main_pipeline.py --use-spark
    # python main_pipeline.py --phases phase3 phase4 --use-spark
    import argparse
    parser = argparse.ArgumentParser(description="Run DefiYieldOpt local pipeline.")
    parser.add_argument("--phases", nargs='*', help="Specify phases to run (e.g., phase1 phase3). Use 'all' to run all phases.", default=["all"])
    parser.add_argument("--use-spark", action="store_true", help="Use local PySpark for calculate_pool_metrics and forecast_pools instead of standard Python implementation")
    args = parser.parse_args()
    
    run_pipeline(phases=args.phases, use_spark=args.use_spark)