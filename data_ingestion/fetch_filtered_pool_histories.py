import logging
import pandas as pd
from tqdm import tqdm
from datetime import date
from data_ingestion.fetch_defillama_pool_history import fetch_defillama_pool_history
from database.repositories.pool_metrics_repository import PoolMetricsRepository
from database.repositories.raw_data_repository import RawDataRepository

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

def fetch_filtered_pool_histories():
    """
    Fetches historical data for all pools that passed pre-filtering and are active.
    This runs after pre-filtering but before final filtering (icebox).
    """
    logger.info("Starting to fetch filtered pool histories...")
    
    # Initialize repositories
    metrics_repo = PoolMetricsRepository()
    raw_repo = RawDataRepository()
    
    try:
        # Clear the table before fetching new data
        logger.info("Clearing raw_defillama_pool_history table...")
        raw_repo.clear_raw_pool_history()
        
        # Get pools that passed pre-filtering and are active
        filtered_pool_ids = metrics_repo.get_active_filtered_pool_ids(date.today(), is_filtered_out=False)
        
        if not filtered_pool_ids:
            logger.warning("No pre-filtered active pools found to fetch historical data for.")
            return
            
        logger.info(f"Found {len(filtered_pool_ids)} pre-filtered active pools to fetch histories for.")
        
        success_count = 0
        error_count = 0
        
        # Fetch historical data for each pre-filtered active pool
        for pool_id in tqdm(filtered_pool_ids, desc="Fetching pre-filtered active pool histories"):
            try:
                # This function handles its own repository instantiation for RawDataRepository
                # to insert history. We could potentially refactor it to accept a repo instance,
                # but currently it works as a standalone unit.
                fetch_defillama_pool_history(pool_id)
                success_count += 1
            except Exception as e:
                logger.error(f"Error fetching history for pre-filtered active pool {pool_id}: {e}")
                error_count += 1
                # Continue with next pool instead of failing entire process
        
        logger.info(f"Filtered pool histories fetch completed.")
        logger.info(f"Success: {success_count}, Errors: {error_count}")
        
    except Exception as e:
        logger.error(f"Error during filtered pool histories fetch: {e}")
        raise

if __name__ == "__main__":
    fetch_filtered_pool_histories()