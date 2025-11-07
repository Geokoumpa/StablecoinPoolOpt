import logging
import pandas as pd
from tqdm import tqdm
from database.db_utils import get_db_connection
from data_ingestion.fetch_defillama_pool_history import fetch_defillama_pool_history

logger = logging.getLogger(__name__)

def get_pre_filtered_pool_ids() -> list:
    """
    Fetches pool_ids from pool_daily_metrics that passed pre-filtering for the current date
    and belong to pools that are active (is_active = true).
    """
    conn = get_db_connection()
    query = """
    SELECT DISTINCT pdm.pool_id
    FROM pool_daily_metrics pdm
    JOIN pools p ON pdm.pool_id = p.pool_id
    WHERE pdm.date = CURRENT_DATE 
        AND pdm.is_filtered_out = FALSE
        AND p.is_active = TRUE;
    """
    df = pd.read_sql(query, conn)
    conn.dispose()
    return df['pool_id'].tolist()

def fetch_filtered_pool_histories():
    """
    Fetches historical data for all pools that passed pre-filtering and are active.
    This runs after pre-filtering but before final filtering (icebox).
    """
    logger.info("Starting to fetch filtered pool histories...")
    
    try:
        # Clear the table before fetching new data
        conn = get_db_connection()
        with conn.connect() as connection:
            from sqlalchemy import text
            connection.execute(text("DELETE FROM raw_defillama_pool_history;"))
            connection.commit()
        conn.dispose()
        
        # Get pools that passed pre-filtering and are active
        filtered_pool_ids = get_pre_filtered_pool_ids()
        
        if not filtered_pool_ids:
            logger.warning("No pre-filtered active pools found to fetch historical data for.")
            return
            
        logger.info(f"Found {len(filtered_pool_ids)} pre-filtered active pools to fetch histories for.")
        
        success_count = 0
        error_count = 0
        
        # Fetch historical data for each pre-filtered active pool
        for pool_id in tqdm(filtered_pool_ids, desc="Fetching pre-filtered active pool histories"):
            try:
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