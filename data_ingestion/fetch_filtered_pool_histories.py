import logging
import pandas as pd
from tqdm import tqdm
from database.db_utils import get_db_connection
from data_ingestion.fetch_defillama_pool_history import fetch_defillama_pool_history

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_pre_filtered_pool_ids() -> list:
    """
    Fetches pool_ids from pool_daily_metrics that passed pre-filtering for the current date.
    """
    conn = get_db_connection()
    query = """
    SELECT DISTINCT pool_id
    FROM pool_daily_metrics
    WHERE date = CURRENT_DATE AND is_filtered_out = FALSE;
    """
    df = pd.read_sql(query, conn)
    conn.dispose()
    return df['pool_id'].tolist()

def fetch_filtered_pool_histories():
    """
    Fetches historical data for all pools that passed pre-filtering.
    This runs after pre-filtering but before final filtering (icebox).
    """
    logging.info("Starting to fetch filtered pool histories...")
    
    try:
        # Clear the table before fetching new data
        conn = get_db_connection()
        with conn.connect() as connection:
            from sqlalchemy import text
            connection.execute(text("DELETE FROM raw_defillama_pool_history;"))
            connection.commit()
        conn.dispose()
        
        # Get pools that passed pre-filtering
        filtered_pool_ids = get_pre_filtered_pool_ids()
        
        if not filtered_pool_ids:
            logging.warning("No pre-filtered pools found to fetch historical data for.")
            return
            
        logging.info(f"Found {len(filtered_pool_ids)} pre-filtered pools to fetch histories for.")
        
        success_count = 0
        error_count = 0
        
        # Fetch historical data for each pre-filtered pool
        for pool_id in tqdm(filtered_pool_ids, desc="Fetching pre-filtered pool histories"):
            try:
                fetch_defillama_pool_history(pool_id)
                success_count += 1
            except Exception as e:
                logging.error(f"Error fetching history for pre-filtered pool {pool_id}: {e}")
                error_count += 1
                # Continue with next pool instead of failing entire process
        
        logging.info(f"Filtered pool histories fetch completed.")
        logging.info(f"Success: {success_count}, Errors: {error_count}")
        
    except Exception as e:
        logging.error(f"Error during filtered pool histories fetch: {e}")
        raise

if __name__ == "__main__":
    fetch_filtered_pool_histories()