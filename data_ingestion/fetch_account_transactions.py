import json
import logging
from database.db_utils import get_db_connection
from psycopg2 import extras
from config import MAIN_ASSET_HOLDING_ADDRESS
from api_clients.ethplorer_client import get_address_history

# Configure module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def fetch_account_transactions():
    if not MAIN_ASSET_HOLDING_ADDRESS:
        logger.error("MAIN_ASSET_HOLDING_ADDRESS not available from config.")
        return
    
    address = MAIN_ASSET_HOLDING_ADDRESS
    engine = None
    try:
        engine = get_db_connection()
        if not engine:
            logger.error("Could not establish database connection. Exiting.")
            return

        # Fetch transaction history via consolidated Ethplorer client
        try:
            raw_data = get_address_history(address, limit=100)
            if raw_data is None:
                logger.error("Failed to fetch address history from Ethplorer.")
                return

            # Remove duplicates
            unique_transactions = {t['transactionHash']: t for t in raw_data.get('operations', [])}.values()
            
            from datetime import datetime, timezone
            current_date = datetime.now(timezone.utc).date()
            from sqlalchemy import text

            try:
                with engine.connect() as conn:
                    with conn.begin():
                        check_query = text(f"SELECT COUNT(*) FROM raw_ethplorer_account_transactions WHERE DATE(insertion_timestamp) = :current_date;")
                        result = conn.execute(check_query, {"current_date": current_date})
                        existing_count = result.scalar_one()
                        
                        if existing_count == 0:
                            query = text(f"INSERT INTO raw_ethplorer_account_transactions (raw_json_data) VALUES (:raw_json_data);")
                            conn.execute(query, {"raw_json_data": extras.Json(list(unique_transactions))})
                            logger.info(f"Successfully fetched Ethplorer account transactions for {address}.")
                        else:
                            logger.info(f"Raw data already exists for today in raw_ethplorer_account_transactions, skipping insertion.")
            except Exception as e:
                logger.error(f"Error inserting data into raw_ethplorer_account_transactions: {e}")
        except Exception as e:
            logger.error(f"Error fetching Ethplorer account transactions: {e}")
    finally:
        if engine:
            engine.dispose()

if __name__ == "__main__":
    fetch_account_transactions()