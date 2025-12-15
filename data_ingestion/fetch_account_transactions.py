
import logging
from datetime import datetime, timezone
from config import MAIN_ASSET_HOLDING_ADDRESS
from api_clients.ethplorer_client import get_address_history
from database.repositories.raw_data_repository import RawDataRepository

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
    repo = RawDataRepository()

    try:
        # Fetch transaction history via consolidated Ethplorer client
        try:
            raw_data = get_address_history(address, limit=100)
            if raw_data is None:
                logger.error("Failed to fetch address history from Ethplorer.")
                return

            # Remove duplicates
            unique_transactions = list({t['transactionHash']: t for t in raw_data.get('operations', [])}.values())
            
            current_date = datetime.now(timezone.utc).date()
            
            try:
                if not repo.has_raw_ethplorer_transactions_for_date(current_date):
                    # Insert new data
                    # RawDataRepository requires a list of dicts. 
                    # RawEthplorerAccountTransaction expects 'raw_json_data'
                    
                    data_to_insert = [{
                        'raw_json_data': unique_transactions
                    }]
                    
                    repo.insert_raw_transactions(data_to_insert, source='ethplorer')
                    
                    logger.info(f"Successfully fetched Ethplorer account transactions for {address}.")
                else:
                    logger.info(f"Raw data already exists for today in raw_ethplorer_account_transactions, skipping insertion.")
            except Exception as e:
                logger.error(f"Error inserting data into raw_ethplorer_account_transactions: {e}")
        except Exception as e:
            logger.error(f"Error fetching Ethplorer account transactions: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in fetch_account_transactions: {e}")

if __name__ == "__main__":
    fetch_account_transactions()