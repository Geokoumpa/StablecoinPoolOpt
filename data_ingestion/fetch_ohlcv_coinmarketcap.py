import json
import logging
from datetime import datetime, timezone
from config import COINMARKETCAP_API_KEY
from api_clients.coinmarketcap_client import get_historical_ohlcv_data
from database.repositories.raw_data_repository import RawDataRepository
from database.repositories.token_repository import TokenRepository
from psycopg2 import extras

logger = logging.getLogger(__name__)

def fetch_ohlcv_coinmarketcap():
    if not COINMARKETCAP_API_KEY:
        logger.error("COINMARKETCAP_API_KEY not available from config.")
        return
    
    # Initialize repositories
    raw_repo = RawDataRepository()
    token_repo = TokenRepository()

    try:
        # Fetch approved tokens from database
        approved_token_objects = token_repo.get_approved_tokens()
        approved_symbol_list = [t.token_symbol for t in approved_token_objects]

        # Include BTC and ETH which are not in approved_tokens but still needed
        symbols = ['BTC', 'ETH'] + approved_symbol_list

        if not symbols:
            logger.warning("No symbols to fetch. Exiting.")
            return

        logger.info(f"Fetching OHLCV data for symbols: {symbols}")

        for symbol in symbols:
            logger.info(f"Fetching up to 365 days of OHLCV data for {symbol}.")
            all_quotes = get_historical_ohlcv_data(symbol=symbol, count=365)
            
            if not all_quotes:
                logger.warning(f"No historical data returned from API for {symbol}. Skipping database insertion.")
                continue

            try:
                # Collect all data for bulk upsert
                bulk_data = []
                for quote_data in all_quotes:
                    quote_usd = quote_data.get('quote', {}).get('USD', {})
                    timestamp_str = quote_data.get('time_close')

                    if timestamp_str and quote_usd:
                        # Ensure data_timestamp is timezone-aware and then convert to date for consistency
                        data_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')).astimezone(timezone.utc).date()
                        
                        # Use extras.Json for JSON serialization if needed, 
                        # but SQLAlchemy handles dicts for JSONB columns if passed correctly.
                        # BaseRepository.execute_bulk_values uses psycopg2 directly, which needs extras.Json or json.dumps.
                        # Wait, BaseRepository calls execute_values.
                        # Reviewing BaseRepository using psycopg2.extras.execute_values:
                        # If we pass a dict to %s in psycopg2, it might need casting to JSON using extras.Json or stringifying.
                        # The original script used `extras.Json`.
                        # However, RawDataRepository code:
                        # values = [(..., d.get('raw_json_data'), ...)]
                        # If d['raw_json_data'] is a dict, psycopg2 might complain unless we adapt it.
                        # SQLAlchemy session.bulk_insert_mappings handles it. 
                        # But RawDataRepository.insert_ohlcv_data uses `self.execute_bulk_values`.
                        # And `execute_bulk_values` uses `psycopg2.extras.execute_values`.
                        # So we DO need `extras.Json` for the raw_json_data field values 
                        # OR `RawDataRepository` should handle it.
                        
                        # Let's check RawDataRepository.insert_ohlcv_data again.
                        # It takes `d.get('raw_json_data')` directly.
                        # The caller (this script) needs to ensure it's compatible with psycopg2 if `execute_bulk_values` is used.
                        # BUT, usually repositories abstract this detail.
                        # If I pass a dict, psycopg2 Adapt won't automatically make it JSON unless registered.
                        # I should probably pass `extras.Json` wrapper in the dict value, or a json string.
                        # Or checking `RawDataRepository` methods again.
                        
                        bulk_data.append({
                            "data_timestamp": data_timestamp,
                            "symbol": symbol,
                            "raw_json_data": extras.Json(quote_data.get('quote')) 
                        })

                # Perform bulk upsert if there's data to insert
                if bulk_data:
                    raw_repo.insert_ohlcv_data(bulk_data)
                    logger.info(f"Successfully bulk upserted {len(bulk_data)} CoinMarketCap OHLCV records for {symbol}.")
                else:
                    logger.warning(f"No valid data to upsert for {symbol}.")

                logger.info(f"Successfully fetched and processed CoinMarketCap OHLCV data for {symbol}.")
            except Exception as e:
                logger.error(f"Error processing or inserting CoinMarketCap OHLCV for {symbol}: {e}")
                
    except Exception as e:
         logger.error(f"Unexpected error in fetch_ohlcv_coinmarketcap: {e}")

if __name__ == "__main__":
    fetch_ohlcv_coinmarketcap()