import json
import logging
from datetime import datetime, timezone, timedelta
from database.db_utils import get_db_connection
from config import COINMARKETCAP_API_KEY
from api_clients.coinmarketcap_client import get_historical_ohlcv_data

logger = logging.getLogger(__name__)

def fetch_ohlcv_coinmarketcap():
    if not COINMARKETCAP_API_KEY:
        logger.error("COINMARKETCAP_API_KEY not available from config.")
        return
    engine = None
    try:
        engine = get_db_connection()
        if not engine:
            logger.error("Could not establish database connection. Exiting.")
            return

        # Fetch approved tokens from database
        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT token_symbol FROM approved_tokens
                WHERE removed_timestamp IS NULL
            """))
            approved_symbols = [row[0] for row in result.fetchall()]

        # Include BTC and ETH which are not in approved_tokens but still needed
        symbols = ['BTC', 'ETH'] + approved_symbols

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
                from psycopg2 import extras  # Import extras for Json type and execute_values
                # Collect all data for bulk upsert
                bulk_data = []
                for quote_data in all_quotes:
                    quote_usd = quote_data.get('quote', {}).get('USD', {})
                    timestamp_str = quote_data.get('time_close')

                    if timestamp_str and quote_usd:
                        # Ensure data_timestamp is timezone-aware and then convert to date for consistency
                        data_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')).astimezone(timezone.utc).date()
                        bulk_data.append({
                            "data_timestamp": data_timestamp,
                            "symbol": symbol,
                            "raw_json_data": extras.Json(quote_data.get('quote'))
                        })

                # Perform bulk upsert if there's data to insert
                if bulk_data:
                    # Prepare data for bulk insert
                    values = [(item['data_timestamp'], item['symbol'], item['raw_json_data']) for item in bulk_data]

                    # Use psycopg2's execute_values for efficient bulk upsert
                    upsert_query = """
                        INSERT INTO raw_coinmarketcap_ohlcv (data_timestamp, symbol, raw_json_data)
                        VALUES %s
                        ON CONFLICT (data_timestamp, symbol) DO UPDATE SET
                            raw_json_data = EXCLUDED.raw_json_data,
                            insertion_timestamp = CURRENT_TIMESTAMP;
                    """
                    raw_conn = engine.raw_connection()
                    try:
                        cursor = raw_conn.driver_connection.cursor()
                        extras.execute_values(cursor, upsert_query, values)
                        raw_conn.driver_connection.commit()
                    finally:
                        raw_conn.close()
                    logger.info(f"Successfully bulk upserted {len(bulk_data)} CoinMarketCap OHLCV records for {symbol}.")
                else:
                    logger.warning(f"No valid data to upsert for {symbol}.")

                logger.info(f"Successfully fetched and processed CoinMarketCap OHLCV data for {symbol}.")
            except Exception as e:  # Catch a broader exception for issues during processing/insertion
                logger.error(f"Error processing or inserting CoinMarketCap OHLCV for {symbol}: {e}")
    finally:
        if engine:
            engine.dispose()
if __name__ == "__main__":
    fetch_ohlcv_coinmarketcap()