import json
from datetime import datetime
from database.db_utils import get_db_connection
from config import COINMARKETCAP_API_KEY
from api_clients.coinmarketcap_client import get_historical_ohlcv_data

def fetch_ohlcv_coinmarketcap(api_key, symbols=['BTC', 'ETH']):
    engine = None
    try:
        engine = get_db_connection()
        if not engine:
            print("Could not establish database connection. Exiting.")
            return

        for symbol in symbols:
            # Get historical data for the past 30 days using the API client
            historical_quotes = get_historical_ohlcv_data(symbol=symbol, count=30)
            
            if not historical_quotes:
                print(f"No historical data fetched for {symbol}. Skipping database insertion.")
                continue

            try:
                for quote_data in historical_quotes:
                    quote_usd = quote_data.get('quote', {}).get('USD', {})
                    timestamp_str = quote_usd.get('timestamp')
                    
                    if timestamp_str:
                        data_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        # Directly execute UPSERT query for raw_coinmarketcap_ohlcv
                        # This bypasses the generic insert_raw_data's duplicate check
                        # and uses the specific unique constraint on (data_timestamp, symbol)
                        from sqlalchemy import text
                        from psycopg2 import extras # Import extras for Json type

                        upsert_query = text(f"""
                            INSERT INTO raw_coinmarketcap_ohlcv (data_timestamp, symbol, raw_json_data)
                            VALUES (:data_timestamp, :symbol, :raw_json_data)
                            ON CONFLICT (data_timestamp, symbol) DO UPDATE SET
                                raw_json_data = EXCLUDED.raw_json_data,
                                insertion_timestamp = CURRENT_TIMESTAMP;
                        """)
                        with engine.connect() as conn:
                            with conn.begin():
                                conn.execute(upsert_query, {
                                    "data_timestamp": data_timestamp,
                                    "symbol": symbol,
                                    "raw_json_data": extras.Json(quote_usd)
                                })
                        print(f"Successfully upserted CoinMarketCap OHLCV data for {symbol} on {data_timestamp.date()}.")
                print(f"Successfully fetched and processed CoinMarketCap OHLCV data for {symbol}.")
            except Exception as e: # Catch a broader exception for issues during processing/insertion
                print(f"Error processing or inserting CoinMarketCap OHLCV for {symbol}: {e}")
    finally:
        if engine:
            engine.dispose()

if __name__ == "__main__":
    if not COINMARKETCAP_API_KEY:
        print("COINMARKETCAP_API_KEY environment variable not set in config.py.")
    else:
        fetch_ohlcv_coinmarketcap(COINMARKETCAP_API_KEY)