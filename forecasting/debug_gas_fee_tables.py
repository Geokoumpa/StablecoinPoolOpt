from database.db_utils import get_db_connection

def print_table_row_counts():
    engine = get_db_connection()
    with engine.connect() as conn:
        from sqlalchemy import text
        gas_count = conn.execute(text("SELECT COUNT(*) FROM gas_fees_daily;")).scalar()
        eth_count = conn.execute(text("SELECT COUNT(*) FROM raw_coinmarketcap_ohlcv WHERE symbol = 'ETH';")).scalar()
        btc_count = conn.execute(text("SELECT COUNT(*) FROM raw_coinmarketcap_ohlcv WHERE symbol = 'BTC';")).scalar()
        print(f"Rows in gas_fees_daily: {gas_count}")
        print(f"Rows in raw_coinmarketcap_ohlcv for ETH: {eth_count}")
        print(f"Rows in raw_coinmarketcap_ohlcv for BTC: {btc_count}")

if __name__ == "__main__":
    print_table_row_counts()