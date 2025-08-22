import requests
import json
from database.db_utils import get_db_connection
from psycopg2 import extras
from config import ETHERSCAN_API_KEY, MAIN_ASSET_HOLDING_ADDRESS

def fetch_account_data_etherscan(api_key, address):
    base_url = "https://api.etherscan.io/api"

    engine = None
    try:
        engine = get_db_connection()
        if not engine:
            print("Could not establish database connection. Exiting.")
            return

        # Fetch account balance
        balance_params = {
            'module': 'account',
            'action': 'balance',
            'address': address,
            'tag': 'latest',
            'apikey': api_key
        }
        try:
            response = requests.get(base_url, params=balance_params)
            response.raise_for_status()
            raw_data = response.json()
            from datetime import datetime, timezone
            current_date = datetime.now(timezone.utc).date()
            from sqlalchemy import text

            try:
                with engine.connect() as conn:
                    with conn.begin():
                        check_query = text(f"SELECT COUNT(*) FROM raw_etherscan_account_balances WHERE DATE(insertion_timestamp) = :current_date;")
                        result = conn.execute(check_query, {"current_date": current_date})
                        existing_count = result.scalar_one()
                        
                        if existing_count == 0:
                            query = text(f"INSERT INTO raw_etherscan_account_balances (raw_json_data) VALUES (:raw_json_data);")
                            conn.execute(query, {"raw_json_data": extras.Json(raw_data)})
                            print(f"Successfully fetched Etherscan account balance for {address}.")
                        else:
                            print(f"Raw data already exists for today in raw_etherscan_account_balances, skipping insertion.")
            except Exception as e:
                print(f"Error inserting data into raw_etherscan_account_balances: {e}")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching Etherscan account balance: {e}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response for account balance: {e}")

        # Fetch transaction history
        transactions_params = {
            'module': 'account',
            'action': 'txlist',
            'address': address,
            'startblock': 0,
            'endblock': 99999999,
            'sort': 'asc',
            'apikey': api_key
        }
        try:
            response = requests.get(base_url, params=transactions_params)
            response.raise_for_status()
            raw_data = response.json()
            from datetime import datetime, timezone
            current_date = datetime.now(timezone.utc).date()
            from sqlalchemy import text

            try:
                with engine.connect() as conn:
                    with conn.begin():
                        check_query = text(f"SELECT COUNT(*) FROM raw_etherscan_account_transactions WHERE DATE(insertion_timestamp) = :current_date;")
                        result = conn.execute(check_query, {"current_date": current_date})
                        existing_count = result.scalar_one()
                        
                        if existing_count == 0:
                            query = text(f"INSERT INTO raw_etherscan_account_transactions (raw_json_data) VALUES (:raw_json_data);")
                            conn.execute(query, {"raw_json_data": extras.Json(raw_data)})
                            print(f"Successfully fetched Etherscan account transactions for {address}.")
                        else:
                            print(f"Raw data already exists for today in raw_etherscan_account_transactions, skipping insertion.")
            except Exception as e:
                print(f"Error inserting data into raw_etherscan_account_transactions: {e}")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching Etherscan account transactions: {e}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response for account transactions: {e}")
    finally:
        if engine:
            engine.dispose()

if __name__ == "__main__":
    if not ETHERSCAN_API_KEY:
        print("ETHERSCAN_API_KEY environment variable not set in config.py.")
    elif not MAIN_ASSET_HOLDING_ADDRESS:
        print("MAIN_ASSET_HOLDING_ADDRESS environment variable not set in config.py.")
    else:
        fetch_account_data_etherscan(ETHERSCAN_API_KEY, MAIN_ASSET_HOLDING_ADDRESS)