import pandas as pd
import json
import csv
import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal, InvalidOperation
from database.db_utils import get_db_connection
from psycopg2 import extras
from api_clients.ethgastracker_client import get_hourly_gas_averages_past_week
from sqlalchemy import text
import numpy as np # Import numpy

logger = logging.getLogger(__name__)

def fetch_crypto_open_prices_from_raw_data(engine, symbols=['ETH', 'BTC']) -> pd.DataFrame:
    """
    Fetches historical daily crypto open prices from raw_coinmarketcap_ohlcv table.
    Returns a DataFrame with date as index and columns for each crypto open price.
    """
    all_prices_dfs = []
    with engine.connect() as conn:
        for symbol in symbols:
            query = text("""
                WITH daily_data AS (
                    SELECT
                        data_timestamp,
                        (raw_json_data->'USD'->>'close')::numeric AS close_price
                    FROM
                        raw_coinmarketcap_ohlcv
                    WHERE
                        symbol = :symbol
                    UNION ALL
                    -- Ensure today's date is included for the LAG function to work correctly for today
                    SELECT CAST(CURRENT_DATE AS TIMESTAMP), NULL
                )
                SELECT
                    data_timestamp AS date,
                    LAG(close_price, 1) OVER (ORDER BY data_timestamp) AS open_price
                FROM
                    daily_data
                ORDER BY
                    data_timestamp;
            """)
            df = pd.read_sql(query, conn, params={"symbol": symbol}, parse_dates=['date'], index_col='date')
            if not df.empty:
                # Ensure the index is timezone-aware and normalized to date-only (midnight UTC)
                df.index = df.index.tz_localize('UTC').normalize() if df.index.tz is None else df.index.tz_convert('UTC').normalize()
                df = df.rename(columns={'open_price': f'{symbol.lower()}_open'})
                all_prices_dfs.append(df)

    if not all_prices_dfs:
        return pd.DataFrame()

    # Merge all crypto price dataframes
    merged_crypto_prices = all_prices_dfs[0]
    for i in range(1, len(all_prices_dfs)):
        merged_crypto_prices = pd.merge(merged_crypto_prices, all_prices_dfs[i], left_index=True, right_index=True, how='outer')

    return merged_crypto_prices
def fetch_existing_gas_fees_daily(engine) -> pd.DataFrame:
    """
    Fetches all existing records from the gas_fees_daily table.
    """
    query = text("""
        SELECT
            date,
            actual_avg_gas_gwei,
            actual_max_gas_gwei,
            eth_open,
            btc_open,
            forecasted_avg_gas_gwei
        FROM
            gas_fees_daily
        ORDER BY
            date;
    """)
    df = pd.read_sql(query, engine, parse_dates=['date'], index_col='date')
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
    df.index = df.index.normalize() # Ensure date-only index
    return df


def wei_to_gwei(wei_value):
    """
    Convert wei to gwei (1 gwei = 10^9 wei)

    Args:
        wei_value (str): Gas price in wei as string

    Returns:
        Decimal: Gas price in gwei, rounded to 2 decimal places
    """
    try:
        if not wei_value or wei_value == "0":
            return Decimal("0.00")

        wei_decimal = Decimal(wei_value)
        gwei_decimal = wei_decimal / Decimal("1000000000")  # 10^9
        return gwei_decimal.quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError) as e:
        logger.error(f"Error converting wei to gwei for value '{wei_value}': {e}")
        return Decimal("0.00")


def parse_date(date_str):
    """
    Parse date from MM/DD/YYYY format to YYYY-MM-DD format

    Args:
        date_str (str): Date in MM/DD/YYYY format

    Returns:
        str: Date in YYYY-MM-DD format, or None if parsing fails
    """
    try:
        parsed_date = datetime.strptime(date_str, "%m/%d/%Y")
        return parsed_date.strftime("%Y-%m-%d")
    except ValueError as e:
        logger.error(f"Error parsing date '{date_str}': {e}")
        return None


def check_historical_data_exists(engine, target_date):
    """
    Check if historical data exists for the target date in gas_fees_daily table.

    Args:
        engine: SQLAlchemy engine
        target_date (str): Date in YYYY-MM-DD format

    Returns:
        bool: True if data exists, False otherwise
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT COUNT(*) FROM gas_fees_daily WHERE date = :date"),
                {"date": target_date}
            )
            count = result.scalar_one()
            return count > 0
    except Exception as e:
        logger.error(f"Error checking historical data existence: {e}")
        return False


def import_historical_gas_data_from_csv(engine, csv_filepath):
    """
    Import historical gas data from CSV file into gas_fees_daily table.
    Only imports actual_avg_gas_gwei column.

    Args:
        engine: SQLAlchemy engine
        csv_filepath (str): Path to the AvgGasPrice.csv file

    Returns:
        bool: True if import was successful, False otherwise
    """
    if not os.path.exists(csv_filepath):
        logger.error(f"CSV file not found: {csv_filepath}")
        return False

    try:
        logger.info(f"Starting import from {csv_filepath}")

        imported_count = 0
        skipped_count = 0
        error_count = 0

        with engine.begin() as conn:
            # Open and read the CSV file
            with open(csv_filepath, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)

                for row_num, row in enumerate(reader, start=2):  # Start at 2 because row 1 is header
                    try:
                        # Extract data from CSV row
                        date_str = row.get('Date(UTC)', '').strip()
                        wei_value = row.get('Value (Wei)', '').strip()

                        if not date_str or not wei_value:
                            logger.warning(f"Row {row_num}: Missing date or value, skipping")
                            skipped_count += 1
                            continue

                        # Parse and convert data
                        formatted_date = parse_date(date_str)
                        if not formatted_date:
                            logger.warning(f"Row {row_num}: Could not parse date '{date_str}', skipping")
                            skipped_count += 1
                            continue

                        gwei_value = wei_to_gwei(wei_value)

                        # Insert into database (using ON CONFLICT to handle duplicates)
                        conn.execute(
                            text("""
                                INSERT INTO gas_fees_daily (date, actual_avg_gas_gwei)
                                VALUES (:date, :actual_avg_gas_gwei)
                                ON CONFLICT (date) DO UPDATE SET
                                    actual_avg_gas_gwei = EXCLUDED.actual_avg_gas_gwei
                            """),
                            {
                                "date": formatted_date,
                                "actual_avg_gas_gwei": gwei_value
                            }
                        )

                        imported_count += 1

                        # Progress indicator
                        if imported_count % 100 == 0:
                            logger.info(f"Imported {imported_count} records...")

                    except Exception as e:
                        logger.error(f"Row {row_num}: Error processing row - {e}")
                        error_count += 1
                        continue

        logger.info(f"\nImport completed successfully!")
        logger.info(f"Records imported: {imported_count}")
        logger.info(f"Records skipped: {skipped_count}")
        logger.info(f"Errors encountered: {error_count}")

        return True

    except Exception as e:
        logger.error(f"Error during import: {e}")
        return False


def fetch_gas_ethgastracker():
    engine = None
    try:
        engine = get_db_connection()
        if not engine:
            logger.error("Could not establish database connection. Exiting.")
            return

        # Check if historical data for 6 months ago exists, if not, import from CSV
        six_months_ago = datetime.now(timezone.utc) - timedelta(days=180)
        target_date = six_months_ago.strftime("%Y-%m-%d")

        if not check_historical_data_exists(engine, target_date):
            logger.info(f"No historical data found for {target_date} (6 months ago). Importing from CSV...")
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            csv_filepath = os.path.join(project_root, "AvgGasPrice.csv")
            success = import_historical_gas_data_from_csv(engine, csv_filepath)
            if success:
                logger.info("Historical data import completed successfully.")
            else:
                logger.warning("Historical data import failed, but continuing with current data fetch.")
        else:
            logger.info(f"Historical data for {target_date} already exists. Skipping CSV import.")

        # Fetch the full raw historical data response using the API client
        raw_response_data = get_hourly_gas_averages_past_week()
        logger.debug(f"Debug: raw_response_data content: {raw_response_data}")
        
        if not raw_response_data:
            logger.warning("No historical gas data fetched. Skipping database insertion.")
            return

        # Extract individual hourly data points from the nested 'data' array
        historical_data_points = raw_response_data.get('data', {}).get('data', [])
        logger.debug(f"Debug: historical_data_points content: {historical_data_points}")

        # Insert the entire raw response into raw_ethgastracker_hourly_gas_data
        current_date = datetime.now(timezone.utc).date()
        

        try:
            with engine.connect() as conn:
                with conn.begin():
                    check_query = text(f"SELECT COUNT(*) FROM raw_ethgastracker_hourly_gas_data WHERE DATE(insertion_timestamp) = :current_date;")
                    result = conn.execute(check_query, {"current_date": current_date})
                    existing_count = result.scalar_one()
                    
                    if existing_count == 0:
                        query = text(f"INSERT INTO raw_ethgastracker_hourly_gas_data (raw_json_data) VALUES (:raw_json_data);")
                        conn.execute(query, {"raw_json_data": extras.Json(raw_response_data)})
                        logger.info("Successfully inserted raw EthGasTracker response.")
                    else:
                        logger.info("Raw data already exists for today in raw_ethgastracker_hourly_gas_data, skipping insertion.")
        except Exception as e:
            logger.error(f"Error inserting data into raw_ethgastracker_hourly_gas_data: {e}")

        # Prepare records for bulk insert into gas_fees_hourly
        records_to_insert = []
        for data_point in historical_data_points:
            # Ensure all necessary fields are present and converted
            timestamp = datetime.fromisoformat(data_point['period'].replace('Z', '+00:00')) if 'period' in data_point else None
            base_fee = data_point.get('baseFee')

            if timestamp and base_fee is not None:
                gas_price_gwei = base_fee # baseFee is already in Gwei

                records_to_insert.append({
                    'timestamp': timestamp,
                    'gas_price_gwei': gas_price_gwei
                })
        logger.debug(f"Debug: records_to_insert content: {records_to_insert}")

        hourly_df = pd.DataFrame() # Initialize hourly_df
        enriched_historical_data = pd.DataFrame() # Initialize enriched_historical_data

        if records_to_insert:
            try:
                with engine.connect() as conn:
                    with conn.begin():
                        # Fetch the newly inserted hourly data to calculate daily aggregates
                        hourly_df = pd.DataFrame(records_to_insert)
                        
                        # Fetch all existing daily gas data from the database
                        existing_gas_fees_daily = fetch_existing_gas_fees_daily(engine)

                        # Fetch ETH and BTC open prices from raw_coinmarketcap_ohlcv
                        crypto_prices_df = fetch_crypto_open_prices_from_raw_data(engine, symbols=['ETH', 'BTC'])
                        
                        # Merge crypto prices with existing gas data to enrich historical records
                        # Use 'outer' merge to ensure all dates from both sources are included
                        enriched_historical_data = pd.merge(crypto_prices_df, existing_gas_fees_daily, left_index=True, right_index=True, how='outer', suffixes=(None, '_old'))
            except Exception as e:
                logger.error(f"Error inserting hourly gas data or processing daily aggregates: {e}")
        
        # Prioritize new crypto prices over old ones if there's a conflict
        # This block should only execute if enriched_historical_data was successfully created
        if not enriched_historical_data.empty:
            for col in ['eth_open', 'btc_open']:
                if f'{col}_old' in enriched_historical_data.columns:
                    enriched_historical_data[col] = enriched_historical_data[col].fillna(enriched_historical_data[f'{col}_old'])
                    enriched_historical_data = enriched_historical_data.drop(columns=[f'{col}_old'])
            
            # Forward fill any remaining gaps in crypto prices
            enriched_historical_data['eth_open'] = enriched_historical_data['eth_open'].ffill()
            enriched_historical_data['btc_open'] = enriched_historical_data['btc_open'].ffill()

        # If there's new hourly data, aggregate it to daily
        if not hourly_df.empty:
            hourly_df['timestamp'] = pd.to_datetime(hourly_df['timestamp'], utc=True)
            hourly_df = hourly_df.set_index('timestamp')

            # Aggregate up to today's date
            today_utc = pd.Timestamp.now(tz='UTC').normalize()
            daily_aggregates_from_api = hourly_df.resample('D').agg(
                actual_avg_gas_gwei=('gas_price_gwei', 'mean'),
                actual_max_gas_gwei=('gas_price_gwei', 'max')
            )
            
            # Set today's gas fees to NaN as they are incomplete
            if today_utc in daily_aggregates_from_api.index:
                daily_aggregates_from_api.loc[today_utc, ['actual_avg_gas_gwei', 'actual_max_gas_gwei']] = np.nan
            daily_aggregates_from_api.index = daily_aggregates_from_api.index.normalize()

            # Combine the enriched historical data with the new daily aggregates from API
            # New API data takes precedence for actual_avg_gas_gwei and actual_max_gas_gwei
            # Existing crypto prices are retained if not provided by new API data (which it won't be)
            final_daily_data = daily_aggregates_from_api.combine_first(enriched_historical_data)
        else:
            final_daily_data = enriched_historical_data
            logger.info("No new hourly gas data fetched. Only updating crypto prices for existing daily records.")

        logger.debug("\n--- Debug: final_daily_data before insertion ---")
        logger.debug(final_daily_data.head())
        logger.debug(final_daily_data.tail())
        logger.debug(f"Shape: {final_daily_data.shape}")
        logger.debug(f"Date range: {final_daily_data.index.min()} to {final_daily_data.index.max()}")
        logger.debug(f"Null counts in final_daily_data:\n{final_daily_data.isnull().sum()}")

        daily_records_to_insert = []
        for date, row in final_daily_data.iterrows():
            record = {
                'date': date.date(),
                'actual_avg_gas_gwei': float(row['actual_avg_gas_gwei']) if pd.notna(row['actual_avg_gas_gwei']) else None,
                'actual_max_gas_gwei': float(row['actual_max_gas_gwei']) if pd.notna(row['actual_max_gas_gwei']) else None,
                'eth_open': float(row['eth_open']) if pd.notna(row['eth_open']) else None,
                'btc_open': float(row['btc_open']) if pd.notna(row['btc_open']) else None,
                'forecasted_avg_gas_gwei': float(row['forecasted_avg_gas_gwei']) if 'forecasted_avg_gas_gwei' in row and pd.notna(row['forecasted_avg_gas_gwei']) else None
            }
            daily_records_to_insert.append(record)
        
        if daily_records_to_insert:
            try:
                with engine.connect() as conn:
                    with conn.begin():
                        columns = daily_records_to_insert[0].keys()
                        placeholders = ', '.join([f':{col}' for col in columns])
                        column_names = ', '.join(columns)
                        
                        # Dynamically build the ON CONFLICT DO UPDATE SET clause
                        update_set_clauses = []
                        for col in columns:
                            if col != 'date': # 'date' is the primary key for ON CONFLICT
                                update_set_clauses.append(f"{col} = EXCLUDED.{col}")
                        
                        update_clause_str = ", ".join(update_set_clauses)

                        query = text(f"""
                            INSERT INTO gas_fees_daily ({column_names})
                            VALUES ({placeholders})
                            ON CONFLICT (date) DO UPDATE SET
                                {update_clause_str};
                        """)

                        processed_records = []
                        for record in daily_records_to_insert:
                            processed_record = {}
                            for key, value in record.items():
                                if isinstance(value, (dict, list)):
                                    processed_record[key] = extras.Json(value)
                                else:
                                    processed_record[key] = value
                            processed_records.append(processed_record)

                        conn.execute(query, processed_records)
                        logger.info(f"Successfully bulk inserted/updated {len(daily_records_to_insert)} records into gas_fees_daily.")
            except Exception as e:
                logger.error(f"Error during bulk insert/update into gas_fees_daily: {e}")
        else:
            logger.warning("No valid daily gas data points to insert/update into gas_fees_daily.")
    except Exception as e: # Catch a broader exception for issues during API call or processing
        logger.error(f"Error during EthGasTracker data fetch or processing: {e}")
    finally:
        if engine:
            engine.dispose()

if __name__ == "__main__":
    fetch_gas_ethgastracker()