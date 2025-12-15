import pandas as pd
import json
import csv
import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal, InvalidOperation
from database.repositories.gas_fee_repository import GasFeeRepository
from database.repositories.raw_data_repository import RawDataRepository
from api_clients.ethgastracker_client import get_hourly_gas_averages_past_week
import numpy as np

logger = logging.getLogger(__name__)

def fetch_crypto_open_prices_from_raw_data(repo: RawDataRepository, symbols=['ETH', 'BTC']) -> pd.DataFrame:
    """
    Fetches historical daily crypto open prices from raw_coinmarketcap_ohlcv table.
    Returns a DataFrame with date as index and columns for each crypto open price.
    """
    all_prices_dfs = []
    
    for symbol in symbols:
        # Use repository method to get data
        rows = repo.get_crypto_open_prices(symbol)
        if not rows:
            continue
            
        # Rows are tuples: (date, open_price)
        df = pd.DataFrame(rows, columns=['date', 'open_price'])
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            # Ensure the index is timezone-aware and normalized to date-only (midnight UTC)
            df = df.set_index('date')
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

def fetch_existing_gas_fees_daily(repo: GasFeeRepository) -> pd.DataFrame:
    """
    Fetches all existing records from the gas_fees_daily table.
    """
    data_objects = repo.get_all_daily_data()
    
    if not data_objects:
        return pd.DataFrame()
        
    records = []
    for d in data_objects:
        records.append({
            'date': d.date,
            'actual_avg_gas_gwei': float(d.actual_avg_gas_gwei) if d.actual_avg_gas_gwei is not None else None,
            'actual_max_gas_gwei': float(d.actual_max_gas_gwei) if d.actual_max_gas_gwei is not None else None,
            'eth_open': float(d.eth_open) if d.eth_open is not None else None,
            'btc_open': float(d.btc_open) if d.btc_open is not None else None,
            'forecasted_avg_gas_gwei': float(d.forecasted_avg_gas_gwei) if d.forecasted_avg_gas_gwei is not None else None
        })
        
    df = pd.DataFrame(records)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        if df['date'].dt.tz is None:
             df['date'] = df['date'].dt.tz_localize('UTC')
        else:
             df['date'] = df['date'].dt.tz_convert('UTC')
             
        df['date'] = df['date'].dt.normalize()
        df = df.set_index('date')
        
    return df

def wei_to_gwei(wei_value):
    """
    Convert wei to gwei (1 gwei = 10^9 wei)
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
    """
    try:
        parsed_date = datetime.strptime(date_str, "%m/%d/%Y")
        return parsed_date.strftime("%Y-%m-%d")
    except ValueError as e:
        logger.error(f"Error parsing date '{date_str}': {e}")
        return None

def import_historical_gas_data_from_csv(repo: GasFeeRepository, csv_filepath):
    """
    Import historical gas data from CSV file into gas_fees_daily table.
    """
    if not os.path.exists(csv_filepath):
        logger.error(f"CSV file not found: {csv_filepath}")
        return False

    try:
        logger.info(f"Starting import from {csv_filepath}")

        imported_count = 0
        skipped_count = 0
        error_count = 0
        
        batch_records = []

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
                    
                    batch_records.append({
                        "date": formatted_date,
                        "actual_avg_gas_gwei": gwei_value,
                        "actual_max_gas_gwei": None,
                        "eth_open": None, 
                        "btc_open": None
                    })
                    
                    imported_count += 1

                except Exception as e:
                    logger.error(f"Row {row_num}: Error processing row - {e}")
                    error_count += 1
                    continue
        
        if batch_records:
            repo.bulk_upsert_daily_gas(batch_records)

        logger.info(f"\nImport completed successfully!")
        logger.info(f"Records imported: {imported_count}")
        logger.info(f"Records skipped: {skipped_count}")
        logger.info(f"Errors encountered: {error_count}")

        return True

    except Exception as e:
        logger.error(f"Error during import: {e}")
        return False


def fetch_gas_ethgastracker():
    # Initialize repositories
    gas_repo = GasFeeRepository()
    raw_repo = RawDataRepository()

    try:
        # Check if historical data for 6 months ago exists, if not, import from CSV
        six_months_ago = datetime.now(timezone.utc) - timedelta(days=180)
        target_date = six_months_ago.date() # compare as date

        if not gas_repo.has_daily_data_for_date(target_date):
            logger.info(f"No historical data found for {target_date} (6 months ago). Importing from CSV...")
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            csv_filepath = os.path.join(project_root, "AvgGasPrice.csv")
            success = import_historical_gas_data_from_csv(gas_repo, csv_filepath)
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
        current_date_val = datetime.now(timezone.utc).date()
        
        try:
            if not raw_repo.has_raw_gas_data_for_date(current_date_val):
                raw_repo.insert_raw_gas_data([{'raw_json_data': raw_response_data}])
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
                gas_repo.bulk_insert_hourly(records_to_insert)
                logger.info(f"Successfully bulk inserted/updated {len(records_to_insert)} records into gas_fees_hourly.")

                # Fetch the newly inserted hourly data to calculate daily aggregates
                # We can just use the records_to_insert since we just upserted them
                hourly_df = pd.DataFrame(records_to_insert)
                
                # Fetch all existing daily gas data from the database
                existing_gas_fees_daily = fetch_existing_gas_fees_daily(gas_repo)

                # Fetch ETH and BTC open prices from raw_coinmarketcap_ohlcv
                crypto_prices_df = fetch_crypto_open_prices_from_raw_data(raw_repo, symbols=['ETH', 'BTC'])
                
                # Merge crypto prices with existing gas data to enrich historical records
                # Use 'outer' merge to ensure all dates from both sources are included
                enriched_historical_data = pd.merge(crypto_prices_df, existing_gas_fees_daily, left_index=True, right_index=True, how='outer', suffixes=(None, '_old'))
            except Exception as e:
                logger.error(f"Error inserting hourly gas data or processing daily aggregates: {e}")
        
        # Prioritize new crypto prices over old ones if there's a conflict
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
            final_daily_data = daily_aggregates_from_api.combine_first(enriched_historical_data)
        else:
            final_daily_data = enriched_historical_data
            logger.info("No new hourly gas data fetched. Only updating crypto prices for existing daily records.")

        logger.debug("\n--- Debug: final_daily_data before insertion ---")
        if not final_daily_data.empty:
             logger.debug(final_daily_data.head())
             logger.debug(f"Shape: {final_daily_data.shape}")

        daily_records_to_insert = []
        for date, row in final_daily_data.iterrows():
            record = {
                'date': date.date(),
                'actual_avg_gas_gwei': float(row['actual_avg_gas_gwei']) if pd.notna(row['actual_avg_gas_gwei']) else None,
                'actual_max_gas_gwei': float(row['actual_max_gas_gwei']) if pd.notna(row['actual_max_gas_gwei']) else None,
                'eth_open': float(row['eth_open']) if pd.notna(row['eth_open']) else None,
                'btc_open': float(row['btc_open']) if pd.notna(row['btc_open']) else None
                # 'forecasted_avg_gas_gwei' is not updated by this script usually, it's from forecasting.
                # But here we are recombining existing data. So we should preserve it if possible?
                # The combine_first takes existing data.
                # If existing data had forecast, it should be in enriched_historical_data ?
                # Yes, fetch_existing_gas_fees_daily retrieves forecasted_avg_gas_gwei.
            }
            # Add forecasted back if present
            if 'forecasted_avg_gas_gwei' in row and pd.notna(row['forecasted_avg_gas_gwei']):
                 record['forecasted_avg_gas_gwei'] = float(row['forecasted_avg_gas_gwei'])
            
            daily_records_to_insert.append(record)
        
        if daily_records_to_insert:
            try:
                gas_repo.bulk_upsert_daily_gas(daily_records_to_insert)
                logger.info(f"Successfully bulk inserted/updated {len(daily_records_to_insert)} records into gas_fees_daily.")
            except Exception as e:
                logger.error(f"Error during bulk insert/update into gas_fees_daily: {e}")
        else:
            logger.warning("No valid daily gas data points to insert/update into gas_fees_daily.")

    except Exception as e:
        logger.error(f"Error during EthGasTracker data fetch or processing: {e}")

if __name__ == "__main__":
    fetch_gas_ethgastracker()