import pandas as pd
import json
from datetime import datetime, timezone
from database.db_utils import get_db_connection
from psycopg2 import extras
from config import ETHGASTRACKER_API_KEY
from api_clients.ethgastracker_client import get_historical_gas_data_raw_response

def fetch_gas_ethgastracker():
    engine = None
    try:
        engine = get_db_connection()
        if not engine:
            print("Could not establish database connection. Exiting.")
            return

        # Fetch the full raw historical data response using the API client
        raw_response_data = get_historical_gas_data_raw_response()
        
        if not raw_response_data:
            print("No historical gas data fetched. Skipping database insertion.")
            return

        # Extract individual hourly data points from the nested 'data' array
        historical_data_points = raw_response_data.get('data', {}).get('data', [])

        # Insert the entire raw response into raw_ethgastracker_hourly_gas_data
        current_date = datetime.now(timezone.utc).date()
        from sqlalchemy import text

        try:
            with engine.connect() as conn:
                with conn.begin():
                    check_query = text(f"SELECT COUNT(*) FROM raw_ethgastracker_hourly_gas_data WHERE DATE(insertion_timestamp) = :current_date;")
                    result = conn.execute(check_query, {"current_date": current_date})
                    existing_count = result.scalar_one()
                    
                    if existing_count == 0:
                        query = text(f"INSERT INTO raw_ethgastracker_hourly_gas_data (raw_json_data) VALUES (:raw_json_data);")
                        conn.execute(query, {"raw_json_data": extras.Json(raw_response_data)})
                        print("Successfully inserted raw EthGasTracker response.")
                    else:
                        print("Raw data already exists for today in raw_ethgastracker_hourly_gas_data, skipping insertion.")
        except Exception as e:
            print(f"Error inserting data into raw_ethgastracker_hourly_gas_data: {e}")

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
        
        if records_to_insert:
            try:
                with engine.connect() as conn:
                    with conn.begin():
                        columns = records_to_insert[0].keys()
                        placeholders = ', '.join([f':{col}' for col in columns])
                        column_names = ', '.join(columns)
                        
                        query = text(f"""
                            INSERT INTO gas_fees_hourly ({column_names})
                            VALUES ({placeholders});
                        """)

                        processed_records = []
                        for record in records_to_insert:
                            processed_record = {}
                            for key, value in record.items():
                                if isinstance(value, (dict, list)):
                                    processed_record[key] = extras.Json(value)
                                else:
                                    processed_record[key] = value
                            processed_records.append(processed_record)

                        conn.execute(query, processed_records)
                        print(f"Successfully bulk inserted {len(records_to_insert)} records into gas_fees_hourly.")
            except Exception as e:
                print(f"Error during bulk insert into gas_fees_hourly: {e}")
        # Fetch the newly inserted hourly data to calculate daily aggregates
        hourly_df = pd.DataFrame(records_to_insert)
        if not hourly_df.empty:
            hourly_df['timestamp'] = pd.to_datetime(hourly_df['timestamp'], utc=True)
            hourly_df = hourly_df.set_index('timestamp')

            daily_aggregates = hourly_df.resample('D').agg(
                actual_avg_gas_gwei=('gas_price_gwei', 'mean'),
                actual_max_gas_gwei=('gas_price_gwei', 'max') # Assuming max gas is also from gas_price_gwei
            ).dropna()

            daily_records_to_insert = []
            for date, row in daily_aggregates.iterrows():
                daily_records_to_insert.append({
                    'date': date.date(),
                    'actual_avg_gas_gwei': float(row['actual_avg_gas_gwei']),
                    'actual_max_gas_gwei': float(row['actual_max_gas_gwei'])
                })
            
            if daily_records_to_insert:
                try:
                    with engine.connect() as conn:
                        with conn.begin():
                            columns = daily_records_to_insert[0].keys()
                            placeholders = ', '.join([f':{col}' for col in columns])
                            column_names = ', '.join(columns)
                            
                            query = text(f"""
                                INSERT INTO gas_fees_daily ({column_names})
                                VALUES ({placeholders})
                                ON CONFLICT (date) DO UPDATE SET
                                    actual_avg_gas_gwei = EXCLUDED.actual_avg_gas_gwei,
                                    actual_max_gas_gwei = EXCLUDED.actual_max_gas_gwei;
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
                            print(f"Successfully bulk inserted {len(daily_records_to_insert)} records into gas_fees_daily.")
                except Exception as e:
                    print(f"Error during bulk insert into gas_fees_daily: {e}")
            else:
                print("No valid daily gas data points to insert into gas_fees_daily.")
        else:
            print("No valid historical gas data points to insert into gas_fees_hourly.")
    except Exception as e: # Catch a broader exception for issues during API call or processing
        print(f"Error during EthGasTracker data fetch or processing: {e}")
    finally:
        if engine:
            engine.dispose()

if __name__ == "__main__":
    if not ETHGASTRACKER_API_KEY:
        print("ETHGASTRACKER_API_KEY environment variable not set in config.py.")
    else:
        fetch_gas_ethgastracker()