import time
import requests
import json
import os
from datetime import datetime, timezone
from database.db_utils import get_db_connection
from psycopg2 import extras
from sqlalchemy import text

def fetch_defillama_pool_history(pool_id):
    url = f"https://yields.llama.fi/chart/{pool_id}"
    conn = None
    retries = 7
    backoff_factor = 2
    delay = 1  # Start with a 1-second delay
    start_time = time.time()

    print(f"Fetching pool history for: {pool_id}")
    
    for i in range(retries):
        try:
            conn = get_db_connection()
            response = requests.get(url)
            response.raise_for_status()
            raw_data = response.json()
            
            # If the actual data is nested under a 'data' key, extract it
            chart_data = raw_data.get('data', raw_data) if isinstance(raw_data, dict) else raw_data

            # Ensure chart_data is a list
            if not isinstance(chart_data, list):
                print(f"Warning: Expected a list of data points for pool {pool_id}, but got {type(chart_data)}. Skipping.")
                return

            data_points = len(chart_data)
            
            # Prepare records for bulk insertion
            records_to_insert = []
            for data_point in chart_data:
                # Parse the timestamp from the raw data
                if isinstance(data_point, dict) and 'timestamp' in data_point:
                    timestamp = datetime.fromisoformat(data_point['timestamp'].replace('Z', '+00:00'))
                    records_to_insert.append({
                        'pool_id': pool_id,
                        'raw_json_data': data_point,
                        'timestamp': timestamp
                    })
                else:
                    print(f"Warning: Data point missing timestamp for pool {pool_id}. Skipping.")
            
            # Perform bulk insert
            if records_to_insert:
                try:
                    with conn.connect() as connection:
                        with connection.begin():
                            # Dynamically determine columns from the first record's keys
                            columns = records_to_insert[0].keys()
                            
                            # Construct the query with named placeholders for all columns
                            query = text(f"""
                                INSERT INTO raw_defillama_pool_history ({', '.join(columns)})
                                VALUES ({', '.join([f':{col}' for col in columns])});
                            """)

                            # Convert all json-like fields to Json for insertion
                            processed_records = []
                            for record in records_to_insert:
                                processed_record = {}
                                for key, value in record.items():
                                    if isinstance(value, (dict, list)):
                                        processed_record[key] = extras.Json(value)
                                    else:
                                        processed_record[key] = value
                                processed_records.append(processed_record)

                            connection.execute(query, processed_records)
                            print(f"Successfully bulk inserted {len(records_to_insert)} records into raw_defillama_pool_history.")
                except Exception as e:
                    print(f"Error during bulk insert into raw_defillama_pool_history: {e}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Print summary
            print("\n" + "="*50)
            print("📈 POOL HISTORY INGESTION SUMMARY")
            print("="*50)
            print(f"🏊 Pool ID: {pool_id}")
            print(f"🌐 API endpoint: {url}")
            print(f"📊 Data points fetched: {data_points:,}")
            print(f"💾 Data stored in: raw_defillama_pool_history")
            print(f"⏱️  Processing time: {processing_time:.2f}s")
            print(f"🔄 Attempts: {i+1}/{retries}")
            print("="*50)
            return  # Success, exit the loop
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching DeFiLlama pool history for {pool_id}: {e}")
            if i < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= backoff_factor
            else:
                processing_time = time.time() - start_time
                print("\n" + "="*50)
                print("❌ POOL HISTORY INGESTION FAILED")
                print("="*50)
                print(f"🏊 Pool ID: {pool_id}")
                print(f"🌐 API endpoint: {url}")
                print(f"🔄 Failed after {retries} attempts")
                print(f"⏱️  Total time: {processing_time:.2f}s")
                print("="*50)
        except json.JSONDecodeError as e:
            processing_time = time.time() - start_time
            print(f"Error decoding JSON for pool {pool_id}: {e}")
            print("\n" + "="*50)
            print("❌ POOL HISTORY INGESTION FAILED")
            print("="*50)
            print(f"🏊 Pool ID: {pool_id}")
            print(f"📋 Error: JSON decode failure")
            print(f"⏱️  Processing time: {processing_time:.2f}s")
            print("="*50)
            break  # Do not retry on JSON decoding errors
        finally:
            if conn:
                conn.dispose()

if __name__ == "__main__":
    # This script would typically be called with a pool_id from the main pipeline
    # For testing, you can use a sample pool_id
    sample_pool_id = "747c1d2a-c668-4682-b9f9-296708a3dd90" # Example pool ID
    fetch_defillama_pool_history(sample_pool_id)