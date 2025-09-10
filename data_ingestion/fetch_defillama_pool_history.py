import time
import requests
import json
import os
import logging
from datetime import datetime, timezone
from database.db_utils import get_db_connection
from psycopg2 import extras
from sqlalchemy import text

logger = logging.getLogger(__name__)

def fetch_defillama_pool_history(pool_id):
    url = f"https://yields.llama.fi/chart/{pool_id}"
    conn = None
    retries = 7
    backoff_factor = 2
    delay = 1  # Start with a 1-second delay
    start_time = time.time()

    logger.info(f"Fetching pool history for: {pool_id}")
    
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
                logger.warning(f"Expected a list of data points for pool {pool_id}, but got {type(chart_data)}. Skipping.")
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
                    logger.warning(f"Data point missing timestamp for pool {pool_id}. Skipping.")
            
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
                            logger.info(f"Successfully bulk inserted {len(records_to_insert)} records into raw_defillama_pool_history.")
                except Exception as e:
                    logger.error(f"Error during bulk insert into raw_defillama_pool_history: {e}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Print summary
            logger.info("\n" + "="*50)
            logger.info("📈 POOL HISTORY INGESTION SUMMARY")
            logger.info("="*50)
            logger.info(f"🏊 Pool ID: {pool_id}")
            logger.info(f"🌐 API endpoint: {url}")
            logger.info(f"📊 Data points fetched: {data_points:,}")
            logger.info(f"💾 Data stored in: raw_defillama_pool_history")
            logger.info(f"⏱️  Processing time: {processing_time:.2f}s")
            logger.info(f"🔄 Attempts: {i+1}/{retries}")
            logger.info("="*50)
            return  # Success, exit the loop
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching DeFiLlama pool history for {pool_id}: {e}")
            if i < retries - 1:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= backoff_factor
            else:
                processing_time = time.time() - start_time
                logger.error("\n" + "="*50)
                logger.error("❌ POOL HISTORY INGESTION FAILED")
                logger.error("="*50)
                logger.error(f"🏊 Pool ID: {pool_id}")
                logger.error(f"🌐 API endpoint: {url}")
                logger.error(f"🔄 Failed after {retries} attempts")
                logger.error(f"⏱️  Total time: {processing_time:.2f}s")
                logger.error("="*50)
        except json.JSONDecodeError as e:
            processing_time = time.time() - start_time
            logger.error(f"Error decoding JSON for pool {pool_id}: {e}")
            logger.error("\n" + "="*50)
            logger.error("❌ POOL HISTORY INGESTION FAILED")
            logger.error("="*50)
            logger.error(f"🏊 Pool ID: {pool_id}")
            logger.error(f"📋 Error: JSON decode failure")
            logger.error(f"⏱️  Processing time: {processing_time:.2f}s")
            logger.error("="*50)
            break  # Do not retry on JSON decoding errors
        finally:
            if conn:
                conn.dispose()

if __name__ == "__main__":
    # This script would typically be called with a pool_id from the main pipeline
    # For testing, you can use a sample pool_id
    sample_pool_id = "747c1d2a-c668-4682-b9f9-296708a3dd90" # Example pool ID
    fetch_defillama_pool_history(sample_pool_id)