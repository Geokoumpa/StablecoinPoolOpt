import requests
import json
from datetime import datetime, timezone
from database.db_utils import get_db_connection
from psycopg2.extras import execute_values

def insert_raw_defillama_pools(engine, raw_json_data):
    from sqlalchemy import text
    from datetime import datetime, timezone
    from psycopg2 import extras

    current_date = datetime.now(timezone.utc).date()

    try:
        with engine.connect() as conn:
            with conn.begin():
                # Check for existing data for today
                check_query = text(f"SELECT COUNT(*) FROM raw_defillama_pools WHERE DATE(insertion_timestamp) = :current_date;")
                result = conn.execute(check_query, {"current_date": current_date})
                existing_count = result.scalar_one()
                
                if existing_count == 0:
                    # Insert new data
                    query = text(f"INSERT INTO raw_defillama_pools (raw_json_data) VALUES (:raw_json_data);")
                    conn.execute(query, {"raw_json_data": extras.Json(raw_json_data)})
                    print(f"Successfully inserted raw data into raw_defillama_pools.")
                else:
                    print(f"Raw data already exists for today in raw_defillama_pools, skipping insertion.")
    except Exception as e:
        print(f"Error inserting data into raw_defillama_pools: {e}")

def update_pools_metadata_bulk(conn, pools_data):
    """
    Parses raw pool data and bulk inserts or updates it into the 'pools' master table.
    """
    values_to_insert = []
    for pool_data in pools_data:
        pool_id = pool_data.get('pool')
        chain = pool_data.get('chain')
        project = pool_data.get('project')
        symbol = pool_data.get('symbol')
        tvl_usd = pool_data.get('tvlUsd')
        apy = pool_data.get('apy')

        if not all([pool_id, chain, project, symbol]):
            print(f"Skipping pool with missing essential data: {pool_data}")
            continue
        
        values_to_insert.append((
            pool_id, symbol, chain, project, symbol, tvl_usd, apy, datetime.now(timezone.utc)
        ))

    if not values_to_insert:
        print("No valid pools to insert after filtering.")
        return

    try:
        with conn.begin() as connection:
            update_query = """
                INSERT INTO pools (
                    pool_id, name, chain, protocol, symbol, tvl, apy, last_updated
                ) VALUES %s
                ON CONFLICT (pool_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    chain = EXCLUDED.chain,
                    protocol = EXCLUDED.protocol,
                    symbol = EXCLUDED.symbol,
                    tvl = EXCLUDED.tvl,
                    apy = EXCLUDED.apy,
                    last_updated = EXCLUDED.last_updated;
            """
            execute_values(
                connection.connection.cursor(),
                update_query,
                values_to_insert,
                template=None,
                page_size=100
            )
        print(f"Successfully inserted/updated {len(values_to_insert)} pools.")
    except Exception as e:
        print(f"Error during bulk update of pools metadata: {e}")

def fetch_defillama_pools():
    url = "https://yields.llama.fi/pools"
    engine = None
    try:
        engine = get_db_connection()
        if not engine:
            print("Could not establish database connection. Exiting.")
            return

        print("Fetching DeFiLlama pools data...")
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        raw_data = response.json()

        # Store raw JSON data
        insert_raw_defillama_pools(engine, raw_data)

        # Parse and update pools metadata in bulk
        pools_data = raw_data.get('data', [])
        total_pools_from_api = len(pools_data) if pools_data else 0
        
        processed_pools = 0
        if pools_data:
            # Count valid pools before processing
            valid_pools = [pool for pool in pools_data if all([
                pool.get('pool'), pool.get('chain'), pool.get('project'), pool.get('symbol')
            ])]
            processed_pools = len(valid_pools)
            update_pools_metadata_bulk(engine, pools_data)

        # Print detailed summary
        print("\n" + "="*60)
        print("📥 DEFILLAMA POOLS INGESTION SUMMARY")
        print("="*60)
        print(f"🌐 API endpoint: {url}")
        print(f"📊 Total pools from API: {total_pools_from_api:,}")
        print(f"✅ Valid pools processed: {processed_pools:,}")
        print(f"❌ Skipped (missing data): {total_pools_from_api - processed_pools:,}")
        print(f"💾 Raw data stored in: raw_defillama_pools")
        print(f"🔄 Pools metadata updated in: pools")
        if total_pools_from_api > 0:
            print(f"📊 Processing success rate: {(processed_pools/total_pools_from_api*100):.1f}%")
        print("="*60)
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching DeFiLlama pools: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
    finally:
        if engine:
            engine.dispose()

if __name__ == "__main__":
    fetch_defillama_pools()