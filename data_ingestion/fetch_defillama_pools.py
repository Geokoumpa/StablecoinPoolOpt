import requests
import json
import logging
from datetime import datetime, timezone
from database.db_utils import get_db_connection
from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)

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
                    logger.info(f"Successfully inserted raw data into raw_defillama_pools.")
                else:
                    logger.info(f"Raw data already exists for today in raw_defillama_pools, skipping insertion.")
    except Exception as e:
        logger.error(f"Error inserting data into raw_defillama_pools: {e}")

def detect_and_mark_stale_pools(engine, current_api_pool_ids):
    """
    Detects pools that exist in database but not in current API response and marks them as inactive.
    """
    from sqlalchemy import text
    
    try:
        with engine.connect() as conn:
            with conn.begin():
                # Get all existing pool IDs from database
                existing_pools_query = text("SELECT pool_id FROM pools;")
                result = conn.execute(existing_pools_query)
                existing_pool_ids = {row[0] for row in result.fetchall()}
                
                # Identify pools that are in DB but not in API response
                stale_pool_ids = existing_pool_ids - set(current_api_pool_ids)
                
                if stale_pool_ids:
                    # Mark stale pools as inactive
                    stale_ids_str = "', '".join(stale_pool_ids)
                    update_query = text(f"""
                        UPDATE pools 
                        SET is_active = FALSE 
                        WHERE pool_id IN ('{stale_ids_str}');
                    """)
                    conn.execute(update_query)
                    
                    logger.info(f"🗑️ Marked {len(stale_pool_ids)} pools as inactive (no longer in DeFiLlama API)")
                    for pool_id in stale_pool_ids:
                        logger.info(f"   - Inactive pool: {pool_id}")
                else:
                    logger.info("✅ No stale pools detected - all existing pools are still active in API")
                    
                return len(stale_pool_ids)
    except Exception as e:
        logger.error(f"Error detecting and marking stale pools: {e}")
        return 0

def update_pools_metadata_bulk(conn, pools_data):
    """
    Parses raw pool data and bulk inserts or updates it into the 'pools' master table.
    """
    values_to_insert = []
    current_api_pool_ids = []
    
    for pool_data in pools_data:
        pool_id = pool_data.get('pool')
        chain = pool_data.get('chain')
        project = pool_data.get('project')
        symbol = pool_data.get('symbol')
        tvl_usd = pool_data.get('tvlUsd')
        apy = pool_data.get('apy')
        pool_meta = pool_data.get('poolMeta')

        if not all([pool_id, chain, project, symbol]):
            logger.warning(f"Skipping pool with missing essential data: {pool_data}")
            continue
        
        # Collect current API pool IDs for stale detection
        current_api_pool_ids.append(pool_id)
        
        # Extract underlying tokens from the pool data
        underlying_tokens = pool_data.get('underlyingTokens', [])
        
        values_to_insert.append((
            pool_id, symbol, chain, project, symbol, tvl_usd, apy, underlying_tokens, pool_meta, datetime.now(timezone.utc)
        ))

    if not values_to_insert:
        logger.info("No valid pools to insert after filtering.")
        return current_api_pool_ids

    try:
        with conn.begin() as connection:
            update_query = """
                INSERT INTO pools (
                    pool_id, name, chain, protocol, symbol, tvl, apy, underlying_token_addresses, poolMeta, last_updated
                ) VALUES %s
                ON CONFLICT (pool_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    chain = EXCLUDED.chain,
                    protocol = EXCLUDED.protocol,
                    symbol = EXCLUDED.symbol,
                    tvl = EXCLUDED.tvl,
                    apy = EXCLUDED.apy,
                    underlying_token_addresses = EXCLUDED.underlying_token_addresses,
                    poolMeta = EXCLUDED.poolMeta,
                    last_updated = EXCLUDED.last_updated,
                    is_active = TRUE;
            """
            execute_values(
                connection.connection.cursor(),
                update_query,
                values_to_insert,
                template=None,
                page_size=100
            )
        logger.info(f"Successfully inserted/updated {len(values_to_insert)} pools.")
        return current_api_pool_ids
    except Exception as e:
        logger.error(f"Error during bulk update of pools metadata: {e}")
        return current_api_pool_ids

def fetch_defillama_pools():
    url = "https://yields.llama.fi/pools"
    engine = None
    try:
        engine = get_db_connection()
        if not engine:
            logger.error("Could not establish database connection. Exiting.")
            return

        logger.info("Fetching DeFiLlama pools data...")
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        raw_data = response.json()

        # Store raw JSON data
        insert_raw_defillama_pools(engine, raw_data)

        # Parse and update pools metadata in bulk
        pools_data = raw_data.get('data', [])
        total_pools_from_api = len(pools_data) if pools_data else 0
        
        processed_pools = 0
        current_api_pool_ids = []
        if pools_data:
            # Count valid pools before processing
            valid_pools = [pool for pool in pools_data if all([
                pool.get('pool'), pool.get('chain'), pool.get('project'), pool.get('symbol')
            ])]
            processed_pools = len(valid_pools)
            
            # Update pools metadata and get current API pool IDs
            current_api_pool_ids = update_pools_metadata_bulk(engine, pools_data)

        # Detect and mark stale pools
        stale_count = detect_and_mark_stale_pools(engine, current_api_pool_ids)

        # Print detailed summary
        logger.info("\n" + "="*60)
        logger.info("📥 DEFILLAMA POOLS INGESTION SUMMARY")
        logger.info("="*60)
        logger.info(f"🌐 API endpoint: {url}")
        logger.info(f"📊 Total pools from API: {total_pools_from_api:,}")
        logger.info(f"✅ Valid pools processed: {processed_pools:,}")
        logger.info(f"❌ Skipped (missing data): {total_pools_from_api - processed_pools:,}")
        logger.info(f"🗑️ Stale pools marked inactive: {stale_count:,}")
        logger.info(f"💾 Raw data stored in: raw_defillama_pools")
        logger.info(f"🔄 Pools metadata updated in: pools")
        if total_pools_from_api > 0:
            logger.info(f"📊 Processing success rate: {(processed_pools/total_pools_from_api*100):.1f}%")
        logger.info("="*60)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching DeFiLlama pools: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON response: {e}")
    finally:
        if engine:
            engine.dispose()

if __name__ == "__main__":
    fetch_defillama_pools()