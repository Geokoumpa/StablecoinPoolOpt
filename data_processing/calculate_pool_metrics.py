import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

from datetime import date, datetime, timezone, timedelta
from database.db_utils import get_db_connection
import pandas as pd
from sqlalchemy import text
from psycopg2 import extras # Import extras for Json type

# Removed latest price imports to fix timing issues when running at midnight UTC
# from api_clients.coinmarketcap_client import get_latest_eth_price, get_latest_btc_price
# from api_clients.ethgastracker_client import get_current_average_gas_price as get_latest_gas_price

logger = logging.getLogger(__name__)

def fetch_exogenous_data(connection) -> pd.DataFrame:
    """
    Fetches exogenous variables (e.g., ETH price, BTC price, gas fees) from the database.
    """
    # Fetch ETH open price data
    query_eth = """
    WITH daily_data AS (
        SELECT
            data_timestamp,
            (raw_json_data->'USD'->>'close')::numeric AS close_price
        FROM
            raw_coinmarketcap_ohlcv
        WHERE
            symbol = 'ETH'
        UNION ALL
        SELECT CAST(CURRENT_DATE AS TIMESTAMP), NULL
    )
    SELECT
        data_timestamp AS date,
        LAG(close_price, 1) OVER (ORDER BY data_timestamp) AS eth_open
    FROM
        daily_data;
    """
    df_eth = pd.read_sql(query_eth, connection, parse_dates=['date'], index_col='date')

    # Fetch BTC open price data
    query_btc = """
    WITH daily_data AS (
        SELECT
            data_timestamp,
            (raw_json_data->'USD'->>'close')::numeric AS close_price
        FROM
            raw_coinmarketcap_ohlcv
        WHERE
            symbol = 'BTC'
        UNION ALL
        SELECT CAST(CURRENT_DATE AS TIMESTAMP), NULL
    )
    SELECT
        data_timestamp AS date,
        LAG(close_price, 1) OVER (ORDER BY data_timestamp) AS btc_open
    FROM
        daily_data;
    """
    df_btc = pd.read_sql(query_btc, connection, parse_dates=['date'], index_col='date')

    # Fetch gas fee data
    query_gas = """
    SELECT
        date,
        actual_avg_gas_gwei AS gas_price_gwei
    FROM
        gas_fees_daily;
    """
    df_gas = pd.read_sql(query_gas, connection, parse_dates=['date'], index_col='date')

    # Ensure all indices are datetime and UTC tz-aware before normalizing
    if not isinstance(df_eth.index, pd.DatetimeIndex):
        df_eth.index = pd.to_datetime(df_eth.index)
    df_eth.index = df_eth.index.tz_localize('UTC') if df_eth.index.tz is None else df_eth.index.tz_convert('UTC')
    
    if not isinstance(df_btc.index, pd.DatetimeIndex):
        df_btc.index = pd.to_datetime(df_btc.index)
    df_btc.index = df_btc.index.tz_localize('UTC') if df_btc.index.tz is None else df_btc.index.tz_convert('UTC')
    
    if not isinstance(df_gas.index, pd.DatetimeIndex):
        df_gas.index = pd.to_datetime(df_gas.index)
    df_gas.index = df_gas.index.tz_localize('UTC') if df_gas.index.tz is None else df_gas.index.tz_convert('UTC')
    
    # Normalize all indices to drop time, keep only date
    df_eth.index = df_eth.index.normalize()
    df_btc.index = df_btc.index.normalize()
    df_gas.index = df_gas.index.normalize()
    
    # Merge the dataframes
    df = pd.merge(df_eth, df_btc, left_index=True, right_index=True, how='outer')
    df = pd.merge(df, df_gas, left_index=True, right_index=True, how='outer')
    return df

def bulk_insert_metrics(connection, metrics_to_insert, batch_size=1000):
    """
    Perform bulk insert using psycopg2's execute_values for better performance.
    """
    if not metrics_to_insert:
        return
        
    # Get raw connection for psycopg2 operations
    raw_conn = connection.connection
    cursor = raw_conn.cursor()
    
    try:
        # Prepare the bulk insert query
        upsert_query = """
            INSERT INTO pool_daily_metrics (
                pool_id, date, actual_apy, forecasted_apy, actual_tvl, forecasted_tvl,
                rolling_apy_7d, rolling_apy_30d, apy_delta_today_yesterday,
                stddev_apy_7d, stddev_apy_30d, stddev_apy_7d_delta, stddev_apy_30d_delta,
                eth_open, btc_open, gas_price_gwei
            ) VALUES %s
            ON CONFLICT (pool_id, date) DO UPDATE SET
                actual_apy = EXCLUDED.actual_apy,
                actual_tvl = EXCLUDED.actual_tvl,
                rolling_apy_7d = EXCLUDED.rolling_apy_7d,
                rolling_apy_30d = EXCLUDED.rolling_apy_30d,
                apy_delta_today_yesterday = EXCLUDED.apy_delta_today_yesterday,
                stddev_apy_7d = EXCLUDED.stddev_apy_7d,
                stddev_apy_30d = EXCLUDED.stddev_apy_30d,
                stddev_apy_7d_delta = EXCLUDED.stddev_apy_7d_delta,
                stddev_apy_30d_delta = EXCLUDED.stddev_apy_30d_delta,
                eth_open = EXCLUDED.eth_open,
                btc_open = EXCLUDED.btc_open,
                gas_price_gwei = EXCLUDED.gas_price_gwei;
        """
        
        # Process in batches to avoid memory issues
        total_batches = (len(metrics_to_insert) + batch_size - 1) // batch_size
        
        for i in range(0, len(metrics_to_insert), batch_size):
            batch = metrics_to_insert[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} with {len(batch)} records...")
            
            # Use execute_values for efficient bulk insert
            extras.execute_values(cursor, upsert_query, batch, template=None, page_size=100)
            
            # Commit each batch to avoid large transactions
            raw_conn.commit()
            
        logger.info(f"Successfully bulk inserted/updated {len(metrics_to_insert)} metrics records.")
        
    except Exception as e:
        raw_conn.rollback()
        logger.error(f"Error during bulk insert: {e}")
        raise
    finally:
        cursor.close()

def calculate_pool_metrics():
    """
    Calculates daily metrics for pools including rolling APY, APY deltas, and standard deviations.
    Stores these metrics in the pool_daily_metrics table for current and historical dates.
    Will fill in metrics for all available dates in the raw data.
    """
    logger.info("Starting pool metrics calculation...")
    conn = None
    try:
        conn = get_db_connection()
        with conn.connect() as connection:
            # Fetch historical pool data only for active pools
            from sqlalchemy import text
            result = connection.execute(text("""
                WITH date_range AS (
                    SELECT generate_series(
                        date(MIN(timestamp)),
                        date(MAX(timestamp)),
                        '1 day'::interval
                    )::date AS date
                    FROM raw_defillama_pool_history
                ),
                unique_pools AS (
                    SELECT DISTINCT h.pool_id
                    FROM raw_defillama_pool_history h
                    JOIN pools p ON h.pool_id = p.pool_id
                    WHERE p.is_active = TRUE
                ),
                daily_data AS (
                    SELECT
                        p.pool_id,
                        d.date,
                        h.timestamp,
                        (h.raw_json_data->>'apy')::numeric AS apy,
                        (h.raw_json_data->>'tvlUsd')::numeric AS tvl_usd
                    FROM unique_pools p
                    CROSS JOIN date_range d
                    LEFT JOIN raw_defillama_pool_history h
                        ON p.pool_id = h.pool_id
                        AND date(h.timestamp) = d.date
                )
                SELECT * FROM daily_data
                ORDER BY pool_id, date;
            """))
            raw_data = result.fetchall()
            df_history = pd.DataFrame(raw_data, columns=['pool_id', 'date', 'timestamp', 'apy', 'tvl_usd'])
            # Convert timestamp to datetime if it's not already
            df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
            # Ensure date column is in date format
            df_history['date'] = pd.to_datetime(df_history['date']).dt.date

            if df_history.empty:
                logger.warning("No historical pool data found. Skipping pool metrics calculation.")
                return

            # Fetch exogenous data (ETH price, gas fees)
            exog_data = fetch_exogenous_data(connection)
            
            # Ensure df_history 'date' column is datetime and UTC tz-aware for merging
            df_history['date'] = pd.to_datetime(df_history['date']).dt.tz_localize('UTC')
            
            # Merge exogenous data with pool history data
            df_history = pd.merge(df_history, exog_data, left_on='date', right_index=True, how='left')
            
            # CRITICAL FIX: Fill missing exogenous data for historical dates to ensure forecasting has complete dataset
            # Forward fill missing values to ensure all historical dates have exogenous data
            logger.info(f"Before exogenous data fill - Missing values:\n{df_history[['eth_open', 'btc_open', 'gas_price_gwei']].isnull().sum()}")

            # Sort by date to ensure proper forward/backward filling
            df_history = df_history.sort_values(['pool_id', 'date'])

            # Fill missing exogenous data using forward fill first, then backward fill
            df_history['eth_open'] = df_history.groupby('pool_id')['eth_open'].ffill().bfill()
            df_history['btc_open'] = df_history.groupby('pool_id')['btc_open'].ffill().bfill()
            df_history['gas_price_gwei'] = df_history.groupby('pool_id')['gas_price_gwei'].ffill().bfill()

            logger.info(f"After exogenous data fill - Missing values:\n{df_history[['eth_open', 'btc_open', 'gas_price_gwei']].isnull().sum()}")
            
            # Convert date column back to date format after merge (normalize to remove time and tz info)
            df_history['date'] = df_history['date'].dt.normalize().dt.date
            logger.info(f"Merged df_history with exogenous data. Shape: {df_history.shape}\nTail:\n{df_history.tail()}")

            # TIMING FIX: Removed fetching and backdating "latest" prices when running at midnight UTC
            # This prevents data leakage where current day prices appear on previous day records
            # The historical data from raw_coinmarketcap_ohlcv and gas_fees_daily already contains
            # the closing prices for the previous day, which is the correct data for forecasting
            logger.info("Using only historical data without fetching latest prices to ensure proper timing for midnight UTC execution")

            # Calculate metrics for each pool
            unique_pool_ids = df_history['pool_id'].unique()
            metrics_to_insert = []
            processed_pools_count = 0
            total_pools = len(unique_pool_ids)

            logger.info(f"Calculating metrics for {total_pools} pools...")

            for i, pool_id in enumerate(unique_pool_ids):
                pool_df = df_history[df_history['pool_id'] == pool_id].sort_values(by='date').copy()
                pool_df['apy'] = pd.to_numeric(pool_df['apy'], errors='coerce')
                
                # Don't drop rows with null APY here - we want to calculate metrics for all dates
                # We'll handle null APY values in the rolling calculations
                
                if len(pool_df) < 2:
                    logger.warning(f"Skipping pool {pool_id} due to insufficient historical data (less than 2 data points).")
                    continue

                # Fill missing APY values using forward fill for better rolling calculations
                pool_df['apy'] = pool_df['apy'].ffill()

                # Calculate rolling 7-day APY
                pool_df['rolling_apy_7d'] = pool_df['apy'].rolling(window=7, min_periods=1).mean()
                # Calculate rolling 30-day APY
                pool_df['rolling_apy_30d'] = pool_df['apy'].rolling(window=30, min_periods=1).mean()

                # Calculate 7-day APY standard deviation
                pool_df['stddev_apy_7d'] = pool_df['apy'].rolling(window=7, min_periods=1).std()
                # Calculate 30-day APY standard deviation
                pool_df['stddev_apy_30d'] = pool_df['apy'].rolling(window=30, min_periods=1).std()

                # Calculate Today - Yesterday APY delta
                pool_df['apy_delta_today_yesterday'] = pool_df['apy'].diff(periods=1)

                # Calculate 7-day APY standard deviation delta
                pool_df['stddev_apy_7d_delta'] = pool_df['stddev_apy_7d'].diff(periods=1)

                # Calculate 30-day APY standard deviation delta
                pool_df['stddev_apy_30d_delta'] = pool_df['stddev_apy_30d'].diff(periods=1)

                # Process metrics for each date in the pool's history - ENSURE ALL DATES GET METRICS
                for _, row in pool_df.iterrows():
                    # Convert numpy types to Python native types
                    # CRITICAL: Only skip rows that have completely null data, not just null APY
                    if pd.isnull(row['date']):
                        continue
                        
                    metrics_to_insert.append((
                        pool_id,
                        row['date'],
                        float(row['apy']) if pd.notnull(row['apy']) else None,
                        None,  # forecasted_apy (will be filled in forecasting phase)
                        float(row['tvl_usd']) if pd.notnull(row['tvl_usd']) else None,
                        None,  # forecasted_tvl (will be filled in forecasting phase)
                        float(row['rolling_apy_7d']) if pd.notnull(row['rolling_apy_7d']) else None,
                        float(row['rolling_apy_30d']) if pd.notnull(row['rolling_apy_30d']) else None,
                        float(row['apy_delta_today_yesterday']) if pd.notnull(row['apy_delta_today_yesterday']) else None,
                        float(row['stddev_apy_7d']) if pd.notnull(row['stddev_apy_7d']) else None,
                        float(row['stddev_apy_30d']) if pd.notnull(row['stddev_apy_30d']) else None,
                        float(row['stddev_apy_7d_delta']) if pd.notnull(row['stddev_apy_7d_delta']) else None,
                        float(row['stddev_apy_30d_delta']) if pd.notnull(row['stddev_apy_30d_delta']) else None,
                        float(row['eth_open']) if pd.notnull(row['eth_open']) else None,
                        float(row['btc_open']) if pd.notnull(row['btc_open']) else None,
                        float(row['gas_price_gwei']) if pd.notnull(row['gas_price_gwei']) else None
                ))
                processed_pools_count += 1
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {processed_pools_count}/{total_pools} pools...")

            logger.info(f"Finished calculating metrics for all {total_pools} pools.")

            # Bulk Insert/Update into pool_daily_metrics using optimized bulk insert
            logger.info(f"Bulk inserting/updating {len(metrics_to_insert)} metrics into the database...")
            bulk_insert_metrics(connection, metrics_to_insert, batch_size=1000)
            
            # Now handle historical metrics for filtered pools to ensure 7 months of data
            logger.info("Checking for missing historical metrics for filtered pools...")

            # Fetch filtered pools
            filtered_pools_query = text("""
                SELECT DISTINCT pool_id FROM pool_daily_metrics
                WHERE date = CURRENT_DATE AND is_filtered_out = FALSE;
            """)
            filtered_pool_ids = [row[0] for row in connection.execute(filtered_pools_query).fetchall()]

            logger.info(f"Processing historical metrics for {len(filtered_pool_ids)} filtered pools...")

            historical_metrics_to_insert = []

            for pool_id in filtered_pool_ids:
                # Define 7 months ago
                seven_months_ago = date.today() - timedelta(days=210)

                # Get existing dates for this pool in the last 7 months
                existing_dates_query = text("""
                    SELECT date FROM pool_daily_metrics
                    WHERE pool_id = :pool_id AND date >= :start_date;
                """)
                existing_dates = set(row[0] for row in connection.execute(existing_dates_query, {"pool_id": pool_id, "start_date": seven_months_ago}).fetchall())

                # Generate all dates from 7 months ago to today
                all_dates = set()
                current = seven_months_ago
                today = date.today()
                while current <= today:
                    all_dates.add(current)
                    current += timedelta(days=1)

                missing_dates = all_dates - existing_dates

                if missing_dates:
                    logger.info(f"Pool {pool_id} has {len(missing_dates)} missing dates in last 6 months.")

                    # Fetch all raw data for this pool in the last 7 months
                    all_raw_query = text("""
                        SELECT
                            DATE(h.timestamp) as date,
                            h.timestamp,
                            (h.raw_json_data->>'apy')::numeric AS apy,
                            (h.raw_json_data->>'tvlUsd')::numeric AS tvl_usd
                        FROM raw_defillama_pool_history h
                        WHERE h.pool_id = :pool_id
                          AND DATE(h.timestamp) >= :start_date
                        ORDER BY DATE(h.timestamp), h.timestamp DESC;
                    """)
                    raw_data = connection.execute(all_raw_query, {"pool_id": pool_id, "start_date": seven_months_ago}).fetchall()

                    if raw_data:
                        df_pool = pd.DataFrame(raw_data, columns=['date', 'timestamp', 'apy', 'tvl_usd'])
                        df_pool['timestamp'] = pd.to_datetime(df_pool['timestamp'])
                        df_pool['date'] = pd.to_datetime(df_pool['date']).dt.date

                        # Merge with exogenous data
                        df_pool['date'] = pd.to_datetime(df_pool['date']).dt.tz_localize('UTC')
                        df_pool = pd.merge(df_pool, exog_data, left_on='date', right_index=True, how='left')
                        df_pool = df_pool.sort_values(['date'])

                        # Fill exogenous data
                        df_pool['eth_open'] = df_pool.groupby(level=0)['eth_open'].ffill().bfill()
                        df_pool['btc_open'] = df_pool.groupby(level=0)['btc_open'].ffill().bfill()
                        df_pool['gas_price_gwei'] = df_pool.groupby(level=0)['gas_price_gwei'].ffill().bfill()

                        df_pool['date'] = df_pool['date'].dt.normalize().dt.date
                        df_pool['apy'] = pd.to_numeric(df_pool['apy'], errors='coerce')
                        df_pool['apy'] = df_pool['apy'].ffill()

                        # Calculate metrics
                        df_pool['rolling_apy_7d'] = df_pool['apy'].rolling(window=7, min_periods=1).mean()
                        df_pool['rolling_apy_30d'] = df_pool['apy'].rolling(window=30, min_periods=1).mean()
                        df_pool['stddev_apy_7d'] = df_pool['apy'].rolling(window=7, min_periods=1).std()
                        df_pool['stddev_apy_30d'] = df_pool['apy'].rolling(window=30, min_periods=1).std()
                        df_pool['apy_delta_today_yesterday'] = df_pool['apy'].diff(periods=1)
                        df_pool['stddev_apy_7d_delta'] = df_pool['stddev_apy_7d'].diff(periods=1)
                        df_pool['stddev_apy_30d_delta'] = df_pool['stddev_apy_30d'].diff(periods=1)

                        # Insert metrics for missing dates
                        for _, row in df_pool.iterrows():
                            if row['date'] in missing_dates and pd.notnull(row['date']):
                                historical_metrics_to_insert.append((
                                    pool_id,
                                    row['date'],
                                    float(row['apy']) if pd.notnull(row['apy']) else None,
                                    None,
                                    float(row['tvl_usd']) if pd.notnull(row['tvl_usd']) else None,
                                    None,
                                    float(row['rolling_apy_7d']) if pd.notnull(row['rolling_apy_7d']) else None,
                                    float(row['rolling_apy_30d']) if pd.notnull(row['rolling_apy_30d']) else None,
                                    float(row['apy_delta_today_yesterday']) if pd.notnull(row['apy_delta_today_yesterday']) else None,
                                    float(row['stddev_apy_7d']) if pd.notnull(row['stddev_apy_7d']) else None,
                                    float(row['stddev_apy_30d']) if pd.notnull(row['stddev_apy_30d']) else None,
                                    float(row['stddev_apy_7d_delta']) if pd.notnull(row['stddev_apy_7d_delta']) else None,
                                    float(row['stddev_apy_30d_delta']) if pd.notnull(row['stddev_apy_30d_delta']) else None,
                                    float(row['eth_open']) if pd.notnull(row['eth_open']) else None,
                                    float(row['btc_open']) if pd.notnull(row['btc_open']) else None,
                                    float(row['gas_price_gwei']) if pd.notnull(row['gas_price_gwei']) else None
                                ))

            # Bulk insert historical metrics using optimized bulk insert
            if historical_metrics_to_insert:
                logger.info(f"Bulk inserting {len(historical_metrics_to_insert)} historical metrics for filtered pools...")
                bulk_insert_metrics(connection, historical_metrics_to_insert, batch_size=1000)

            logger.info("Historical metrics calculation for filtered pools completed.")
            logger.info("Pool metrics calculation completed successfully.")
            
            # Print comprehensive summary
            logger.info("\n" + "="*60)
            logger.info("📊 POOL METRICS CALCULATION SUMMARY")
            logger.info("="*60)
            logger.info(f"📥 Total pools processed: {len(unique_pool_ids)}")
            logger.info(f"📈 Metrics calculated per pool:")
            logger.info("   • Rolling APY (7d & 30d)")
            logger.info("   • APY standard deviation (7d & 30d)")
            logger.info("   • APY delta (today vs yesterday)")
            logger.info("   • Standard deviation deltas")
            logger.info(f"💾 Records updated in: pool_daily_metrics")
            logger.info(f"📅 Date: {date.today()}")
            if len(metrics_to_insert) > 0:
                logger.info(f"✅ Successfully processed: {len(metrics_to_insert)} pool metrics")
            logger.info("="*60)

    except Exception as e:
        logger.error(f"Error during pool metrics calculation: {e}")
        logger.error("\n" + "="*60)
        logger.error("❌ POOL METRICS CALCULATION FAILED")
        logger.error("="*60)
        logger.error(f"Error: {str(e)}")
        logger.error("="*60)
    finally:
        if conn:
            conn.dispose()

if __name__ == "__main__":
    calculate_pool_metrics()