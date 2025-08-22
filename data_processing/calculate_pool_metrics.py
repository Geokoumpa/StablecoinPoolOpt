import logging
from datetime import date, datetime, timezone
from database.db_utils import get_db_connection
import pandas as pd
from sqlalchemy import text
from psycopg2 import extras # Import extras for Json type

from api_clients.coinmarketcap_client import get_latest_eth_price, get_latest_btc_price
from api_clients.ethgastracker_client import get_current_average_gas_price as get_latest_gas_price

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_exogenous_data(connection) -> pd.DataFrame:
    """
    Fetches exogenous variables (e.g., ETH price, BTC price, gas fees) from the database.
    """
    # Fetch ETH price data
    query_eth = """
    SELECT
        data_timestamp AS date,
        (raw_json_data->>'close')::float AS eth_price_usd
    FROM
        raw_coinmarketcap_ohlcv
    WHERE
        symbol = 'ETH';
    """
    df_eth = pd.read_sql(query_eth, connection, parse_dates=['date'], index_col='date')

    # Fetch BTC price data
    query_btc = """
    SELECT
        data_timestamp AS date,
        (raw_json_data->>'close')::float AS btc_price_usd
    FROM
        raw_coinmarketcap_ohlcv
    WHERE
        symbol = 'BTC';
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

def calculate_pool_metrics():
    """
    Calculates daily metrics for pools including rolling APY, APY deltas, and standard deviations.
    Stores these metrics in the pool_daily_metrics table for current and historical dates.
    Will fill in metrics for all available dates in the raw data.
    """
    logging.info("Starting pool metrics calculation...")
    conn = None
    try:
        conn = get_db_connection()
        with conn.connect() as connection:
            # Fetch historical pool data
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
                    SELECT DISTINCT pool_id
                    FROM raw_defillama_pool_history
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
                logging.warning("No historical pool data found. Skipping pool metrics calculation.")
                return

            # Fetch exogenous data (ETH price, gas fees)
            exog_data = fetch_exogenous_data(connection)
            
            # Ensure df_history 'date' column is datetime and UTC tz-aware for merging
            df_history['date'] = pd.to_datetime(df_history['date']).dt.tz_localize('UTC')
            
            # Merge exogenous data with pool history data
            df_history = pd.merge(df_history, exog_data, left_on='date', right_index=True, how='left')
            
            # Convert date column back to date format after merge (normalize to remove time and tz info)
            df_history['date'] = df_history['date'].dt.normalize().dt.date
            logging.info(f"Merged df_history with exogenous data. Shape: {df_history.shape}\nTail:\n{df_history.tail()}")

            # Fetch latest ETH price, BTC price, and Gas price
            latest_eth_price = get_latest_eth_price()
            latest_btc_price = get_latest_btc_price()
            latest_gas_price = get_latest_gas_price()

            # Get the latest date in df_history
            latest_date_in_history = df_history['date'].max()

            # Update df_history with latest ETH price for the latest date
            if latest_eth_price is not None:
                df_history.loc[df_history['date'] == latest_date_in_history, 'eth_price_usd'] = latest_eth_price
                logging.info(f"Updated ETH price for {latest_date_in_history} with latest: {latest_eth_price}")
            else:
                logging.warning("Could not fetch latest ETH price. 'eth_price_usd' for latest date might be NaN.")

            # Update df_history with latest BTC price for the latest date
            if latest_btc_price is not None:
                df_history.loc[df_history['date'] == latest_date_in_history, 'btc_price_usd'] = latest_btc_price
                logging.info(f"Updated BTC price for {latest_date_in_history} with latest: {latest_btc_price}")
            else:
                logging.warning("Could not fetch latest BTC price. 'btc_price_usd' for latest date might be NaN.")

            # Update df_history with latest Gas price for the latest date
            if latest_gas_price is not None:
                df_history.loc[df_history['date'] == latest_date_in_history, 'gas_price_gwei'] = latest_gas_price
                logging.info(f"Updated Gas price for {latest_date_in_history} with latest: {latest_gas_price}")
            else:
                logging.warning("Could not fetch latest Gas price. 'gas_price_gwei' for latest date might be NaN.")

            # Calculate metrics for each pool
            unique_pool_ids = df_history['pool_id'].unique()
            metrics_to_insert = []
            processed_pools_count = 0
            total_pools = len(unique_pool_ids)

            logging.info(f"Calculating metrics for {total_pools} pools...")

            for i, pool_id in enumerate(unique_pool_ids):
                pool_df = df_history[df_history['pool_id'] == pool_id].sort_values(by='timestamp').copy()
                pool_df['apy'] = pd.to_numeric(pool_df['apy'], errors='coerce')
                pool_df.dropna(subset=['apy'], inplace=True)

                if len(pool_df) < 2:
                    logging.warning(f"Skipping pool {pool_id} due to insufficient historical data (less than 2 data points).")
                    continue

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

                # Process metrics for each date in the pool's history
                for _, row in pool_df.iterrows():
                    # Convert numpy types to Python native types
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
                        float(row['eth_price_usd']) if pd.notnull(row['eth_price_usd']) else None,
                        float(row['btc_price_usd']) if pd.notnull(row['btc_price_usd']) else None,
                        float(row['gas_price_gwei']) if pd.notnull(row['gas_price_gwei']) else None
                ))
                processed_pools_count += 1
                if (i + 1) % 10 == 0:
                    logging.info(f"Processed {processed_pools_count}/{total_pools} pools...")

            logging.info(f"Finished calculating metrics for all {total_pools} pools.")

             # Insert/Update into pool_daily_metrics
            logging.info(f"Inserting/updating {len(metrics_to_insert)} metrics into the database...")
            for metric in metrics_to_insert:
                connection.execute(
                    text("""
                        INSERT INTO pool_daily_metrics (
                            pool_id, date, actual_apy, forecasted_apy, actual_tvl, forecasted_tvl,
                            rolling_apy_7d, rolling_apy_30d, apy_delta_today_yesterday,
                            stddev_apy_7d, stddev_apy_30d, stddev_apy_7d_delta, stddev_apy_30d_delta,
                            eth_price_usd, btc_price_usd, gas_price_gwei
                        ) VALUES (:pool_id, :date, :actual_apy, :forecasted_apy, :actual_tvl, :forecasted_tvl,
                                 :rolling_apy_7d, :rolling_apy_30d, :apy_delta_today_yesterday,
                                 :stddev_apy_7d, :stddev_apy_30d, :stddev_apy_7d_delta, :stddev_apy_30d_delta,
                                 :eth_price_usd, :btc_price_usd, :gas_price_gwei)
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
                            eth_price_usd = EXCLUDED.eth_price_usd,
                            btc_price_usd = EXCLUDED.btc_price_usd,
                            gas_price_gwei = EXCLUDED.gas_price_gwei;
                    """),
                    {
                        "pool_id": metric[0],
                        "date": metric[1],
                        "actual_apy": metric[2],
                        "forecasted_apy": metric[3],
                        "actual_tvl": metric[4],
                        "forecasted_tvl": metric[5],
                        "rolling_apy_7d": metric[6],
                        "rolling_apy_30d": metric[7],
                        "apy_delta_today_yesterday": metric[8],
                        "stddev_apy_7d": metric[9],
                        "stddev_apy_30d": metric[10],
                        "stddev_apy_7d_delta": metric[11],
                        "stddev_apy_30d_delta": metric[12],
                        "eth_price_usd": metric[13],
                        "btc_price_usd": metric[14],
                        "gas_price_gwei": metric[15]
                    }
                )

            connection.commit()
            logging.info("Pool metrics calculation completed successfully.")
            
            # Print comprehensive summary
            print("\n" + "="*60)
            print("📊 POOL METRICS CALCULATION SUMMARY")
            print("="*60)
            print(f"📥 Total pools processed: {len(unique_pool_ids)}")
            print(f"📈 Metrics calculated per pool:")
            print("   • Rolling APY (7d & 30d)")
            print("   • APY standard deviation (7d & 30d)")
            print("   • APY delta (today vs yesterday)")
            print("   • Standard deviation deltas")
            print(f"💾 Records updated in: pool_daily_metrics")
            print(f"📅 Date: {date.today()}")
            if len(metrics_to_insert) > 0:
                print(f"✅ Successfully processed: {len(metrics_to_insert)} pool metrics")
            print("="*60)

    except Exception as e:
        logging.error(f"Error during pool metrics calculation: {e}")
        print("\n" + "="*60)
        print("❌ POOL METRICS CALCULATION FAILED")
        print("="*60)
        print(f"Error: {str(e)}")
        print("="*60)
    finally:
        if conn:
            conn.dispose()

if __name__ == "__main__":
    calculate_pool_metrics()