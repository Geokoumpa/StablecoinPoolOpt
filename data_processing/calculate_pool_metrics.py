import logging
import pandas as pd
from datetime import date, timedelta
from database.repositories.pool_metrics_repository import PoolMetricsRepository
from database.repositories.raw_data_repository import RawDataRepository
from database.repositories.gas_fee_repository import GasFeeRepository

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

def fetch_exogenous_data(raw_repo: RawDataRepository, gas_repo: GasFeeRepository) -> pd.DataFrame:
    """
    Fetches exogenous variables (e.g., ETH price, BTC price, gas fees) from the database.
    """
    # Fetch ETH open price data
    eth_rows = raw_repo.get_crypto_open_prices('ETH')
    df_eth = pd.DataFrame(eth_rows, columns=['date', 'eth_open'])
    if not df_eth.empty:
        df_eth['date'] = pd.to_datetime(df_eth['date'])
        df_eth = df_eth.set_index('date')
        df_eth.index = df_eth.index.tz_localize('UTC').normalize() if df_eth.index.tz is None else df_eth.index.tz_convert('UTC').normalize()

    # Fetch BTC open price data
    btc_rows = raw_repo.get_crypto_open_prices('BTC')
    df_btc = pd.DataFrame(btc_rows, columns=['date', 'btc_open'])
    if not df_btc.empty:
        df_btc['date'] = pd.to_datetime(df_btc['date'])
        df_btc = df_btc.set_index('date')
        df_btc.index = df_btc.index.tz_localize('UTC').normalize() if df_btc.index.tz is None else df_btc.index.tz_convert('UTC').normalize()

    # Fetch gas fee data
    gas_data = gas_repo.get_all_daily_data()
    gas_records = [{'date': d.date, 'gas_price_gwei': float(d.actual_avg_gas_gwei) if d.actual_avg_gas_gwei is not None else None} for d in gas_data]
    df_gas = pd.DataFrame(gas_records)
    if not df_gas.empty:
        df_gas['date'] = pd.to_datetime(df_gas['date'])
        df_gas = df_gas.set_index('date')
        df_gas.index = df_gas.index.tz_localize('UTC').normalize() if df_gas.index.tz is None else df_gas.index.tz_convert('UTC').normalize()

    # Merge the dataframes helper
    def merge_df(left, right):
        if left.empty: return right
        if right.empty: return left
        return pd.merge(left, right, left_index=True, right_index=True, how='outer')

    df = df_eth
    df = merge_df(df, df_btc)
    df = merge_df(df, df_gas)
    
    if df.empty:
        return pd.DataFrame(columns=['eth_open', 'btc_open', 'gas_price_gwei']) # basic columns
        
    return df

def calculate_pool_metrics():
    """
    Calculates daily metrics for pools using Repositories.
    """
    logger.info("Starting pool metrics calculation...")
    
    # Initialize repositories
    metrics_repo = PoolMetricsRepository()
    raw_repo = RawDataRepository()
    gas_repo = GasFeeRepository()

    try:
        # Fetch historical pool data only for active pools
        raw_rows = raw_repo.get_raw_history_for_active_pools()
        df_history = pd.DataFrame(raw_rows, columns=['pool_id', 'date', 'timestamp', 'apy', 'tvl_usd'])
        
        # Convert types
        df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
        df_history['date'] = pd.to_datetime(df_history['date']).dt.date

        if df_history.empty:
            logger.warning("No historical pool data found. Skipping pool metrics calculation.")
            return

        # Fetch exogenous data (ETH price, gas fees)
        exog_data = fetch_exogenous_data(raw_repo, gas_repo)
        
        # Ensure df_history 'date' column is datetime and UTC tz-aware for merging
        df_history['date'] = pd.to_datetime(df_history['date']).dt.tz_localize('UTC')
        
        # Merge exogenous data with pool history data
        df_history = pd.merge(df_history, exog_data, left_on='date', right_index=True, how='left')
        
        # Fill missing exogenous data
        logger.info(f"Before exogenous data fill - Missing values:\n{df_history[['eth_open', 'btc_open', 'gas_price_gwei']].isnull().sum()}")

        df_history = df_history.sort_values(['pool_id', 'date'])

        # Groupby fill
        df_history['eth_open'] = df_history.groupby('pool_id')['eth_open'].ffill().bfill()
        df_history['btc_open'] = df_history.groupby('pool_id')['btc_open'].ffill().bfill()
        df_history['gas_price_gwei'] = df_history.groupby('pool_id')['gas_price_gwei'].ffill().bfill()

        logger.info(f"After exogenous data fill - Missing values:\n{df_history[['eth_open', 'btc_open', 'gas_price_gwei']].isnull().sum()}")
        
        # Convert date column back to date format after merge
        df_history['date'] = df_history['date'].dt.normalize().dt.date
        logger.info(f"Merged df_history with exogenous data. Shape: {df_history.shape}")

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
            
            if len(pool_df) < 2:
                logger.warning(f"Skipping pool {pool_id} due to insufficient historical data (less than 2 data points).")
                continue

            # Fill missing APY values for rolling calculations
            pool_df['apy'] = pool_df['apy'].ffill()

            # Calculate rolling metrics
            pool_df['rolling_apy_7d'] = pool_df['apy'].rolling(window=7, min_periods=1).mean()
            pool_df['rolling_apy_30d'] = pool_df['apy'].rolling(window=30, min_periods=1).mean()
            pool_df['stddev_apy_7d'] = pool_df['apy'].rolling(window=7, min_periods=1).std()
            pool_df['stddev_apy_30d'] = pool_df['apy'].rolling(window=30, min_periods=1).std()
            pool_df['apy_delta_today_yesterday'] = pool_df['apy'].diff(periods=1)
            pool_df['stddev_apy_7d_delta'] = pool_df['stddev_apy_7d'].diff(periods=1)
            pool_df['stddev_apy_30d_delta'] = pool_df['stddev_apy_30d'].diff(periods=1)

            for _, row in pool_df.iterrows():
                if pd.isnull(row['date']):
                    continue
                    
                metrics_to_insert.append({
                    'pool_id': pool_id,
                    'date': row['date'],
                    'actual_apy': float(row['apy']) if pd.notnull(row['apy']) else None,
                    'forecasted_apy': None,
                    'actual_tvl': float(row['tvl_usd']) if pd.notnull(row['tvl_usd']) else None,
                    'forecasted_tvl': None,
                    'rolling_apy_7d': float(row['rolling_apy_7d']) if pd.notnull(row['rolling_apy_7d']) else None,
                    'rolling_apy_30d': float(row['rolling_apy_30d']) if pd.notnull(row['rolling_apy_30d']) else None,
                    'apy_delta_today_yesterday': float(row['apy_delta_today_yesterday']) if pd.notnull(row['apy_delta_today_yesterday']) else None,
                    'stddev_apy_7d': float(row['stddev_apy_7d']) if pd.notnull(row['stddev_apy_7d']) else None,
                    'stddev_apy_30d': float(row['stddev_apy_30d']) if pd.notnull(row['stddev_apy_30d']) else None,
                    'stddev_apy_7d_delta': float(row['stddev_apy_7d_delta']) if pd.notnull(row['stddev_apy_7d_delta']) else None,
                    'stddev_apy_30d_delta': float(row['stddev_apy_30d_delta']) if pd.notnull(row['stddev_apy_30d_delta']) else None,
                    'eth_open': float(row['eth_open']) if pd.notnull(row['eth_open']) else None,
                    'btc_open': float(row['btc_open']) if pd.notnull(row['btc_open']) else None,
                    'gas_price_gwei': float(row['gas_price_gwei']) if pd.notnull(row['gas_price_gwei']) else None,
                })
                
            processed_pools_count += 1
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {processed_pools_count}/{total_pools} pools...")

        logger.info(f"Finished calculating metrics for all {total_pools} pools.")

        # Bulk Insert/Update
        logger.info(f"Bulk inserting/updating {len(metrics_to_insert)} metrics into the database...")
        metrics_repo.bulk_upsert_calculated_metrics(metrics_to_insert) # Call new method
        
        # Now handle historical metrics for filtered pools
        logger.info("Checking for missing historical metrics for filtered pools...")

        filtered_pool_ids = metrics_repo.get_filtered_pool_ids_for_date(date.today(), is_filtered_out=False)
        
        historical_metrics_to_insert = []

        for pool_id in filtered_pool_ids:
            seven_months_ago = date.today() - timedelta(days=210)
            existing_dates = set(metrics_repo.get_existing_dates_for_pool(pool_id, seven_months_ago))

            all_dates = set()
            current = seven_months_ago
            today = date.today()
            while current <= today:
                all_dates.add(current)
                current += timedelta(days=1)

            missing_dates = all_dates - existing_dates

            if missing_dates:
                logger.info(f"Pool {pool_id} has {len(missing_dates)} missing dates in last 6 months.")
                
                raw_pool_rows = raw_repo.get_raw_history_for_pool(pool_id, seven_months_ago)
                if not raw_pool_rows: continue
                
                df_pool = pd.DataFrame(raw_pool_rows, columns=['date', 'timestamp', 'apy', 'tvl_usd'])
                df_pool['timestamp'] = pd.to_datetime(df_pool['timestamp'])
                df_pool['date'] = pd.to_datetime(df_pool['date']).dt.date

                # Merge exog
                df_pool['date'] = pd.to_datetime(df_pool['date']).dt.tz_localize('UTC')
                df_pool = pd.merge(df_pool, exog_data, left_on='date', right_index=True, how='left')
                df_pool = df_pool.sort_values(['date'])

                # Fill exog
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

                for _, row in df_pool.iterrows():
                    if row['date'] in missing_dates and pd.notnull(row['date']):
                        historical_metrics_to_insert.append({
                            'pool_id': pool_id,
                            'date': row['date'],
                            'actual_apy': float(row['apy']) if pd.notnull(row['apy']) else None,
                            'actual_tvl': float(row['tvl_usd']) if pd.notnull(row['tvl_usd']) else None,
                            'rolling_apy_7d': float(row['rolling_apy_7d']) if pd.notnull(row['rolling_apy_7d']) else None,
                            'rolling_apy_30d': float(row['rolling_apy_30d']) if pd.notnull(row['rolling_apy_30d']) else None,
                            'apy_delta_today_yesterday': float(row['apy_delta_today_yesterday']) if pd.notnull(row['apy_delta_today_yesterday']) else None,
                            'stddev_apy_7d': float(row['stddev_apy_7d']) if pd.notnull(row['stddev_apy_7d']) else None,
                            'stddev_apy_30d': float(row['stddev_apy_30d']) if pd.notnull(row['stddev_apy_30d']) else None,
                            'stddev_apy_7d_delta': float(row['stddev_apy_7d_delta']) if pd.notnull(row['stddev_apy_7d_delta']) else None,
                            'stddev_apy_30d_delta': float(row['stddev_apy_30d_delta']) if pd.notnull(row['stddev_apy_30d_delta']) else None,
                            'eth_open': float(row['eth_open']) if pd.notnull(row['eth_open']) else None,
                            'btc_open': float(row['btc_open']) if pd.notnull(row['btc_open']) else None,
                            'gas_price_gwei': float(row['gas_price_gwei']) if pd.notnull(row['gas_price_gwei']) else None,
                        })

        if historical_metrics_to_insert:
            logger.info(f"Bulk inserting {len(historical_metrics_to_insert)} historical metrics for filtered pools...")
            metrics_repo.bulk_upsert_calculated_metrics(historical_metrics_to_insert)

        logger.info("Pool metrics calculation completed successfully.")

    except Exception as e:
        logger.error(f"Error during pool metrics calculation: {e}")
        raise

if __name__ == "__main__":
    calculate_pool_metrics()
