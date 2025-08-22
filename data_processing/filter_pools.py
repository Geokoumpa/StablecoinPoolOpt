import logging
from datetime import date
import json
from database.db_utils import get_db_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def filter_pools():
    """
    Applies various filtering criteria to pools and marks filtered-out pools in pool_daily_metrics.
    """
    logging.info("Starting pool filtering...")
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Fetch latest allocation parameters for filtering thresholds and snapshots
        cur.execute("""
            SELECT
                token_marketcap_limit,
                pool_tvl_limit,
                pool_apy_limit,
                pool_pair_tvl_ratio_min,
                pool_pair_tvl_ratio_max,
                approved_protocols_snapshot,
                approved_tokens_snapshot,
                blacklisted_tokens_snapshot,
                icebox_tokens_snapshot
            FROM allocation_parameters
            ORDER BY timestamp DESC LIMIT 1;
        """)
        filter_params = cur.fetchone()

        if not filter_params:
            logging.warning("No allocation parameters found for filtering. Skipping pool filtering.")
            return

        (token_marketcap_limit, pool_tvl_limit, pool_apy_limit,
         pool_pair_tvl_ratio_min, pool_pair_tvl_ratio_max,
         approved_protocols_snapshot, approved_tokens_snapshot,
         blacklisted_tokens_snapshot, icebox_tokens_snapshot) = filter_params

        approved_protocols = {p['protocol_name'] for p in approved_protocols_snapshot} if approved_protocols_snapshot else set()
        approved_tokens = {t['token_symbol'] for t in approved_tokens_snapshot} if approved_tokens_snapshot else set()
        blacklisted_tokens = {t['token_symbol'] for t in blacklisted_tokens_snapshot} if blacklisted_tokens_snapshot else set()
        icebox_tokens = {t['token_symbol'] for t in icebox_tokens_snapshot if t.get('removed_timestamp') is None} if icebox_tokens_snapshot else set()

        print("=== Pool Filtering Criteria and Thresholds ===")
        print(f"Token Marketcap Limit: {token_marketcap_limit}")
        print(f"Pool TVL Limit: {pool_tvl_limit}")
        print(f"Pool APY Limit: {pool_apy_limit}")
        print(f"Pool Pair TVL Ratio Min: {pool_pair_tvl_ratio_min}")
        print(f"Pool Pair TVL Ratio Max: {pool_pair_tvl_ratio_max}")
        print(f"Approved Protocols: {approved_protocols}")
        print(f"Approved Tokens: {approved_tokens}")
        print(f"Blacklisted Tokens: {blacklisted_tokens}")
        print(f"Icebox Tokens: {icebox_tokens}")
        print("=============================================")

        # Fetch pool data using basic pool information (since historical metrics may not be available yet)
        cur.execute("""
            SELECT
                p.pool_id,
                p.protocol,
                p.symbol,
                p.tvl,
                p.apy,
                p.name
            FROM pools p;
        """)
        pools_data = cur.fetchall()

        for pool_id, protocol, symbol, tvl, apy, name in pools_data:
            filter_reason = []
            is_filtered_out = False

            # Exclude pools not belonging to approved protocols
            if protocol not in approved_protocols:
                filter_reason.append(f"Protocol '{protocol}' not in approved protocols.")
                is_filtered_out = True

            # Exclude pools containing blacklisted tokens
            # Check if any blacklisted token appears as a substring in the pool's symbol
            blacklisted_found = [token for token in blacklisted_tokens if token in symbol]
            if blacklisted_found:
                filter_reason.append(f"Pool contains blacklisted token(s): {', '.join(blacklisted_found)}.")
                is_filtered_out = True

            # Exclude pools containing any Icebox tokens
            # Check if any icebox token appears as a substring in the pool's symbol
            icebox_found = [token for token in icebox_tokens if token in symbol]
            if icebox_found:
                filter_reason.append(f"Pool contains Icebox token(s): {', '.join(icebox_found)}.")
                is_filtered_out = True

            # Exclude pools not containing approved tokens
            # Check if at least one approved token appears as a substring in the pool's symbol
            approved_found = [token for token in approved_tokens if token in symbol]
            if not approved_found:
                filter_reason.append(f"Pool does not contain any approved tokens.")
                is_filtered_out = True

            # Apply additional trade logic filters
            # Exclude tokens with MarketCap < token_marketcap_limit (requires fetching market cap)
            # This data is not directly in 'pools' or 'pool_daily_metrics'.
            # Assuming we'd fetch this from raw_coinmarketcap_ohlcv or a processed token_metrics table.
            # For now, skipping this check as data is not readily available in current tables.
            # if token_marketcap < token_marketcap_limit:
            #     filter_reason.append(f"Token market cap below limit ({token_marketcap_limit}).")
            #     is_filtered_out = True

            # Exclude pools with TVL < pool_tvl_limit (using basic pool data)
            if tvl is not None and tvl < pool_tvl_limit:
                filter_reason.append(f"Pool TVL ({tvl}) below limit ({pool_tvl_limit}).")
                is_filtered_out = True

            # Exclude pools with APY < pool_apy_limit (using basic pool data)
            if apy is not None and apy < pool_apy_limit:
                filter_reason.append(f"Pool APY ({apy}) below limit ({pool_apy_limit}).")
                is_filtered_out = True

            # Exclude pools where side of pair is not within pool_pair_tvl_ratio_min - pool_pair_tvl_ratio_max of other side's TVL.
            # This requires parsing pool 'name' or 'symbol' to identify pairs and their individual TVLs.
            # This is complex and depends on how pool names/symbols are structured.
            # Skipping for now due to complexity and lack of direct data in current schema.
            # if not (pool_pair_tvl_ratio_min <= pair_tvl_ratio <= pool_pair_tvl_ratio_max):
            #     filter_reason.append(f"Pool pair TVL ratio outside limits ({pool_pair_tvl_ratio_min}-{pool_pair_tvl_ratio_max}).")
            #     is_filtered_out = True

            if is_filtered_out:
                # Check if record exists, then insert or update
                cur.execute("""
                    SELECT id FROM pool_daily_metrics
                    WHERE pool_id = %s AND date = CURRENT_DATE;
                """, (pool_id,))
                existing_record = cur.fetchone()
                
                if existing_record:
                    cur.execute("""
                        UPDATE pool_daily_metrics
                        SET is_filtered_out = TRUE, filter_reason = %s
                        WHERE pool_id = %s AND date = CURRENT_DATE;
                    """, ("; ".join(filter_reason), pool_id))
                else:
                    cur.execute("""
                        INSERT INTO pool_daily_metrics (pool_id, date, is_filtered_out, filter_reason)
                        VALUES (%s, CURRENT_DATE, TRUE, %s);
                    """, (pool_id, "; ".join(filter_reason)))
                logging.info(f"Pool {pool_id} filtered out. Reasons: {'; '.join(filter_reason)}")
            else:
                # Check if record exists, then insert or update
                cur.execute("""
                    SELECT id FROM pool_daily_metrics
                    WHERE pool_id = %s AND date = CURRENT_DATE;
                """, (pool_id,))
                existing_record = cur.fetchone()
                
                if existing_record:
                    cur.execute("""
                        UPDATE pool_daily_metrics
                        SET is_filtered_out = FALSE, filter_reason = NULL
                        WHERE pool_id = %s AND date = CURRENT_DATE;
                    """, (pool_id,))
                else:
                    cur.execute("""
                        INSERT INTO pool_daily_metrics (pool_id, date, is_filtered_out, filter_reason)
                        VALUES (%s, CURRENT_DATE, FALSE, NULL);
                    """, (pool_id,))
                logging.info(f"Pool {pool_id} passed all filters.")

        conn.commit()
        logging.info("Pool filtering completed successfully.")

    except Exception as e:
        logging.error(f"Error during pool filtering: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            cur.close()
            conn.close()

if __name__ == "__main__":
    filter_pools()