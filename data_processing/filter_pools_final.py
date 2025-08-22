import logging
from database.db_utils import get_db_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def filter_pools_final():
    """
    Final filtering phase: Apply icebox token filtering to pre-filtered pools.
    Only processes pools that passed pre-filtering and adds icebox exclusions.
    """
    logging.info("Starting final pool filtering (icebox)...")
    engine = None
    conn = None
    trans = None
    try:
        engine = get_db_connection()
        conn = engine.raw_connection()
        cur = conn.cursor()

        # Get blacklisted tokens from the blacklisted_tokens table
        cur.execute("SELECT token_symbol FROM blacklisted_tokens;")
        blacklisted_tokens = {row[0] for row in cur.fetchall()}

        # Get icebox tokens from the icebox_tokens table (only active ones)
        cur.execute("SELECT token_symbol FROM icebox_tokens WHERE removed_timestamp IS NULL;")
        icebox_tokens = {row[0] for row in cur.fetchall()}

        print("=== Final Pool Filtering Criteria ===")
        print(f"Blacklisted Tokens: {blacklisted_tokens}")
        print(f"Icebox Tokens: {icebox_tokens}")
        print("=====================================")

        # Fetch pools that passed pre-filtering (not filtered out)
        cur.execute("""
            SELECT pdm.pool_id, p.symbol, pdm.filter_reason
            FROM pool_daily_metrics pdm
            JOIN pools p ON pdm.pool_id = p.pool_id
            WHERE pdm.date = CURRENT_DATE AND pdm.is_filtered_out = FALSE;
        """)
        pre_filtered_pools = cur.fetchall()

        logging.info(f"Processing {len(pre_filtered_pools)} pre-filtered pools for final filtering...")

        for pool_id, symbol, existing_reason in pre_filtered_pools:
            additional_reasons = []
            should_filter_out = False

            # Check for blacklisted tokens
            if symbol:
                blacklisted_found = [token for token in blacklisted_tokens if token in symbol]
                if blacklisted_found:
                    additional_reasons.append(f"Pool contains blacklisted token(s): {', '.join(blacklisted_found)}.")
                    should_filter_out = True

                # Check for icebox tokens
                icebox_found = [token for token in icebox_tokens if token in symbol]
                if icebox_found:
                    additional_reasons.append(f"Pool contains icebox token(s): {', '.join(icebox_found)}.")
                    should_filter_out = True

            if should_filter_out:
                # Combine existing reasons with new ones
                combined_reasons = []
                if existing_reason:
                    combined_reasons.append(existing_reason)
                combined_reasons.extend(additional_reasons)

                # Update the pool to be filtered out
                cur.execute("""
                    UPDATE pool_daily_metrics
                    SET is_filtered_out = TRUE, filter_reason = %s
                    WHERE pool_id = %s AND date = CURRENT_DATE;
                """, ('; '.join(combined_reasons), pool_id))

                logging.info(f"Pool {pool_id} finally filtered out. Additional reasons: {'; '.join(additional_reasons)}.")

        conn.commit()
        
        # Log final statistics
        cur.execute("""
            SELECT COUNT(*) FROM pool_daily_metrics 
            WHERE date = CURRENT_DATE AND is_filtered_out = FALSE;
        """)
        final_count = cur.fetchone()[0]
        
        cur.execute("""
            SELECT COUNT(*) FROM pool_daily_metrics 
            WHERE date = CURRENT_DATE AND is_filtered_out = TRUE;
        """)
        filtered_count = cur.fetchone()[0]

        logging.info(f"Final filtering completed successfully.")
        
        # Print detailed summary
        print("\n" + "="*60)
        print("🏁 FINAL POOL FILTERING SUMMARY")
        print("="*60)
        print(f"📥 Input pools (pre-filtered): {len(pre_filtered_pools)}")
        print(f"🔍 Blacklisted tokens checked: {len(blacklisted_tokens)}")
        print(f"📦 Icebox tokens checked: {len(icebox_tokens)}")
        print(f"✅ Final approved pools: {final_count}")
        print(f"❌ Total filtered out pools: {filtered_count}")
        if len(pre_filtered_pools) > 0:
            print(f"📊 Final approval rate: {(final_count/len(pre_filtered_pools)*100):.1f}%")
        print("="*60)

    except Exception as e:
        logging.error(f"Error during final pool filtering: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()
        if engine:
            engine.dispose()

if __name__ == "__main__":
    filter_pools_final()