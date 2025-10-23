import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

from database.db_utils import get_db_connection

logger = logging.getLogger(__name__)

def filter_pools_final():
    """
    Final filtering phase: Apply TVL, APY, and icebox token filtering to pre-filtered pools.
    Only processes pools that passed pre-filtering and adds final exclusions.
    """
    logger.info("Starting final pool filtering (icebox)...")
    engine = None
    conn = None
    trans = None
    try:
        engine = get_db_connection()
        conn = engine.raw_connection()
        cur = conn.cursor()

        # Get icebox tokens from the icebox_tokens table (only active ones)
        cur.execute("SELECT token_symbol FROM icebox_tokens WHERE removed_timestamp IS NULL;")
        icebox_tokens = {row[0] for row in cur.fetchall()}

        # Get allocation parameters for TVL and APY limits
        cur.execute("""
            SELECT pool_tvl_limit, pool_apy_limit
            FROM allocation_parameters
            ORDER BY timestamp DESC LIMIT 1;
        """)
        filter_params = cur.fetchone()
        if filter_params:
            pool_tvl_limit, pool_apy_limit = filter_params
        else:
            pool_tvl_limit, pool_apy_limit = None, None
            logger.warning("No allocation parameters found for TVL/APY filtering.")

        logger.info("=== Final Pool Filtering Criteria ===")
        logger.info(f"Icebox Tokens: {icebox_tokens}")
        logger.info(f"Pool TVL Limit: {pool_tvl_limit}")
        logger.info(f"Pool APY Limit: {pool_apy_limit}")
        logger.info("Using forecasted values when available, falling back to basic pool data")
        logger.info("=====================================")

        # Fetch pools that passed pre-filtering (not filtered out)
        # Include both forecasted and actual pool data as fallback
        cur.execute("""
            SELECT pdm.pool_id, p.symbol, pdm.forecasted_tvl, pdm.forecasted_apy, 
                   p.tvl as basic_tvl, p.apy as basic_apy, pdm.filter_reason
            FROM pool_daily_metrics pdm
            JOIN pools p ON pdm.pool_id = p.pool_id
            WHERE pdm.date = CURRENT_DATE AND pdm.is_filtered_out = FALSE;
        """)
        pre_filtered_pools = cur.fetchall()

        logger.info(f"Processing {len(pre_filtered_pools)} pre-filtered pools for final filtering...")

        for pool_id, symbol, forecasted_tvl, forecasted_apy, basic_tvl, basic_apy, existing_reason in pre_filtered_pools:
            additional_reasons = []
            should_filter_out = False

            # Check for icebox tokens
            if symbol:
                icebox_found = [token for token in icebox_tokens if token in symbol]
                if icebox_found:
                    additional_reasons.append(f"Pool contains icebox token(s): {', '.join(icebox_found)}.")
                    should_filter_out = True

            # Check TVL limit - use forecasted if available, otherwise fall back to basic TVL
            tvl_to_check = forecasted_tvl if forecasted_tvl is not None else basic_tvl
            if tvl_to_check is not None and pool_tvl_limit is not None and tvl_to_check < pool_tvl_limit:
                data_source = "forecasted" if forecasted_tvl is not None else "basic"
                additional_reasons.append(f"Pool {data_source} TVL ({tvl_to_check:.2f}) below limit ({pool_tvl_limit}).")
                should_filter_out = True

            # Check APY limit - use forecasted if available, otherwise fall back to basic APY
            # Note: forecasted_apy and basic_apy are stored as percentages (0.0633 = 0.0633%)
            # pool_apy_limit is stored as decimal (0.0600 = 6% = 6.0%)
            apy_to_check = forecasted_apy if forecasted_apy is not None else basic_apy
            # Convert pool_apy_limit from decimal to percentage for comparison
            if pool_apy_limit is not None:
                apy_limit_percentage = pool_apy_limit * 100  # Convert 0.0600 to 6.0
                if apy_to_check is not None and apy_to_check < apy_limit_percentage:
                    data_source = "forecasted" if forecasted_apy is not None else "basic"
                    additional_reasons.append(f"Pool {data_source} APY ({apy_to_check:.4f}%) below limit ({apy_limit_percentage:.2f}%).")
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

                logger.info(f"Pool {pool_id} finally filtered out. Additional reasons: {'; '.join(additional_reasons)}.")

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

        logger.info(f"Final filtering completed successfully.")
        
        # Print detailed summary
        logger.info("\n" + "="*60)
        logger.info("🏁 FINAL POOL FILTERING SUMMARY")
        logger.info("="*60)
        logger.info(f"📥 Input pools (pre-filtered): {len(pre_filtered_pools)}")
        logger.info(f"📦 Icebox tokens checked: {len(icebox_tokens)}")
        logger.info(f"💰 TVL limit: ${pool_tvl_limit:,.0f}" if pool_tvl_limit else "💰 TVL limit: N/A")
        logger.info(f"📈 APY limit: {pool_apy_limit:.2%}" if pool_apy_limit else "📈 APY limit: N/A")
        logger.info(f"✅ Final approved pools: {final_count}")
        logger.info(f"❌ Total filtered out pools: {filtered_count}")
        if len(pre_filtered_pools) > 0:
            logger.info(f"📊 Final approval rate: {(final_count/len(pre_filtered_pools)*100):.1f}%")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Error during final pool filtering: {e}")
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