import logging
from database.db_utils import get_db_connection
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def filter_pools_pre():
    """
    Pre-filtering phase: Filter pools based on APY, TVL, approved protocols, and approved tokens only.
    Does NOT filter based on blacklisted tokens or icebox tokens (those come from icebox analysis).
    """
    logging.info("Starting pre-pool filtering...")
    engine = get_db_connection()
    if not engine:
        logging.error("Could not establish database connection. Exiting.")
        return

    with engine.connect() as conn:
        trans = conn.begin()
        try:
            # Fetch basic allocation parameters
            result = conn.execute(text("""
                SELECT
                    token_marketcap_limit,
                    pool_tvl_limit,
                    pool_apy_limit,
                    pool_pair_tvl_ratio_min,
                    pool_pair_tvl_ratio_max
                FROM allocation_parameters
                ORDER BY timestamp DESC LIMIT 1;
            """))
            filter_params = result.fetchone()

            if not filter_params:
                logging.warning("No allocation parameters found for filtering. Skipping pre-pool filtering.")
                trans.rollback()
                return

            (token_marketcap_limit, pool_tvl_limit, pool_apy_limit,
             pool_pair_tvl_ratio_min, pool_pair_tvl_ratio_max) = filter_params

            # Get approved protocols
            result = conn.execute(text("SELECT protocol_name FROM approved_protocols;"))
            approved_protocols = {row[0] for row in result.fetchall()}

            # Get approved tokens
            result = conn.execute(text("SELECT token_symbol FROM approved_tokens;"))
            approved_tokens = {row[0] for row in result.fetchall()}

            print("=== Pre-Pool Filtering Criteria and Thresholds ===")
            print(f"Token Marketcap Limit: {token_marketcap_limit}")
            print(f"Pool TVL Limit: {pool_tvl_limit}")
            print(f"Pool APY Limit: {pool_apy_limit}")
            print(f"Pool Pair TVL Ratio Min: {pool_pair_tvl_ratio_min}")
            print(f"Pool Pair TVL Ratio Max: {pool_pair_tvl_ratio_max}")
            print(f"Approved Protocols: {approved_protocols}")
            print(f"Approved Tokens: {approved_tokens}")
            print("================================================")

            # Fetch all pools for pre-filtering
            result = conn.execute(text("""
                SELECT pool_id, protocol, symbol, tvl, apy, name
                FROM pools p;
            """))
            pools_data = result.fetchall()

            for pool_id, protocol, symbol, tvl, apy, name in pools_data:
                filter_reason = []
                is_filtered_out = False

                # Exclude pools not belonging to approved protocols
                if protocol not in approved_protocols:
                    filter_reason.append(f"Protocol '{protocol}' not in approved protocols.")
                    is_filtered_out = True

                # Exclude pools not containing approved tokens
                approved_found = any(token in symbol for token in approved_tokens) if symbol else False
                if not approved_found:
                    filter_reason.append(f"Pool does not contain any approved tokens.")
                    is_filtered_out = True

                # Exclude pools below TVL limit
                if tvl is not None and tvl < pool_tvl_limit:
                    filter_reason.append(f"Pool TVL ({tvl:.2f}) below limit ({pool_tvl_limit}).")
                    is_filtered_out = True

                # Exclude pools below APY limit
                if apy is not None and apy < pool_apy_limit:
                    filter_reason.append(f"Pool APY ({apy:.4f}) below limit ({pool_apy_limit:.4f}).")
                    is_filtered_out = True

                # Check if pool already exists in pool_daily_metrics for today
                result = conn.execute(
                    text("""
                        SELECT id FROM pool_daily_metrics
                        WHERE pool_id = :pool_id AND date = CURRENT_DATE;
                    """),
                    {"pool_id": pool_id}
                )
                existing_entry = result.fetchone()

                if existing_entry:
                    conn.execute(
                        text("""
                            UPDATE pool_daily_metrics
                            SET is_filtered_out = :is_filtered_out, filter_reason = :filter_reason
                            WHERE pool_id = :pool_id AND date = CURRENT_DATE;
                        """),
                        {
                            "is_filtered_out": is_filtered_out,
                            "filter_reason": '; '.join(filter_reason) if filter_reason else None,
                            "pool_id": pool_id
                        }
                    )
                else:
                    conn.execute(
                        text("""
                            INSERT INTO pool_daily_metrics (
                                pool_id, date, is_filtered_out, filter_reason
                            ) VALUES (:pool_id, CURRENT_DATE, :is_filtered_out, :filter_reason);
                        """),
                        {
                            "pool_id": pool_id,
                            "is_filtered_out": is_filtered_out,
                            "filter_reason": '; '.join(filter_reason) if filter_reason else None
                        }
                    )

                if is_filtered_out:
                    logging.info(f"Pool {pool_id} pre-filtered out. Reasons: {'; '.join(filter_reason)}.")

            # Log final statistics
            result = conn.execute(text("""
                SELECT COUNT(*) FROM pool_daily_metrics
                WHERE date = CURRENT_DATE AND is_filtered_out = FALSE;
            """))
            approved_count = result.fetchone()[0]

            result = conn.execute(text("""
                SELECT COUNT(*) FROM pool_daily_metrics
                WHERE date = CURRENT_DATE AND is_filtered_out = TRUE;
            """))
            filtered_count = result.fetchone()[0]

            total_pools = approved_count + filtered_count

            logging.info("Pre-pool filtering completed successfully.")

            # Print detailed summary
            print("\n" + "="*60)
            print("🔍 PRE-POOL FILTERING SUMMARY")
            print("="*60)
            print(f"📊 Total pools processed: {total_pools}")
            print(f"📋 Approved protocols: {len(approved_protocols)}")
            print(f"🪙 Approved tokens: {len(approved_tokens)}")
            print(f"💰 TVL limit: ${pool_tvl_limit:,.0f}")
            print(f"📈 APY limit: {pool_apy_limit:.2%}")
            print(f"✅ Pools approved (passed pre-filtering): {approved_count}")
            print(f"❌ Pools filtered out: {filtered_count}")
            print(f"📊 Approval rate: {(approved_count/total_pools*100):.1f}%" if total_pools > 0 else "N/A")
            print("="*60)

            trans.commit()
        except Exception as e:
            logging.error(f"Error during pre-pool filtering: {e}")
            trans.rollback()
            raise
    engine.dispose()

if __name__ == "__main__":
    filter_pools_pre()