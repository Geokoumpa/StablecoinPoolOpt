import logging
from database.db_utils import get_db_connection
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def filter_pools_pre():
    """
    Pre-filtering phase: Filter pools based on approved protocols, approved tokens, and blacklisted tokens.
    Does NOT filter based on TVL, APY, or icebox tokens (those come from final filtering).
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

            (token_marketcap_limit,
             pool_pair_tvl_ratio_min, pool_pair_tvl_ratio_max) = filter_params

            # Get approved protocols
            result = conn.execute(text("SELECT protocol_name FROM approved_protocols;"))
            approved_protocols = {row[0] for row in result.fetchall()}

            # Get approved tokens
            result = conn.execute(text("SELECT token_symbol FROM approved_tokens;"))
            approved_tokens = {row[0] for row in result.fetchall()}

            # Get blacklisted tokens
            result = conn.execute(text("SELECT token_symbol FROM blacklisted_tokens;"))
            blacklisted_tokens = {row[0] for row in result.fetchall()}

            print("=== Pre-Pool Filtering Criteria and Thresholds ===")
            print(f"Token Marketcap Limit: {token_marketcap_limit}")
            print(f"Pool Pair TVL Ratio Min: {pool_pair_tvl_ratio_min}")
            print(f"Pool Pair TVL Ratio Max: {pool_pair_tvl_ratio_max}")
            print(f"Approved Protocols: {approved_protocols}")
            print(f"Approved Tokens: {approved_tokens}")
            print(f"Blacklisted Tokens: {blacklisted_tokens}")
            print("================================================")

            # Fetch all pools for pre-filtering
            result = conn.execute(text("""
                SELECT pool_id, protocol, symbol, tvl, apy, name, chain
                FROM pools p;
            """))
            pools_data = result.fetchall()

            for pool_id, protocol, symbol, tvl, apy, name, chain in pools_data:
                filter_reason = []
                is_filtered_out = False

                # Exclude pools not belonging to approved protocols
                if protocol not in approved_protocols:
                    filter_reason.append(f"Protocol '{protocol}' not in approved protocols.")
                    is_filtered_out = True

                # Exclude pools not on Ethereum chain
                if chain != "Ethereum":
                    filter_reason.append(f"Pool is on chain '{chain}', not Ethereum.")
                    is_filtered_out = True

                # Check token composition: all tokens must be approved and not blacklisted
                if symbol:
                    tokens_in_symbol = symbol.split('-')
                    invalid_tokens = []
                    blacklisted_in_pool = []
                    
                    for token in tokens_in_symbol:
                        # Check for partial match with approved tokens
                        has_approved_match = False
                        for approved_token in approved_tokens:
                            if approved_token.lower() in token.lower() or token.lower() in approved_token.lower():
                                has_approved_match = True
                                break
                        
                        if not has_approved_match:
                            invalid_tokens.append(token)
                        
                        # Check for partial match with blacklisted tokens
                        for blacklisted_token in blacklisted_tokens:
                            if blacklisted_token.lower() in token.lower() or token.lower() in blacklisted_token.lower():
                                blacklisted_in_pool.append(token)
                                break
                    
                    if invalid_tokens:
                        filter_reason.append(f"Pool contains non-approved token(s): {', '.join(invalid_tokens)}.")
                        is_filtered_out = True
                    if blacklisted_in_pool:
                        filter_reason.append(f"Pool contains blacklisted token(s): {', '.join(blacklisted_in_pool)}.")
                        is_filtered_out = True
                else:
                    filter_reason.append("Pool has no symbol.")
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
            print(f"🛑 Blacklisted tokens: {len(blacklisted_tokens)}")
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