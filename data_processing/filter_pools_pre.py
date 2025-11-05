import logging
import json
from database.db_utils import get_db_connection
from sqlalchemy import text

logger = logging.getLogger(__name__)

def filter_pools_pre():
    """
    Pre-filtering phase: Filter pools based on approved protocols, and approved tokens.
    Does NOT filter based on TVL, APY, or icebox tokens (those come from final filtering).
    """
    logger.info("Starting pre-pool filtering...")
    engine = get_db_connection()
    if not engine:
        logger.error("Could not establish database connection. Exiting.")
        return

    with engine.connect() as conn:
        trans = conn.begin()
        try:
            # Reset currently_filtered_out flag for all pools
            conn.execute(text("UPDATE pools SET currently_filtered_out = FALSE;"))

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
                logger.warning("No allocation parameters found for filtering. Skipping pre-pool filtering.")
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

            # Get blacklisted tokens for pendle protocol filtering
            result = conn.execute(text("SELECT token_symbol FROM blacklisted_tokens WHERE removed_timestamp IS NULL;"))
            blacklisted_tokens = {row[0] for row in result.fetchall()}

            logger.info("=== Pre-Pool Filtering Criteria and Thresholds ===")
            logger.info(f"Token Marketcap Limit: {token_marketcap_limit}")
            logger.info(f"Pool Pair TVL Ratio Min: {pool_pair_tvl_ratio_min}")
            logger.info(f"Pool Pair TVL Ratio Max: {pool_pair_tvl_ratio_max}")
            logger.info(f"Approved Protocols: {approved_protocols}")
            logger.info(f"Approved Tokens: {approved_tokens}")
            logger.info(f"Blacklisted Tokens: {blacklisted_tokens}")
            
            logger.info("================================================")

            # Fetch all active pools for pre-filtering (exclude inactive pools)
            result = conn.execute(text("""
                SELECT pool_id, protocol, symbol, tvl, apy, name, chain, underlying_token_addresses
                FROM pools p
                WHERE is_active = TRUE;
            """))
            pools_data = result.fetchall()

            for pool_id, protocol, symbol, tvl, apy, name, chain, underlying_token_addresses in pools_data:
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

                # Secondary filtering for pendle protocol pools (takes precedence over other filtering)
                if protocol.lower() == "pendle":
                    pendle_matched_token = None
                    
                    # Check for blacklisted tokens first
                    for blacklisted_token in blacklisted_tokens:
                        if blacklisted_token.lower() in name.lower():
                            filter_reason.append(f"Pool name contains blacklisted token: {blacklisted_token}")
                            is_filtered_out = True
                            break
                    
                    # If not blacklisted, check for approved tokens
                    if not is_filtered_out:
                        for approved_token in approved_tokens:
                            if approved_token.lower() in name.lower():
                                pendle_matched_token = approved_token
                                break
                        
                        if pendle_matched_token:
                            # Set the underlying_tokens to the matched approved token
                            approved_token_symbols = [pendle_matched_token]
                            logger.info(f"Pendle pool {pool_id} matched with approved token: {pendle_matched_token}")
                        else:
                            filter_reason.append("Pendle pool name does not contain any approved tokens")
                            is_filtered_out = True
                else:
                    # Check token composition using address-based filtering for non-pendle protocols
                    approved_token_symbols = []
                    if underlying_token_addresses:
                        # Get all approved token addresses for efficient lookup
                        result = conn.execute(text("SELECT token_address, token_symbol FROM approved_tokens WHERE token_address IS NOT NULL;"))
                        approved_address_to_symbol = {row[0].lower(): row[1] for row in result.fetchall()}
                        
                        # Check each underlying token address against approved addresses
                        for address in underlying_token_addresses:
                            if address.lower() in approved_address_to_symbol:
                                approved_token_symbols.append(approved_address_to_symbol[address.lower()])
                            else:
                                filter_reason.append(f"Pool contains unapproved token address: {address}")
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

                # Update pools table with approved token symbols for approved pools (including pendle pools that matched)
                if approved_token_symbols:
                    conn.execute(text("""
                        UPDATE pools 
                        SET underlying_tokens = :approved_tokens
                        WHERE pool_id = :pool_id;
                    """), {
                        "approved_tokens": approved_token_symbols,
                        "pool_id": pool_id
                    })
                
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
                    logger.info(f"Pool {pool_id} pre-filtered out. Reasons: {'; '.join(filter_reason)}.")
                    conn.execute(
                        text("UPDATE pools SET currently_filtered_out = TRUE WHERE pool_id = :pool_id;"),
                        {"pool_id": pool_id}
                    )

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

            logger.info("Pre-pool filtering completed successfully.")

            # Print detailed summary
            logger.info("\n" + "="*60)
            logger.info("🔍 PRE-POOL FILTERING SUMMARY")
            logger.info("="*60)
            logger.info(f"📊 Total pools processed: {total_pools}")
            logger.info(f"📋 Approved protocols: {len(approved_protocols)}")
            logger.info(f"🪙 Approved tokens: {len(approved_tokens)}")
            
            logger.info(f"✅ Pools approved (passed pre-filtering): {approved_count}")
            logger.info(f"❌ Pools filtered out: {filtered_count}")
            logger.info(f"📊 Approval rate: {(approved_count/total_pools*100):.1f}%" if total_pools > 0 else "N/A")
            logger.info("="*60)

            trans.commit()
        except Exception as e:
            logger.error(f"Error during pre-pool filtering: {e}")
            trans.rollback()
            raise
    engine.dispose()

if __name__ == "__main__":
    filter_pools_pre()