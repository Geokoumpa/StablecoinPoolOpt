import logging
import json
from collections import Counter
from database.db_utils import get_db_connection
from sqlalchemy import text
from api_clients.ethplorer_client import get_address_history

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

            # Get approved token addresses for efficient lookup
            result = conn.execute(text("SELECT token_address, token_symbol FROM approved_tokens WHERE token_address IS NOT NULL;"))
            approved_address_to_symbol = {row[0].lower(): row[1] for row in result.fetchall()}

            logger.info("=== Pre-Pool Filtering Criteria and Thresholds ===")
            logger.info(f"Token Marketcap Limit: {token_marketcap_limit}")
            logger.info(f"Pool Pair TVL Ratio Min: {pool_pair_tvl_ratio_min}")
            logger.info(f"Pool Pair TVL Ratio Max: {pool_pair_tvl_ratio_max}")
            logger.info(f"Approved Protocols: {approved_protocols}")
            logger.info(f"Approved Tokens: {approved_tokens}")
            
            logger.info("================================================")

            # Fetch all active pools for pre-filtering (exclude inactive pools)
            result = conn.execute(text("""
                SELECT pool_id, protocol, symbol, tvl, apy, name, chain, underlying_token_addresses, poolMeta
                FROM pools p
                WHERE is_active = TRUE;
            """))
            pools_data = result.fetchall()

            for pool_id, protocol, symbol, tvl, apy, name, chain, underlying_token_addresses, poolmeta in pools_data:
                filter_reason = []
                is_filtered_out = False
                approved_token_symbols = []

                # Exclude pools not belonging to approved protocols
                if protocol not in approved_protocols:
                    filter_reason.append(f"Protocol '{protocol}' not in approved protocols.")
                    is_filtered_out = True

                # Exclude pools not on Ethereum chain
                if chain != "Ethereum":
                    filter_reason.append(f"Pool is on chain '{chain}', not Ethereum.")
                    is_filtered_out = True

                # Enhanced filtering for pendle protocol pools
                if protocol.lower() == "pendle" and not is_filtered_out:
                    # First check if poolmeta contains "PT" or "LP"
                    if poolmeta and ("PT" in poolmeta or "LP" in poolmeta):
                        logger.info(f"Pendle pool {pool_id} ({name}) contains 'PT' or 'LP' in poolmeta: {poolmeta}")
                        
                        if underlying_token_addresses:
                            # Use Ethplorer API to find the most frequent token address
                            most_frequent_token_address = find_most_frequent_token_address(pool_id, underlying_token_addresses)
                            
                            if most_frequent_token_address:
                                # Check if this token address is in our approved tokens
                                if most_frequent_token_address.lower() in approved_address_to_symbol:
                                    token_symbol = approved_address_to_symbol[most_frequent_token_address.lower()]
                                    approved_token_symbols = [token_symbol]
                                    logger.info(f"Pendle pool {pool_id} matched with approved token: {token_symbol} (address: {most_frequent_token_address})")
                                else:
                                    filter_reason.append(f"Most frequent token address {most_frequent_token_address} not in approved tokens")
                                    is_filtered_out = True
                            else:
                                filter_reason.append("Could not determine most frequent token address from Ethplorer API")
                                is_filtered_out = True
                        else:
                            filter_reason.append("Pendle pool has no underlying token addresses")
                            is_filtered_out = True
                    else:
                        filter_reason.append("Pendle pool poolmeta does not contain 'PT' or 'LP'")
                        is_filtered_out = True
                else:
                    # Check token composition using address-based filtering for non-pendle protocols
                    if underlying_token_addresses and not is_filtered_out:
                        # Check each underlying token address against approved addresses
                        for address in underlying_token_addresses:
                            if address.lower() in approved_address_to_symbol:
                                approved_token_symbols.append(approved_address_to_symbol[address.lower()])
                            else:
                                filter_reason.append(f"Pool contains unapproved token address: {address}")
                                is_filtered_out = True
                    elif not is_filtered_out:
                        filter_reason.append("Pool has no underlying token addresses.")
                        is_filtered_out = True

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
                
                # Only create/update pool_daily_metrics for active pools
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

def find_most_frequent_token_address(pool_id, token_addresses):
    """
    For a list of token addresses, uses Ethplorer API to find the most frequent token address
    based on transaction history.
    
    Args:
        pool_id: The pool ID for logging purposes
        token_addresses: List of token addresses to analyze
        
    Returns:
        The most frequent token address as a string, or None if no transactions found
    """
    token_address_counts = Counter()
    logger.info(f"Analyzing {len(token_addresses)} underlying token addresses for pool {pool_id}")
    
    for address in token_addresses:
        try:
            logger.info(f"Fetching transaction history for address {address} in pool {pool_id}")
            # Get transaction history for this address
            history_data = get_address_history(address, limit=100)
            
            if history_data and 'operations' in history_data:
                operations_count = len(history_data['operations'])
                logger.info(f"Found {operations_count} operations for address {address} in pool {pool_id}")
                
                for operation in history_data['operations']:
                    if 'tokenInfo' in operation and 'address' in operation['tokenInfo']:
                        token_address = operation['tokenInfo']['address']
                        token_address_counts[token_address] += 1
                        logger.debug(f"Token address {token_address} count increased to {token_address_counts[token_address]}")
            else:
                logger.warning(f"No transaction history found for address {address} in pool {pool_id}")
                
        except Exception as e:
            logger.error(f"Error fetching transaction history for address {address} in pool {pool_id}: {e}")
            continue
    
    if token_address_counts:
        most_common_address, count = token_address_counts.most_common(1)[0]
        logger.info(f"Most frequent token address for pool {pool_id}: {most_common_address} (appeared {count} times)")
        return most_common_address
    else:
        logger.warning(f"No token addresses found in transaction history for pool {pool_id}")
        return None

if __name__ == "__main__":
    filter_pools_pre()