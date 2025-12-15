import logging

from collections import Counter
from datetime import datetime, timezone, date
from api_clients.ethplorer_client import get_address_history
from database.repositories.pool_repository import PoolRepository
from database.repositories.pool_metrics_repository import PoolMetricsRepository
from database.repositories.parameter_repository import ParameterRepository
from database.repositories.token_repository import TokenRepository

logger = logging.getLogger(__name__)

def filter_pools_pre():
    """
    Pre-filtering phase: Filter pools based on approved protocols, and approved tokens.
    Does NOT filter based on TVL, APY, or icebox tokens (those come from final filtering).
    """
    logger.info("Starting pre-pool filtering...")
    
    # Initialize Repositories
    pool_repo = PoolRepository()
    metrics_repo = PoolMetricsRepository()
    param_repo = ParameterRepository()
    token_repo = TokenRepository()

    try:
        # Reset currently_filtered_out flag for all pools
        pool_repo.reset_all_currently_filtered_out()

        # Fetch basic allocation parameters
        filter_params = param_repo.get_latest_parameters()

        if not filter_params:
            logger.warning("No allocation parameters found for filtering. Skipping pre-pool filtering.")
            return

        token_marketcap_limit = filter_params.token_marketcap_limit
        pool_pair_tvl_ratio_min = filter_params.pool_pair_tvl_ratio_min
        pool_pair_tvl_ratio_max = filter_params.pool_pair_tvl_ratio_max
        pool_tvl_limit = filter_params.pool_tvl_limit if filter_params.pool_tvl_limit else 500000

        # Get approved protocols
        approved_protocols_list = pool_repo.get_approved_protocols()
        approved_protocols = set(approved_protocols_list)

        # Get approved tokens
        approved_tokens_objs = token_repo.get_approved_tokens()
        approved_tokens = {t.token_symbol for t in approved_tokens_objs}
        
        # Approved token addresses for efficient lookup
        approved_address_to_symbol = {t.token_address.lower(): t.token_symbol for t in approved_tokens_objs if t.token_address}

        logger.info("=== Pre-Pool Filtering Criteria and Thresholds ===")
        logger.info(f"Token Marketcap Limit: {token_marketcap_limit}")
        logger.info(f"Pool Pair TVL Ratio Min: {pool_pair_tvl_ratio_min}")
        logger.info(f"Pool Pair TVL Ratio Max: {pool_pair_tvl_ratio_max}")
        logger.info(f"Pool TVL Limit (Optimization): {pool_tvl_limit}")
        logger.info(f"Approved Protocols: {approved_protocols}")
        logger.info(f"Approved Tokens: {approved_tokens}")
        logger.info("================================================")

        # Fetch all active pools for pre-filtering
        pools_data = pool_repo.get_active_pools()

        token_updates = []
        metrics_updates = []
        filtered_ids = []
        
        approved_count = 0
        filtered_count = 0

        for pool in pools_data:
            pool_id = pool.pool_id
            protocol = pool.protocol
            symbol = pool.symbol
            tvl = pool.tvl
            apy = pool.apy
            name = pool.name
            chain = pool.chain
            underlying_token_addresses = pool.underlying_token_addresses
            poolmeta = pool.pool_meta
            pool_address = pool.pool_address

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
                    # Pool has no underlying token addresses
                    # Check TVL first to avoid expensive API calls for small pools
                    if tvl and tvl < pool_tvl_limit:
                         filter_reason.append(f"Pool has no underlying tokens and TVL {tvl} < {pool_tvl_limit}, skipping API lookup")
                         is_filtered_out = True
                    # If TVL is high enough, try to use pool address to find frequent tokens
                    elif pool_address:
                        logger.info(f"Pool {pool_id} has no underlying token addresses, attempting to find frequent tokens from pool address: {pool_address}")
                        most_frequent_token_address = find_most_frequent_token_address(pool_id, [pool_address])
                        
                        if most_frequent_token_address:
                            # Check if this token address is in our approved tokens
                            if most_frequent_token_address.lower() in approved_address_to_symbol:
                                token_symbol = approved_address_to_symbol[most_frequent_token_address.lower()]
                                approved_token_symbols = [token_symbol]
                                logger.info(f"Pool {pool_id} matched with approved token from pool address: {token_symbol} (address: {most_frequent_token_address})")
                            else:
                                filter_reason.append(f"Most frequent token address {most_frequent_token_address} from pool address not in approved tokens")
                                is_filtered_out = True
                        else:
                            filter_reason.append("Could not determine most frequent token address from pool address")
                            is_filtered_out = True
                    else:
                        filter_reason.append("Pool has no underlying token addresses and no pool address available")
                        is_filtered_out = True

            # Update pools table with approved token symbols for approved pools
            if approved_token_symbols:
                token_updates.append((pool_id, approved_token_symbols))
            
            # Prepare metric update
            metrics_updates.append({
                'pool_id': pool_id,
                'date': datetime.now(timezone.utc).date(),
                'is_filtered_out': is_filtered_out,
                'filter_reason': '; '.join(filter_reason) if filter_reason else None
            })

            if is_filtered_out:
                filtered_ids.append(pool_id)
                filtered_count += 1
            else:
                approved_count += 1

        # Perform updates
        if token_updates:
            pool_repo.bulk_update_underlying_tokens(token_updates)
            
        if metrics_updates:
            metrics_repo.bulk_upsert_filtering_status(metrics_updates)
            
        if filtered_ids:
            pool_repo.bulk_update_currently_filtered_out(filtered_ids)

        logger.info("Pre-pool filtering completed successfully.")
        
        # Summary
        total_pools = approved_count + filtered_count
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

    except Exception as e:
        logger.error(f"Error during pre-pool filtering: {e}")
        raise

def find_most_frequent_token_address(pool_id, token_addresses):
    """
    For a list of token addresses, uses Ethplorer API to find the most frequent token address
    based on transaction history.
    """
    token_address_counts = Counter()
    logger.info(f"Analyzing {len(token_addresses)} addresses for pool {pool_id}")
    
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
