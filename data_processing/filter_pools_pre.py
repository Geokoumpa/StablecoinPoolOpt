
import logging

from collections import Counter
from datetime import datetime, timezone, date
from api_clients.ethplorer_client import get_address_history, get_tx_info, get_token_history
from api_clients.pendle_client import get_sdk_tokens, get_all_markets
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

        # Pre-fetch Pendle Market Map for optimization
        logger.info("Fetching Pendle Market Map...")
        pendle_market_map = get_all_markets(chain_id=1) or {}

        # Fetch all active pools for pre-filtering
        pools_data = pool_repo.get_active_pools()

        token_updates = []
        address_updates = [] # New: Collect updates for underlying token addresses
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
            underlying_token_addresses = pool.underlying_token_addresses or []
            poolmeta = pool.pool_meta
            pool_address = pool.pool_address

            filter_reason = []
            is_filtered_out = False
            approved_token_symbols = []
            
            # Helper to store new addresses for this pool if we find them
            rescued_addresses = []

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
                
                # 1. Check Local Data First: Do we already have approved tokens associated?
                local_approved_candidates = [
                    t for t in underlying_token_addresses 
                    if t.lower() in approved_address_to_symbol
                ]

                if local_approved_candidates:
                     # Already approved locally (e.g. from previous rescue run)
                     # Note: we don't need to push address update here, it's already local
                     approved_token_symbols = [approved_address_to_symbol[t.lower()] for t in local_approved_candidates]
                     logger.info(f"Pendle pool {pool_id} ({name}) approved via LOCAL data: {approved_token_symbols}")
                
                else:
                    # 2. Not found locally, attempt Rescue via API Map & Analyis
                    # Only if poolmeta indicates it's likely a Pendle market
                    if poolmeta and ("PT" in poolmeta or "LP" in poolmeta):
                         logger.info(f"Pendle pool {pool_id} missing valid underlying. Checking API Map...")
                         
                         rescued_symbol, rescued_address = identify_pendle_underlying(pool_id, pool_address, pendle_market_map, approved_address_to_symbol)
                         
                         if rescued_symbol and rescued_address:
                             approved_token_symbols = [rescued_symbol]
                             rescued_addresses = [rescued_address] # We want to overwrite with the true underlying
                         else:
                             filter_reason.append(f"Pendle pool has no valid underlying token (Local or API Analysis failed)")
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
                                # Note: For non-pendle, we are not actively 'rescuing' missing addresses here in the same way, 
                                # but we COULD update addresses if we wanted. 
                                # Logic: If `underlying_token_addresses` was empty, we found one. We should save it.
                                rescued_addresses = [most_frequent_token_address] 
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

            # Update pools table with approved token symbols 
            if approved_token_symbols:
                token_updates.append((pool_id, approved_token_symbols))
            
            # Update pools table with underlying addresses (if we found new ones/overwrote them)
            # Note: We only update if we genuinely found new info (pendle rescue or frequent token scan)
            # Existing addresses are assumed fine if we didn't touch them.
            if rescued_addresses:
                 address_updates.append((pool_id, rescued_addresses))

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
            
        if address_updates:
            pool_repo.bulk_update_underlying_token_addresses(address_updates)
            
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

def identify_pendle_underlying(pool_id, pool_address, pendle_market_map, approved_address_to_symbol):
    """
    Identifies the underlying token for a Pendle pool using the market map and Mint Analysis.
    
    1. Looks up `pool_address` in `pendle_market_map`.
    2. If found, gets the `asset_address`.
    3. If `asset_address` is APPROVED: returns its symbol.
    4. If not (e.g. reUSDe): Performs Mint Analysis on `asset_address` (treating it as derivative)
       to find if it was minted using an APPROVED token (e.g. USDe).
    
    Returns (Approved Symbol, Approved Address) or (None, None).
    """
    logger.info(f"Identifying underlying for Pendle pool {pool_id}...")
    
    if not pool_address:
        return None, None
        
    # 1. Look up in Map
    asset_address = pendle_market_map.get(pool_address.lower())
    if not asset_address:
         logger.warning(f"Pool {pool_address} not found in Pendle Market Map.")
         # Fallback to direct tokensIn call? No, logic dictates map is source of truth.
         return None, None
         
    logger.info(f"Pool {pool_id} maps to asset: {asset_address}")
    
    # 2. Check Direct Approval
    if asset_address.lower() in approved_address_to_symbol:
         symbol = approved_address_to_symbol[asset_address.lower()]
         logger.info(f"Asset {asset_address} is DIRECTLY APPROVED: {symbol}")
         return symbol, asset_address
         
    # 3. Derivative Rescue (Mint Analysis)
    logger.info(f"Asset {asset_address} is NOT approved. Analyzing as derivative (Mint Analysis)...")
    
    try:
        # Fetch Issuance History for this Asset
        history = get_token_history(asset_address, type_='issuance', limit=20)
        
        if not history or 'operations' not in history or not history['operations']:
             # Fallback to transfer from 0x0
             history = get_token_history(asset_address, type_='transfer', limit=20)
             
        if not history or 'operations' not in history:
             return None, None
             
        mint_tx_hashes = []
        zero_addr = '0x0000000000000000000000000000000000000000'
        
        for op in history['operations']:
            op_type = op.get('type')
            from_addr = op.get('from', '').lower()
            if op_type == 'issuance' or op_type == 'mint' or (op_type == 'transfer' and from_addr == zero_addr):
                 mint_tx_hashes.append(op.get('transactionHash'))

        unique_mints = list(set(mint_tx_hashes))[:3]
        
        for tx_hash in unique_mints:
             tx_info = get_tx_info(tx_hash)
             if not tx_info or 'operations' not in tx_info: continue
             
             for op in tx_info['operations']:
                 if 'tokenInfo' in op and 'address' in op['tokenInfo']:
                     ta = op['tokenInfo']['address'].lower()
                     if ta in approved_address_to_symbol:
                         symbol = approved_address_to_symbol[ta]
                         logger.info(f"RESCUED via Mint Analysis: {asset_address} -> {symbol} (tx {tx_hash})")
                         return symbol, ta
    except Exception as e:
        logger.error(f"Error in identify_pendle_underlying for {pool_id}: {e}")
        
    return None, None

if __name__ == "__main__":
    filter_pools_pre()
