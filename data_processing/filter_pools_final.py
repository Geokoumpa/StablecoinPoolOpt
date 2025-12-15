import logging
from datetime import date
from database.repositories.pool_metrics_repository import PoolMetricsRepository
from database.repositories.parameter_repository import ParameterRepository
from database.repositories.token_repository import TokenRepository

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

def filter_pools_final():
    """
    Final filtering phase: Apply TVL, APY, and icebox token filtering to pre-filtered pools.
    Only processes pools that passed pre-filtering and adds final exclusions.
    Now uses forecasted values that are available from the forecasting phase.
    """
    logger.info("Starting final pool filtering (icebox)...")
    
    # Initialize repositories
    metrics_repo = PoolMetricsRepository()
    param_repo = ParameterRepository()
    token_repo = TokenRepository()

    try:
        # Get icebox tokens (only active ones)
        icebox_tokens_objs = token_repo.get_icebox_tokens()
        icebox_tokens = {t.token_symbol for t in icebox_tokens_objs}

        # Get allocation parameters for TVL and APY limits
        filter_params = param_repo.get_latest_parameters()
        
        if filter_params:
            pool_tvl_limit = filter_params.pool_tvl_limit
            pool_apy_limit = filter_params.pool_apy_limit
        else:
            pool_tvl_limit, pool_apy_limit = None, None
            logger.warning("No allocation parameters found for TVL/APY filtering.")

        logger.info("=== Final Pool Filtering Criteria ===")
        logger.info(f"Icebox Tokens: {icebox_tokens}")
        logger.info(f"Pool TVL Limit: {pool_tvl_limit}")
        logger.info(f"Pool APY Limit: {pool_apy_limit}")
        logger.info("Using forecasted values for filtering decisions")
        logger.info("=====================================")

        # Fetch pools that passed pre-filtering (not filtered out)
        pre_filtered_pools = metrics_repo.get_pre_filtered_pools_with_forecasts(date.today())

        logger.info(f"Processing {len(pre_filtered_pools)} pre-filtered pools for final filtering...")

        # Validate that forecasted values are available
        pools_missing_forecasts = []
        for pool_id, symbol, forecasted_tvl, forecasted_apy, existing_reason in pre_filtered_pools:
            if forecasted_tvl is None or forecasted_apy is None:
                pools_missing_forecasts.append((pool_id, symbol, forecasted_tvl, forecasted_apy))
        
        if pools_missing_forecasts:
            logger.error(f"Found {len(pools_missing_forecasts)} pools missing forecasted values! These will be filtered out.")
            for pool_id, symbol, tvl, apy in pools_missing_forecasts[:5]:  # Log first 5
                logger.error(f"Pool {pool_id} ({symbol}): TVL={tvl}, APY={apy}")
            if len(pools_missing_forecasts) > 5:
                logger.error(f"... and {len(pools_missing_forecasts) - 5} more pools")

        updates = []

        for pool_id, symbol, forecasted_tvl, forecasted_apy, existing_reason in pre_filtered_pools:
            additional_reasons = []
            should_filter_out = False

            # Check for missing forecasts
            if forecasted_tvl is None or forecasted_apy is None:
                additional_reasons.append("Forecasted values missing")
                should_filter_out = True

            # Check for icebox tokens
            if symbol:
                icebox_found = [token for token in icebox_tokens if token in symbol]
                if icebox_found:
                    additional_reasons.append(f"Pool contains icebox token(s): {', '.join(icebox_found)}.")
                    should_filter_out = True

            # Check TVL limit using forecasted values
            if forecasted_tvl is not None and pool_tvl_limit is not None and forecasted_tvl < pool_tvl_limit:
                additional_reasons.append(f"Pool forecasted TVL ({forecasted_tvl:.2f}) below limit ({pool_tvl_limit}).")
                should_filter_out = True

            # Check APY limit using forecasted values
            # Note: forecasted_apy is stored as percentage (0.0633 = 0.0633%)
            # pool_apy_limit is stored as decimal (0.0600 = 6% = 6.0%)
            if pool_apy_limit is not None:
                apy_limit_percentage = pool_apy_limit * 100  # Convert 0.0600 to 6.0
                if forecasted_apy is not None and forecasted_apy < apy_limit_percentage:
                    additional_reasons.append(f"Pool forecasted APY ({forecasted_apy:.4f}%) below limit ({apy_limit_percentage:.2f}%).")
                    should_filter_out = True

            if should_filter_out:
                # Combine existing reasons with new ones
                combined_reasons = []
                if existing_reason:
                    combined_reasons.append(existing_reason)
                combined_reasons.extend(additional_reasons)
                
                final_reason = '; '.join(combined_reasons)
                
                updates.append((pool_id, date.today(), True, final_reason)) # is_filtered_out = True
                
                logger.info(f"Pool {pool_id} finally filtered out. Reasons: {'; '.join(additional_reasons)}.")

        if updates:
            metrics_repo.bulk_update_icebox_status(updates)
            logger.info(f"Updated filtering status for {len(updates)} pools.")

        # Log final statistics
        final_approved_ids = metrics_repo.get_filtered_pool_ids_for_date(date.today(), is_filtered_out=False)
        final_filtered_ids = metrics_repo.get_filtered_pool_ids_for_date(date.today(), is_filtered_out=True)
        
        final_count = len(final_approved_ids)
        filtered_count = len(final_filtered_ids)

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
        raise

if __name__ == "__main__":
    filter_pools_final()