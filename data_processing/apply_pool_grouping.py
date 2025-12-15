import logging
from datetime import date
from database.repositories.pool_metrics_repository import PoolMetricsRepository
from database.repositories.parameter_repository import ParameterRepository

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

def apply_pool_grouping():
    """
    Applies pool grouping logic based on calculated metrics and stores assignments.
    """
    logger.info("Starting pool grouping application...")
    
    metrics_repo = PoolMetricsRepository()
    param_repo = ParameterRepository()
    
    try:
        # Fetch latest allocation parameters for grouping thresholds
        grouping_params = param_repo.get_latest_parameters()

        if not grouping_params:
            logging.warning("No allocation parameters found for grouping. Skipping pool grouping.")
            return

        group1_apy_delta_max = grouping_params.group1_apy_delta_max
        group1_7d_stddev_max = grouping_params.group1_7d_stddev_max
        group1_30d_stddev_max = grouping_params.group1_30d_stddev_max
        group2_apy_delta_max = grouping_params.group2_apy_delta_max
        group2_7d_stddev_max = grouping_params.group2_7d_stddev_max
        group2_30d_stddev_max = grouping_params.group2_30d_stddev_max
        group3_apy_delta_min = grouping_params.group3_apy_delta_min
        group3_7d_stddev_min = grouping_params.group3_7d_stddev_min
        group3_30d_stddev_min = grouping_params.group3_30d_stddev_min

        # Fetch calculated metrics from pool_daily_metrics only for active pools
        # Returns list of (pool_id, date, apy_delta, stddev_7d_delta, stddev_30d_delta, stddev_apy_7d, stddev_apy_30d)
        pool_metrics_data = metrics_repo.get_metrics_for_grouping(date.today())

        if not pool_metrics_data:
            logging.warning("No pool daily metrics found for today. Skipping pool grouping.")
            return

        # Prepare data for batch update
        update_data = []
        for pool_id, metric_date, apy_delta, stddev_7d_delta, stddev_30d_delta, _, _ in pool_metrics_data:
            group_assignment = None # Default to None if no group matches

            # Group 1: ΔAPY ≤ 1% AND Δ7DSTDDEV ≤ 1.5% AND Δ30DSTDDEV ≤ 2%
            if (apy_delta is not None and group1_apy_delta_max is not None and apy_delta <= group1_apy_delta_max and
                stddev_7d_delta is not None and group1_7d_stddev_max is not None and stddev_7d_delta <= group1_7d_stddev_max and
                stddev_30d_delta is not None and group1_30d_stddev_max is not None and stddev_30d_delta <= group1_30d_stddev_max):
                group_assignment = 1
            # Group 2: ΔAPY ≤ 3% AND Δ7DSTDDEV ≤ 4% AND Δ30DSTDDEV ≤ 5%
            elif (apy_delta is not None and group2_apy_delta_max is not None and apy_delta <= group2_apy_delta_max and
                  stddev_7d_delta is not None and group2_7d_stddev_max is not None and stddev_7d_delta <= group2_7d_stddev_max and
                  stddev_30d_delta is not None and group2_30d_stddev_max is not None and stddev_30d_delta <= group2_30d_stddev_max):
                group_assignment = 2
            # Group 3: ΔAPY > 3% AND Δ7DSTDDEV > 4% AND Δ30DSTDDEV > 2%
            elif (apy_delta is not None and group3_apy_delta_min is not None and apy_delta > group3_apy_delta_min and
                  stddev_7d_delta is not None and group3_7d_stddev_min is not None and stddev_7d_delta > group3_7d_stddev_min and
                  stddev_30d_delta is not None and group3_30d_stddev_min is not None and stddev_30d_delta > group3_30d_stddev_min):
                group_assignment = 3
            else:
                group_assignment = 4 # Default or other criteria

            update_data.append((pool_id, metric_date, group_assignment))

        # Update pool_daily_metrics with group assignments
        # Use bulk_update_groups which expects (pool_id, date, group_id)
        if update_data:
            metrics_repo.bulk_update_groups(update_data)
            logger.info(f"Upserted {len(update_data)} pool daily metrics with group assignments.")

        # Calculate group statistics for summary
        group_stats = {}
        for pool_id, metric_date, group_assignment in update_data:
            group_stats[group_assignment] = group_stats.get(group_assignment, 0) + 1

        # Print comprehensive summary
        logger.info("\n" + "="*60)
        logger.info("🎯 POOL GROUPING SUMMARY")
        logger.info("="*60)
        logger.info(f"📊 Total pools processed: {len(update_data)}")
        logger.info(f"📅 Date: {date.today()}")
        logger.info("")
        logger.info("📈 Group Assignments:")
        for group in sorted(group_stats.keys()):
            count = group_stats[group]
            if group == 1:
                logger.info(f"   🟢 Group 1 (Low Risk): {count} pools")
            elif group == 2:
                logger.info(f"   🟡 Group 2 (Medium Risk): {count} pools")
            elif group == 3:
                logger.info(f"   🟠 Group 3 (High Risk): {count} pools")
            else:
                logger.info(f"   ⚪️ Group 4 (Other): {count} pools")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Error during pool grouping application: {e}")
        raise

if __name__ == "__main__":
    apply_pool_grouping()