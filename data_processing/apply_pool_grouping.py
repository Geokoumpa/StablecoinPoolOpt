import logging
from datetime import date
from database.db_utils import get_db_connection

logger = logging.getLogger(__name__)

def apply_pool_grouping():
    """
    Applies pool grouping logic based on calculated metrics and stores assignments.
    """
    logger.info("Starting pool grouping application...")
    conn = None
    try:
        conn = get_db_connection()
        from sqlalchemy import text
        with conn.begin() as connection:
            # Fetch latest allocation parameters for grouping thresholds
            result = connection.execute(text("""
                SELECT
                    group1_apy_delta_max,
                    group1_7d_stddev_max,
                    group1_30d_stddev_max,
                    group2_apy_delta_max,
                    group2_7d_stddev_max,
                    group2_30d_stddev_max,
                    group3_apy_delta_min,
                    group3_7d_stddev_min,
                    group3_30d_stddev_min
                FROM allocation_parameters
                ORDER BY timestamp DESC LIMIT 1;
            """))
            grouping_params = result.fetchone()

            if not grouping_params:
                logging.warning("No allocation parameters found for grouping. Skipping pool grouping.")
                return

            (group1_apy_delta_max, group1_7d_stddev_max, group1_30d_stddev_max,
             group2_apy_delta_max, group2_7d_stddev_max, group2_30d_stddev_max,
             group3_apy_delta_min, group3_7d_stddev_min, group3_30d_stddev_min) = grouping_params

            # Fetch calculated metrics from pool_daily_metrics
            result = connection.execute(text("""
                SELECT
                    pool_id,
                    date,
                    apy_delta_today_yesterday,
                    stddev_apy_7d_delta,
                    stddev_apy_30d_delta,
                    stddev_apy_7d, -- Mean volatility for 7d
                    stddev_apy_30d -- Mean volatility for 30d
                FROM pool_daily_metrics
                WHERE date = CURRENT_DATE;
            """))
            pool_metrics_data = result.fetchall()

            if not pool_metrics_data:
                logging.warning("No pool daily metrics found for today. Skipping pool grouping.")
                return

            # Prepare data for batch update
            update_data = []
            for pool_id, metric_date, apy_delta, stddev_7d_delta, stddev_30d_delta, mean_vol_7d, mean_vol_30d in pool_metrics_data:
                group_assignment = None # Default to None if no group matches

                # Group 1: ΔAPY ≤ 1% AND Δ7DSTDDEV ≤ 1.5% AND Δ30DSTDDEV ≤ 2%
                if (apy_delta is not None and apy_delta <= group1_apy_delta_max and
                    stddev_7d_delta is not None and stddev_7d_delta <= group1_7d_stddev_max and
                    stddev_30d_delta is not None and stddev_30d_delta <= group1_30d_stddev_max):
                    group_assignment = 1
                # Group 2: ΔAPY ≤ 3% AND Δ7DSTDDEV ≤ 4% AND Δ30DSTDDEV ≤ 5%
                elif (apy_delta is not None and apy_delta <= group2_apy_delta_max and
                      stddev_7d_delta is not None and stddev_7d_delta <= group2_7d_stddev_max and
                      stddev_30d_delta is not None and stddev_30d_delta <= group2_30d_stddev_max):
                    group_assignment = 2
                # Group 3: ΔAPY > 3% AND Δ7DSTDDEV > 4% AND Δ30DSTDDEV > 2%
                elif (apy_delta is not None and apy_delta > group3_apy_delta_min and
                      stddev_7d_delta is not None and stddev_7d_delta > group3_7d_stddev_min and
                      stddev_30d_delta is not None and stddev_30d_delta > group3_30d_stddev_min):
                    group_assignment = 3
                else:
                    group_assignment = 4 # Default or other criteria

                update_data.append((group_assignment, pool_id, metric_date))

            # Update pool_daily_metrics with group assignments using ON CONFLICT for idempotency
            upsert_query = text("""
                INSERT INTO pool_daily_metrics (pool_id, date, pool_group)
                VALUES (:pool_id, :date, :pool_group)
                ON CONFLICT (pool_id, date) DO UPDATE SET
                    pool_group = EXCLUDED.pool_group;
            """)

            # Reformat data for upsert
            upsert_data = [{"pool_id": pool_id, "date": metric_date, "pool_group": group_assignment} for group_assignment, pool_id, metric_date in update_data]

            for params in upsert_data:
                connection.execute(upsert_query, params)

            # Calculate group statistics for summary
            group_stats = {}
            for group_assignment, pool_id, metric_date in update_data:
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

            logger.info(f"Upserted {len(update_data)} pool daily metrics with group assignments.")

    except Exception as e:
        logger.error(f"Error during pool grouping application: {e}")
    finally:
        if conn:
            conn.dispose()

if __name__ == "__main__":
    apply_pool_grouping()