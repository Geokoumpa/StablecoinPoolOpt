
from typing import List, Optional, Dict, Any, Tuple
from datetime import date
from sqlalchemy import select
from database.models.pool_metrics import PoolMetrics
from database.repositories.base_repository import BaseRepository

class PoolMetricsRepository(BaseRepository[PoolMetrics]):
    """
    Repository for PoolMetrics entity operations.
    """
    def __init__(self):
        super().__init__(model_class=PoolMetrics)

    def bulk_upsert_metrics(self, metrics_data: List[Dict[str, Any]]) -> None:
        """
        Bulk upsert pool metrics.
        """
        if not metrics_data:
            return

        sql = """
            INSERT INTO pool_daily_metrics (
                pool_id, date, actual_apy, actual_tvl, 
                eth_open, btc_open, gas_price_gwei,
                is_filtered_out, filter_reason,
                rolling_apy_7d, rolling_apy_30d, apy_delta_today_yesterday,
                stddev_apy_7d, stddev_apy_30d, stddev_apy_7d_delta, stddev_apy_30d_delta,
                pool_group
            ) VALUES %s
            ON CONFLICT (pool_id, date) DO UPDATE SET
                actual_apy = EXCLUDED.actual_apy,
                actual_tvl = EXCLUDED.actual_tvl,
                eth_open = EXCLUDED.eth_open,
                btc_open = EXCLUDED.btc_open,
                gas_price_gwei = EXCLUDED.gas_price_gwei,
                is_filtered_out = EXCLUDED.is_filtered_out,
                filter_reason = EXCLUDED.filter_reason,
                rolling_apy_7d = EXCLUDED.rolling_apy_7d,
                rolling_apy_30d = EXCLUDED.rolling_apy_30d,
                apy_delta_today_yesterday = EXCLUDED.apy_delta_today_yesterday,
                stddev_apy_7d = EXCLUDED.stddev_apy_7d,
                stddev_apy_30d = EXCLUDED.stddev_apy_30d,
                stddev_apy_7d_delta = EXCLUDED.stddev_apy_7d_delta,
                stddev_apy_30d_delta = EXCLUDED.stddev_apy_30d_delta,
                pool_group = EXCLUDED.pool_group
        """
        
        values = [
            (
                m['pool_id'], m['date'], m.get('actual_apy'), m.get('actual_tvl'),
                m.get('eth_open'), m.get('btc_open'), m.get('gas_price_gwei'),
                m.get('is_filtered_out', False), m.get('filter_reason'),
                m.get('rolling_apy_7d'), m.get('rolling_apy_30d'), m.get('apy_delta_today_yesterday'),
                m.get('stddev_apy_7d'), m.get('stddev_apy_30d'), m.get('stddev_apy_7d_delta'), 
                m.get('stddev_apy_30d_delta'), m.get('pool_group')
            )
            for m in metrics_data
        ]
        
        self.execute_bulk_values(sql, values)

    def get_pool_history(self, pool_id: str, start_date: Optional[date] = None, end_date: Optional[date] = None) -> List[PoolMetrics]:
        """Get historical metrics for a pool."""
        with self.session() as session:
            stmt = select(PoolMetrics).where(PoolMetrics.pool_id == pool_id)
            if start_date:
                stmt = stmt.where(PoolMetrics.date >= start_date)
            if end_date:
                stmt = stmt.where(PoolMetrics.date <= end_date)
            stmt = stmt.order_by(PoolMetrics.date)
            return session.execute(stmt).scalars().all()

    def bulk_update_forecasts(self, forecasts_data: List[Dict[str, Any]]) -> None:
        """
        Update forecasted APY and TVL for existing metrics records.
        """
        if not forecasts_data:
            return

        sql = """
            INSERT INTO pool_daily_metrics (
                pool_id, date, forecasted_apy, forecasted_tvl
            ) VALUES %s
            ON CONFLICT (pool_id, date) DO UPDATE SET
                forecasted_apy = EXCLUDED.forecasted_apy,
                forecasted_tvl = EXCLUDED.forecasted_tvl
        """
        
        values = [
            (f['pool_id'], f['date'], f.get('forecasted_apy'), f.get('forecasted_tvl'))
            for f in forecasts_data
        ]
        
        self.execute_bulk_values(sql, values)

    def get_training_data(self, pool_id: str = None) -> List[PoolMetrics]:
        """Get data with actual values for training."""
        with self.session() as session:
            stmt = select(PoolMetrics).where(PoolMetrics.actual_apy.is_not(None))
            if pool_id:
                stmt = stmt.where(PoolMetrics.pool_id == pool_id)
            stmt = stmt.order_by(PoolMetrics.pool_id, PoolMetrics.date)
            return session.execute(stmt).scalars().all()
            
    def bulk_update_groups(self, groups_data: List[Tuple[str, date, int]]) -> None:
        """
        Update pool groups using bulk update.
        groups_data: List of (pool_id, date, group_id)
        """
        if not groups_data:
            return

        sql = """
            UPDATE pool_daily_metrics as m
            SET pool_group = data.pool_group
            FROM (VALUES %s) AS data (pool_id, date, pool_group)
            WHERE m.pool_id = data.pool_id AND m.date = data.date
        """
        
        self.execute_bulk_values(sql, groups_data)

    def bulk_update_icebox_status(self, icebox_data: List[Tuple[str, date, bool, str]]) -> None:
        """
        Update is_filtered_out and filter_reason.
        icebox_data: List of (pool_id, date, is_filtered_out, filter_reason)
        """
        if not icebox_data:
            return

        sql = """
            UPDATE pool_daily_metrics as m
            SET is_filtered_out = data.is_filtered_out,
                filter_reason = data.filter_reason
            FROM (VALUES %s) AS data (pool_id, date, is_filtered_out, filter_reason)
            WHERE m.pool_id = data.pool_id AND m.date = data.date
        """
        self.execute_bulk_values(sql, icebox_data)
