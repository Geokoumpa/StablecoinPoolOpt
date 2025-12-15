from typing import List, Optional, Dict, Any, Tuple
from datetime import date
from sqlalchemy import select, text
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
            results = session.execute(stmt).scalars().all()
            session.expunge_all()
            return results

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
            results = session.execute(stmt).scalars().all()
            session.expunge_all()
            return results
            
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

    def bulk_upsert_filtering_status(self, metrics_data: List[Dict[str, Any]]) -> None:
        """
        Bulk upsert filtering status.
        Inserts new row if not exists (other fields null), or updates status if exists.
        """
        if not metrics_data:
            return

        sql = """
            INSERT INTO pool_daily_metrics (
                pool_id, date, is_filtered_out, filter_reason
            ) VALUES %s
            ON CONFLICT (pool_id, date) DO UPDATE SET
                is_filtered_out = EXCLUDED.is_filtered_out,
                filter_reason = EXCLUDED.filter_reason
        """
        
        values = [
            (m['pool_id'], m['date'], m['is_filtered_out'], m.get('filter_reason'))
            for m in metrics_data
        ]
        
        self.execute_bulk_values(sql, values)

    def get_filtered_pool_ids_for_date(self, target_date: date, is_filtered_out: bool = False) -> List[str]:
        """Get pool IDs based on filtering status for a specific date."""
        with self.session() as session:
            stmt = select(PoolMetrics.pool_id).distinct().where(
                PoolMetrics.date == target_date,
                PoolMetrics.is_filtered_out == is_filtered_out
            )
            return session.execute(stmt).scalars().all()

    def get_existing_dates_for_pool(self, pool_id: str, start_date: date) -> List[date]:
        """Get existing metrics dates for a pool since start_date."""
        with self.session() as session:
            stmt = select(PoolMetrics.date).where(
                PoolMetrics.pool_id == pool_id,
                PoolMetrics.date >= start_date
            )
            return session.execute(stmt).scalars().all()

    def get_metrics_for_grouping(self, target_date: date) -> List[Any]:
        """
        Get metrics required for grouping for a specific date (active pools only).
        Returns list of (pool_id, date, apy_delta, stddev_7d_delta, stddev_30d_delta, stddev_apy_7d, stddev_apy_30d).
        """
        sql = text("""
            SELECT
                pdm.pool_id,
                pdm.date,
                pdm.apy_delta_today_yesterday,
                pdm.stddev_apy_7d_delta,
                pdm.stddev_apy_30d_delta,
                pdm.stddev_apy_7d,
                pdm.stddev_apy_30d
            FROM pool_daily_metrics pdm
            JOIN pools p ON pdm.pool_id = p.pool_id
            WHERE pdm.date = :date
              AND p.is_active = TRUE
        """)
        with self.session() as session:
            return session.execute(sql, {'date': target_date}).fetchall()


    def bulk_upsert_calculated_metrics(self, metrics_data: List[Dict[str, Any]]) -> None:
        """
        Bulk upsert calculated metrics (actuals, rolling stats, exogenous data).
        Does NOT update forecasting columns or filtering status on conflict.
        """
        if not metrics_data:
            return

        sql = """
            INSERT INTO pool_daily_metrics (
                pool_id, date, actual_apy, actual_tvl,
                rolling_apy_7d, rolling_apy_30d, apy_delta_today_yesterday,
                stddev_apy_7d, stddev_apy_30d, stddev_apy_7d_delta, stddev_apy_30d_delta,
                eth_open, btc_open, gas_price_gwei
            ) VALUES %s
            ON CONFLICT (pool_id, date) DO UPDATE SET
                actual_apy = EXCLUDED.actual_apy,
                actual_tvl = EXCLUDED.actual_tvl,
                rolling_apy_7d = EXCLUDED.rolling_apy_7d,
                rolling_apy_30d = EXCLUDED.rolling_apy_30d,
                apy_delta_today_yesterday = EXCLUDED.apy_delta_today_yesterday,
                stddev_apy_7d = EXCLUDED.stddev_apy_7d,
                stddev_apy_30d = EXCLUDED.stddev_apy_30d,
                stddev_apy_7d_delta = EXCLUDED.stddev_apy_7d_delta,
                stddev_apy_30d_delta = EXCLUDED.stddev_apy_30d_delta,
                eth_open = EXCLUDED.eth_open,
                btc_open = EXCLUDED.btc_open,
                gas_price_gwei = EXCLUDED.gas_price_gwei
        """
        
        values = [
            (
                m['pool_id'], m['date'], m.get('actual_apy'), m.get('actual_tvl'),
                m.get('rolling_apy_7d'), m.get('rolling_apy_30d'), m.get('apy_delta_today_yesterday'),
                m.get('stddev_apy_7d'), m.get('stddev_apy_30d'), m.get('stddev_apy_7d_delta'), 
                m.get('stddev_apy_30d_delta'),
                m.get('eth_open'), m.get('btc_open'), m.get('gas_price_gwei')
            )
            for m in metrics_data
        ]
        
        
        self.execute_bulk_values(sql, values)

    def get_pre_filtered_pools_with_forecasts(self, target_date: date) -> List[Any]:
        """
        Get pools that are not filtered out, with their forecasts.
        """
        sql = text("""
            SELECT pdm.pool_id, p.symbol, pdm.forecasted_tvl, pdm.forecasted_apy, 
                   pdm.filter_reason
            FROM pool_daily_metrics pdm
            JOIN pools p ON pdm.pool_id = p.pool_id
            WHERE pdm.date = :date AND pdm.is_filtered_out = FALSE;
        """)
        with self.session() as session:
            return session.execute(sql, {'date': target_date}).fetchall()

    def get_active_filtered_pool_ids(self, target_date: date, is_filtered_out: bool = False) -> List[str]:
        """Get active pool IDs based on filtering status for a specific date."""
        sql = text("""
            SELECT pdm.pool_id
            FROM pool_daily_metrics pdm
            JOIN pools p ON pdm.pool_id = p.pool_id
            WHERE pdm.date = :date
              AND pdm.is_filtered_out = :filtered
              AND p.is_active = TRUE
        """)
        with self.session() as session:
             return session.execute(sql, {'date': target_date, 'filtered': is_filtered_out}).scalars().all()

    def get_panel_data(self, pool_ids: List[str], start_date: date, end_date: date, group_col: str = "pool_group") -> List[Any]:
        """
        Get panel data for forecasting.
        Returns: pool_id, date, apy_7d, actual_apy, actual_tvl, eth_open, btc_open, gas_price_gwei, pool_group
        """
        # Ensure group_col is safe (white-listed) or use parameter binding if possible, but column names usually need string formatting in existing infrastructure or careful handling. 
        # Since this is an internal repo method, we'll assume group_col is valid or default to 'pool_group'.
        # However, for safety, let's stick to standard column 'pool_group' in query unless dynamic is strictly needed.
        # The original code allowed dynamic group_col.
        
        safe_group_col = "pool_group" # Defaulting for now as it's the main case
        
        sql = text(f"""
            SELECT
                date,
                pool_id,
                rolling_apy_7d AS apy_7d,
                actual_apy,
                actual_tvl AS tvl_usd,
                eth_open,
                btc_open,
                gas_price_gwei,
                {safe_group_col}
            FROM pool_daily_metrics
            WHERE pool_id = ANY(:pool_ids)
              AND date >= :start_date
              AND date <= :end_date
            ORDER BY date ASC
        """)
        
        with self.session() as session:
             return session.execute(sql, {
                 "pool_ids": pool_ids,
                 "start_date": start_date,
                 "end_date": end_date
             }).fetchall()

    def get_realized_metrics_for_date(self, target_date: date) -> List[Any]:
        """
        Get realized metrics (actual_py, actual_tvl) for a specific date.
        Returns: pool_id, date, actual_apy, actual_tvl
        """
        sql = text("""
            SELECT pool_id, date, actual_apy, actual_tvl
            FROM pool_daily_metrics
            WHERE date = :date
        """)
        with self.session() as session:
            return session.execute(sql, {'date': target_date}).fetchall()

    def get_pool_candidates_for_optimization(self, target_date: date, allocated_pool_ids: List[str]) -> List[Any]:
        """
        Get pools for optimization. Includes active filtered pools AND any currently allocated pools.
        Returns: pool_id, symbol, chain, protocol, forecasted_apy, forecasted_tvl, underlying_tokens
        """
        sql = text("""
            SELECT
                pdm.pool_id,
                p.symbol,
                p.chain,
                p.protocol,
                pdm.forecasted_apy,
                pdm.forecasted_tvl,
                p.underlying_tokens
            FROM pool_daily_metrics pdm
            JOIN pools p ON pdm.pool_id = p.pool_id
            WHERE (
                -- Active pools meeting all criteria
                pdm.date = :date
                AND pdm.is_filtered_out = FALSE
                AND pdm.forecasted_apy IS NOT NULL
                AND pdm.forecasted_apy > 0
                AND pdm.forecasted_tvl IS NOT NULL
                AND pdm.forecasted_tvl > 0
                AND p.is_active = TRUE
            )
            OR (
                -- Already allocated pools (regardless of active status)
                pdm.pool_id = ANY(:allocated_ids)
                AND pdm.date = :date
                AND pdm.forecasted_apy IS NOT NULL
            )
        """)
        
        # Ensure list is not empty for ANY() check, passed as empty dict/None if empty in python usually works
        # but postgres ANY(NULL) or ANY('{}') logic needs care.
        # If allocated_pool_ids is empty, pass empty list which adapts to '{}'.
        ids_param = allocated_pool_ids if allocated_pool_ids else []
        
        with self.session() as session:
             return session.execute(sql, {'date': target_date, 'allocated_ids': ids_param}).fetchall()

    def get_pool_metrics_batch(self, pool_ids: List[str], target_date: date) -> List[Any]:
        """
        Get info and metrics for a list of pool IDs.
        Returns: pool_id, symbol, forecasted_apy, forecasted_tvl
        """
        if not pool_ids:
            return []
            
        sql = text("""
            SELECT p.pool_id, p.symbol, pdm.forecasted_apy, pdm.forecasted_tvl
            FROM pools p
            LEFT JOIN pool_daily_metrics pdm ON p.pool_id = pdm.pool_id 
                AND pdm.date = :date
            WHERE p.pool_id = ANY(:pool_ids)
        """)
        
        with self.session() as session:
            return session.execute(sql, {'date': target_date, 'pool_ids': pool_ids}).fetchall()









