
from typing import List, Optional, Dict, Any
from datetime import date, datetime
from sqlalchemy import select
from database.models.gas_fees import GasFeesDaily
from database.repositories.base_repository import BaseRepository

class GasFeeRepository(BaseRepository[GasFeesDaily]):
    """
    Repository for Gas Fee operations (Daily and Hourly).
    """
    def __init__(self):
        super().__init__(model_class=GasFeesDaily)

    def bulk_upsert_daily_gas(self, gas_data: List[Dict[str, Any]]) -> None:
        """
        Bulk upsert daily gas data.
        """
        if not gas_data:
            return

        sql = """
            INSERT INTO gas_fees_daily (
                date, actual_avg_gas_gwei, actual_max_gas_gwei,
                eth_open, btc_open
            ) VALUES %s
            ON CONFLICT (date) DO UPDATE SET
                actual_avg_gas_gwei = EXCLUDED.actual_avg_gas_gwei,
                actual_max_gas_gwei = EXCLUDED.actual_max_gas_gwei,
                eth_open = EXCLUDED.eth_open,
                btc_open = EXCLUDED.btc_open
        """
        
        values = [
            (
                g['date'], g.get('actual_avg_gas_gwei'), g.get('actual_max_gas_gwei'),
                g.get('eth_open'), g.get('btc_open')
            )
            for g in gas_data
        ]
        
        self.execute_bulk_values(sql, values)

    def upsert_forecasts(self, forecasts: List[Dict[str, Any]]) -> None:
        """
        Update forecasts for existing days or insert new days.
        """
        if not forecasts:
            return

        sql = """
            INSERT INTO gas_fees_daily (
                date, forecasted_avg_gas_gwei, forecasted_max_gas_gwei
            ) VALUES %s
            ON CONFLICT (date) DO UPDATE SET
                forecasted_avg_gas_gwei = EXCLUDED.forecasted_avg_gas_gwei,
                forecasted_max_gas_gwei = EXCLUDED.forecasted_max_gas_gwei
        """
        
        values = [
            (f['date'], f.get('forecasted_avg_gas_gwei'), f.get('forecasted_max_gas_gwei'))
            for f in forecasts
        ]
        
        self.execute_bulk_values(sql, values)

    def bulk_insert_hourly(self, hourly_data: List[Dict[str, Any]]) -> None:
        """Bulk insert hourly gas data."""
        if not hourly_data:
            return

        sql = """
            INSERT INTO gas_fees_hourly (timestamp, gas_price_gwei)
            VALUES %s
            ON CONFLICT (timestamp) DO UPDATE SET
                gas_price_gwei = EXCLUDED.gas_price_gwei
        """
        
        values = [(h['timestamp'], h.get('gas_price_gwei')) for h in hourly_data]
        self.execute_bulk_values(sql, values)

    def get_historical_data(self, start_date: Optional[date] = None, end_date: Optional[date] = None) -> List[GasFeesDaily]:
        """Get historical daily gas data."""
        with self.session() as session:
            stmt = select(GasFeesDaily)
            if start_date:
                stmt = stmt.where(GasFeesDaily.date >= start_date)
            if end_date:
                stmt = stmt.where(GasFeesDaily.date <= end_date)
            stmt = stmt.order_by(GasFeesDaily.date)
            results = session.execute(stmt).scalars().all()
            session.expunge_all()
            return results

    def has_daily_data_for_date(self, check_date: date) -> bool:
        """Check if daily data exists for a specific date."""
        with self.session() as session:
            from sqlalchemy import func
            stmt = select(func.count()).select_from(GasFeesDaily).where(GasFeesDaily.date == check_date)
            count = session.execute(stmt).scalar()
            return count > 0

    def get_all_daily_data(self) -> List[GasFeesDaily]:
        """Get all daily gas data."""
        return self.get_historical_data()

    def get_data_for_forecasting(self) -> List[Any]:
        """
        Get daily gas data for forecasting models.
        Returns: date, actual_avg_gas_gwei, actual_max_gas_gwei, eth_open, btc_open
        """
        from sqlalchemy import text
        sql = text("""
            SELECT
                date,
                actual_avg_gas_gwei,
                actual_max_gas_gwei,
                eth_open,
                btc_open
            FROM
                gas_fees_daily
            WHERE actual_avg_gas_gwei IS NOT NULL
              AND eth_open IS NOT NULL
              AND btc_open IS NOT NULL
            ORDER BY
                date;
        """)
        with self.session() as session:
            return session.execute(sql).fetchall()


