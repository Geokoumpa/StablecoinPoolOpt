
from typing import List, Optional, Dict, Any
from datetime import date, datetime
from sqlalchemy import select, and_
from database.models.macroeconomic_data import MacroeconomicData
from database.repositories.base_repository import BaseRepository

class MacroeconomicRepository(BaseRepository[MacroeconomicData]):
    """
    Repository for MacroeconomicData entity operations.
    """
    def __init__(self):
        super().__init__(model_class=MacroeconomicData)

    def bulk_upsert_economic_data(self, data: List[Dict[str, Any]]) -> None:
        """
        Bulk upsert macroeconomic data.
        """
        if not data:
            return

        sql = """
            INSERT INTO macroeconomic_data (
                series_id, series_name, frequency, date, value, unit, description, insertion_timestamp
            ) VALUES %s
            ON CONFLICT (series_id, date) DO UPDATE SET
                value = EXCLUDED.value,
                insertion_timestamp = EXCLUDED.insertion_timestamp
        """
        
        now_val = datetime.now()
        
        values = [
            (
                d['series_id'], d.get('series_name', 'Unknown'), d.get('frequency', 'Unknown'),
                d['date'], d.get('value'), d.get('unit'), d.get('description'), now_val
            )
            for d in data
        ]
        
        self.execute_bulk_values(sql, values)

    def get_series_data(self, series_id: str, start_date: Optional[date] = None, end_date: Optional[date] = None) -> List[MacroeconomicData]:
        """Get time series data for a specific economic indicator."""
        with self.session() as session:
            stmt = select(MacroeconomicData).where(MacroeconomicData.series_id == series_id)
            if start_date:
                stmt = stmt.where(MacroeconomicData.date >= start_date)
            if end_date:
                stmt = stmt.where(MacroeconomicData.date <= end_date)
            stmt = stmt.order_by(MacroeconomicData.date)
            return session.execute(stmt).scalars().all()

    def get_all_series_data(self, start_date: date, end_date: date) -> List[Any]:
        """Get data for all series within a date range."""
        with self.session() as session:
             # We fetch dictionary-like objects or just fields needed for pivoting
             # Using SQL text or select of specific fields for efficiency
             from sqlalchemy import text
             stmt = text("""
                SELECT series_name, frequency, date, value
                FROM macroeconomic_data
                WHERE date BETWEEN :start AND :end
                ORDER BY date, series_name
             """)
             return session.execute(stmt, {"start": start_date, "end": end_date}).fetchall()

