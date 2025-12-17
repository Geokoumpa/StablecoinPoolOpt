
from typing import List, Optional, Dict, Any
from datetime import date
from sqlalchemy import select, delete, func, text, or_, and_
from database.models.daily_balance import DailyBalance
from database.repositories.base_repository import BaseRepository

class DailyBalanceRepository(BaseRepository[DailyBalance]):
    """
    Repository for DailyBalance entity operations.
    """
    def __init__(self):
        super().__init__(model_class=DailyBalance)

    def bulk_upsert_balances(self, balances_data: List[Dict[str, Any]], ensure_unique_by_date: bool = True) -> None:
        """
        Bulk upsert daily balances.
        
        Args:
            balances_data: List of dictionary data for balances
            ensure_unique_by_date: If True, deletes existing balances for the dates present in data 
                                   before inserting (Overwrite strategy), since explicit unique constraints
                                   might vary.
        """
        if not balances_data:
            return

        dates_to_clear = {b['date'] for b in balances_data}
        
        # We handle this in a transaction
        with self.transaction() as conn:
            if ensure_unique_by_date and dates_to_clear:
                # Delete existing entries for these dates to avoid duplication
                date_list = list(dates_to_clear)
                dates_sql = ", ".join(f"'{d.isoformat()}'" for d in date_list)
                delete_sql = f"DELETE FROM daily_balances WHERE date IN ({dates_sql})"
                conn.execute(text(delete_sql))

            sql = """
                INSERT INTO daily_balances (
                    date, token_symbol, wallet_address, unallocated_balance, 
                    allocated_balance, pool_id
                ) VALUES %s
            """
            
            values = [
                (
                    b['date'], b['token_symbol'], b.get('wallet_address'),
                    b.get('unallocated_balance'), b.get('allocated_balance'), b.get('pool_id')
                )
                for b in balances_data
            ]
            
            # execute_values requires a raw DBAPI cursor
            # In SQLAlchemy 1.4/2.0, access raw connection via conn.connection.dbapi_connection
            raw_conn = conn.connection.dbapi_connection
            with raw_conn.cursor() as cur:
                import psycopg2.extras
                psycopg2.extras.execute_values(
                    cur, sql, values, page_size=1000
                )


            




    def get_current_balances(self, wallet_address: str, target_date: date) -> List[DailyBalance]:
        """
        Get balances for a wallet on a date, including entries with NULL wallet address.
        """
        with self.session() as session:
            stmt = select(DailyBalance).where(
                and_(
                    DailyBalance.date == target_date,
                    or_(
                        DailyBalance.wallet_address == wallet_address,
                        DailyBalance.wallet_address.is_(None)
                    )
                )
            )
            results = session.execute(stmt).scalars().all()
            session.expunge_all()
            return results

    def get_allocated_pool_ids(self, wallet_address: str, target_date: date) -> List[str]:
        """
        Get IDs of pools that have allocations on the target date.
        """
        # We need to join with pool_daily_metrics to ensure pool is valid/exists in metrics?
        # The original query joined with pool_daily_metrics ON date AND pool_id.
        
        sql = text("""
            SELECT DISTINCT db.pool_id
            FROM daily_balances db
            WHERE db.date = :date
              AND (db.wallet_address = :wallet_address OR db.wallet_address IS NULL)
              AND db.allocated_balance > 0
        """)
        
        with self.session() as session:
            return session.execute(sql, {'date': target_date, 'wallet_address': wallet_address}).scalars().all()

    def get_total_aum(self, target_date: date) -> float:
        """
        Calculates total AUM (allocated + unallocated) for a specific date.
        """
        sql = text("""
            SELECT SUM(COALESCE(allocated_balance, 0) + COALESCE(unallocated_balance, 0))
            FROM daily_balances
            WHERE date = :date
        """)
        
        with self.session() as session:
             result = session.execute(sql, {'date': target_date}).scalar()
             return float(result) if result else 0.0


