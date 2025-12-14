
from typing import List, Optional, Dict, Any
from datetime import date
from sqlalchemy import select, delete, func, text
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
                # Constructing WHERE IN clause
                date_list = list(dates_to_clear)
                # We interpret this as: delete all balances for these dates
                # Note: This is a heavy operation if many dates are passed.
                
                # Using SQLAlchemy delete for clarity/security, or raw SQL
                # Since we are in raw transaction, better use raw SQL or careful construction
                # But self.transaction() yields a connection.
                
                # We can't easily use ORM session inside 'transaction' context manager which gives raw connection
                # So we use SQL.
                
                dates_sql = ", ".join(f"'{d.isoformat()}'" for d in date_list)
                delete_sql = f"DELETE FROM daily_balances WHERE date IN ({dates_sql})"
                conn.execute(text(delete_sql))

            sql = """
                INSERT INTO daily_balances (
                    date, token_symbol, wallet_address, unallocated_balance, 
                    allocated_balance, pool_id
                ) VALUES %s
            """
            
            # If there was a unique constraint, we could use ON CONFLICT. 
            # Given the ambiguity, the delete-insert strategy is safer for "upsert" on daily granularity.
            
            values = [
                (
                    b['date'], b['token_symbol'], b.get('wallet_address'),
                    b.get('unallocated_balance'), b.get('allocated_balance'), b.get('pool_id')
                )
                for b in balances_data
            ]
            
            # execute_values requires a cursor, which we can get from connection
            with conn.cursor() as cur:
                import psycopg2.extras
                psycopg2.extras.execute_values(
                    cur, sql, values, page_size=1000
                )

    def get_balances_by_date(self, balance_date: date) -> List[DailyBalance]:
        """Get balances for a specific date."""
        with self.session() as session:
            stmt = select(DailyBalance).where(DailyBalance.date == balance_date)
            return session.execute(stmt).scalars().all()
            
    def get_latest_date(self) -> Optional[date]:
        """Get the latest date available in balances."""
        with self.session() as session:
            stmt = select(func.max(DailyBalance.date))
            return session.execute(stmt).scalar()

    def get_wallet_balances(self, wallet_address: str, date: date) -> List[DailyBalance]:
        """Get balances for a specific wallet on a date."""
        with self.session() as session:
            stmt = select(DailyBalance).where(
                and_(DailyBalance.wallet_address == wallet_address, DailyBalance.date == date)
            )
            return session.execute(stmt).scalars().all()
