
from typing import List, Optional, Dict, Any
from uuid import UUID
from sqlalchemy import select, func, desc, text
from database.models.asset_allocation import AssetAllocation
from database.repositories.base_repository import BaseRepository

class AllocationRepository(BaseRepository[AssetAllocation]):
    """
    Repository for AssetAllocation entity operations.
    """
    def __init__(self):
        super().__init__(model_class=AssetAllocation)

    def bulk_insert_allocations(self, allocations_data: List[Dict[str, Any]]) -> None:
        """
        Bulk insert allocations. On conflict do nothing.
        """
        if not allocations_data:
            return

        sql = """
            INSERT INTO asset_allocations (
                run_id, timestamp, step_number, operation, 
                from_asset, to_asset, amount, pool_id
            ) VALUES %s
            ON CONFLICT (run_id, step_number) DO NOTHING
        """
        
        values = [
            (
                a['run_id'], a.get('timestamp'), a['step_number'], a['operation'],
                a.get('from_asset'), a.get('to_asset'), a.get('amount'), a.get('pool_id')
            )
            for a in allocations_data
        ]
        
        self.execute_bulk_values(sql, values)

    def get_allocations_by_run_id(self, run_id: UUID) -> List[AssetAllocation]:
        """Get all allocations for a specific run ID."""
        with self.session() as session:
            stmt = select(AssetAllocation).where(AssetAllocation.run_id == run_id).order_by(AssetAllocation.step_number)
            results = session.execute(stmt).scalars().all()
            session.expunge_all()
            return results



    def delete_allocations_for_date(self, target_date: Any) -> int:
        """
        Delete all allocations for a specific date.
        Returns the number of deleted rows.
        """
        sql = text("""
            DELETE FROM asset_allocations 
            WHERE DATE(timestamp) = :date
        """)
        
        with self.session() as session:
            result = session.execute(sql, {'date': target_date})
            return result.rowcount

