
from typing import List, Optional, Dict, Any, Union
from uuid import UUID
from sqlalchemy import select, func, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from database.models.allocation_parameters import AllocationParameters, DefaultAllocationParameters
from database.repositories.base_repository import BaseRepository

class ParameterRepository(BaseRepository[AllocationParameters]):
    """
    Repository for AllocationParameters and DefaultAllocationParameters operations.
    """
    def __init__(self):
        super().__init__(model_class=AllocationParameters)

    def save_parameters(self, params: AllocationParameters) -> AllocationParameters:
        """Save a new set of allocation parameters."""
        return self.create(params)
    
    def get_parameters(self, run_id: UUID) -> Optional[AllocationParameters]:
        """Get parameters for a specific run."""
        return self.get_by_id(run_id)

    def get_latest_parameters(self) -> Optional[AllocationParameters]:
        """Get the latest allocation parameters."""
        with self.session() as session:
            stmt = select(AllocationParameters).order_by(AllocationParameters.timestamp.desc()).limit(1)
            result = session.execute(stmt).scalar()
            if result:
                session.expunge(result)
            return result


    def get_all_default_parameters(self) -> Dict[str, Any]:
        """Get all default parameters as a dictionary of name -> value."""
        with self.session() as session:
            stmt = select(DefaultAllocationParameters)
            results = session.execute(stmt).scalars().all()
            return {p.parameter_name: p.parameter_value for p in results}
            
    def get_default_parameter(self, name: str) -> Optional[Any]:
        """Get a specific default parameter value."""
        with self.session() as session:
            stmt = select(DefaultAllocationParameters).where(DefaultAllocationParameters.parameter_name == name)
            result = session.execute(stmt).scalar()
            return result.parameter_value if result else None

    def set_default_parameter(self, name: str, value: Any, description: str = None) -> None:
        """Set or update a default parameter value."""
        # Using raw connection for upsert if desired, but here we can use sqlalchemy's dialect support
        
        # Note: We need to use valid JSON for JSONB column if value is complex, 
        # but SQLAlchemy usually handles python dicts -> JSONB automatic conversion.
        # However if value is primitive (int, str), JSONB might require it to be wrapped or just standard behavior.
        
        stmt = pg_insert(DefaultAllocationParameters).values(
             parameter_name=name,
             parameter_value=value,
             description=description
        ).on_conflict_do_update(
             index_elements=['parameter_name'],
             set_=dict(parameter_value=value, description=description, updated_at=func.now())
        )
        
        with self.session() as session:
             session.execute(stmt)
             
    def bulk_create_defaults(self, defaults: Dict[str, Any]) -> None:
        """Bulk create or update default parameters."""
        if not defaults:
            return
            
        values = [
            {'parameter_name': k, 'parameter_value': v} for k, v in defaults.items()
        ]
        
        stmt = pg_insert(DefaultAllocationParameters).values(values)
        stmt = stmt.on_conflict_do_update(
             index_elements=['parameter_name'],
             set_=dict(parameter_value=stmt.excluded.parameter_value, updated_at=func.now())
        )
        
        with self.session() as session:
            session.execute(stmt)

    def update_run_results(self, run_id: UUID, projected_apy: float, transaction_costs: float, transaction_sequence: str) -> None:
        """
        Update allocation parameters with results from an optimization run.
        """
        sql = text("""
            UPDATE allocation_parameters
            SET 
                projected_apy = :projected_apy,
                transaction_costs = :transaction_costs,
                transaction_sequence = :transaction_sequence
            WHERE run_id = :run_id
        """)
        
        with self.session() as session:
            session.execute(sql, {
                'projected_apy': projected_apy,
                'transaction_costs': transaction_costs,
                'transaction_sequence': transaction_sequence,
                'run_id': run_id
            })

    def update_snapshots(self, approved_tokens: List[Dict], blacklisted_tokens: List[Dict],
                         approved_protocols: List[Dict], icebox_tokens: List[Dict]) -> None:
        """
        Update snapshots in the latest allocation_parameters entry.
        """
        sql = text("""
            UPDATE allocation_parameters
            SET
                approved_tokens_snapshot = :approved_tokens,
                blacklisted_tokens_snapshot = :blacklisted_tokens,
                approved_protocols_snapshot = :approved_protocols,
                icebox_tokens_snapshot = :icebox_tokens,
                timestamp = :timestamp
            WHERE run_id = (
                SELECT run_id FROM allocation_parameters
                ORDER BY timestamp DESC LIMIT 1
            )
        """)
        
        # We need to import datetime here or passed in, but using func.now() or python datetime
        from datetime import datetime
        
        # Serialize check: JSON columns in sqlalchemy usually take python dicts/lists and handle serialization if using pg8000/psycopg2 with proper types or explicit json.dumps (if using text() with :param).
        # Since we use text(), we should pass proper JSON types. 
        # If the columns are JSONB, sending python list of dicts should work with most drivers, but explicit json.dumps is safer for raw text queries.
        
        import json
        
        params = {
            "approved_tokens": json.dumps(approved_tokens),
            "blacklisted_tokens": json.dumps(blacklisted_tokens),
            "approved_protocols": json.dumps(approved_protocols),
            "icebox_tokens": json.dumps(icebox_tokens),
            "timestamp": datetime.now()
        }
        
        with self.session() as session:
            session.execute(sql, params)


