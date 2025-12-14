
from typing import List, Optional, Dict, Any, Union
from uuid import UUID
from sqlalchemy import select, func
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
