
from sqlalchemy import Column, String, Integer, DECIMAL, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database.models.base import Base

class AssetAllocation(Base):
    __tablename__ = 'asset_allocations'

    id = Column(Integer, primary_key=True)
    run_id = Column(UUID(as_uuid=True), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    step_number = Column(Integer, nullable=False)
    operation = Column(String(50), nullable=False)
    from_asset = Column(String(50))
    to_asset = Column(String(50))
    amount = Column(DECIMAL(20, 8))
    pool_id = Column(String(255), ForeignKey('pools.pool_id'))

    # Relationships
    pool = relationship("Pool", back_populates="allocations")

    __table_args__ = (
        UniqueConstraint('run_id', 'step_number', name='asset_allocations_run_id_step_number_key'),
    )
