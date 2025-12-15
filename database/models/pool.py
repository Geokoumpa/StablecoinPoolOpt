
from sqlalchemy import Column, String, DECIMAL, DateTime, Boolean, Text, ARRAY
from sqlalchemy.orm import relationship
from database.models.base import Base

class Pool(Base):
    __tablename__ = 'pools'

    pool_id = Column(String(255), primary_key=True)
    name = Column(String(255), nullable=False)
    chain = Column(String(255), nullable=False)
    protocol = Column(String(255), nullable=False)
    symbol = Column(String(255), nullable=False)
    tvl = Column(DECIMAL(20, 2))
    apy = Column(DECIMAL(20, 4))
    last_updated = Column(DateTime(timezone=True))
    pool_address = Column(String(255))
    underlying_tokens = Column(ARRAY(Text))
    underlying_token_addresses = Column(ARRAY(Text))
    pool_meta = Column("poolmeta", Text)
    is_active = Column(Boolean, default=True)
    currently_filtered_out = Column(Boolean, default=False)

    # Relationships
    metrics = relationship("PoolMetrics", back_populates="pool")
    allocations = relationship("AssetAllocation", back_populates="pool")
    # We'll add other relationships as we define other models
