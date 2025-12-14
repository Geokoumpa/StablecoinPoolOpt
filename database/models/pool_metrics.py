
from sqlalchemy import Column, String, Integer, DECIMAL, Date, Boolean, Text, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from database.models.base import Base

class PoolMetrics(Base):
    __tablename__ = 'pool_daily_metrics'

    id = Column(Integer, primary_key=True)
    pool_id = Column(String(255), ForeignKey('pools.pool_id'), nullable=False)
    date = Column(Date, nullable=False)
    actual_apy = Column(DECIMAL(20, 4))
    forecasted_apy = Column(DECIMAL(20, 4))
    actual_tvl = Column(DECIMAL(20, 2))
    forecasted_tvl = Column(DECIMAL(20, 2))
    eth_open = Column(DECIMAL(20, 8))
    btc_open = Column(DECIMAL(20, 8))
    gas_price_gwei = Column(DECIMAL(20, 4))
    is_filtered_out = Column(Boolean, default=False)
    filter_reason = Column(Text)
    
    # Added in V4
    rolling_apy_7d = Column(DECIMAL(20, 4))
    rolling_apy_30d = Column(DECIMAL(20, 4))
    apy_delta_today_yesterday = Column(DECIMAL(20, 4))
    stddev_apy_7d = Column(DECIMAL(20, 4))
    stddev_apy_30d = Column(DECIMAL(20, 4))
    stddev_apy_7d_delta = Column(DECIMAL(20, 4))
    stddev_apy_30d_delta = Column(DECIMAL(20, 4))
    
    # Added in V5
    pool_group = Column(Integer)

    # Relationships
    pool = relationship("Pool", back_populates="metrics")
    
    __table_args__ = (
        UniqueConstraint('pool_id', 'date', name='pool_daily_metrics_pool_id_date_key'),
    )
