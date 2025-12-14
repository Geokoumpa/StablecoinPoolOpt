
from sqlalchemy import Column, String, Integer, DECIMAL, DateTime, Boolean, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from database.models.base import Base

class AllocationParameters(Base):
    __tablename__ = 'allocation_parameters'

    run_id = Column(UUID(as_uuid=True), primary_key=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    tvl_limit_percentage = Column(DECIMAL(5, 4))
    max_alloc_percentage = Column(DECIMAL(5, 4))
    conversion_rate = Column(DECIMAL(20, 10))
    min_pools = Column(Integer)
    profit_optimization = Column(Boolean)
    
    approved_tokens_snapshot = Column(JSONB)
    blacklisted_tokens_snapshot = Column(JSONB)
    approved_protocols_snapshot = Column(JSONB)
    icebox_tokens_snapshot = Column(JSONB)
    
    token_marketcap_limit = Column(DECIMAL(20, 2))
    pool_tvl_limit = Column(DECIMAL(20, 2))
    pool_apy_limit = Column(DECIMAL(20, 4))
    pool_pair_tvl_ratio_min = Column(DECIMAL(5, 4))
    pool_pair_tvl_ratio_max = Column(DECIMAL(5, 4))
    
    group1_max_pct = Column(DECIMAL(5, 4))
    group2_max_pct = Column(DECIMAL(5, 4))
    group3_max_pct = Column(DECIMAL(5, 4))
    
    position_max_pct_total_assets = Column(DECIMAL(5, 4))
    position_max_pct_pool_tvl = Column(DECIMAL(5, 4))
    
    group1_apy_delta_max = Column(DECIMAL(20, 4))
    group1_7d_stddev_max = Column(DECIMAL(20, 4))
    group1_30d_stddev_max = Column(DECIMAL(20, 4))
    
    group2_apy_delta_max = Column(DECIMAL(20, 4))
    group2_7d_stddev_max = Column(DECIMAL(20, 4))
    group2_30d_stddev_max = Column(DECIMAL(20, 4))
    
    group3_apy_delta_min = Column(DECIMAL(20, 4))
    group3_7d_stddev_min = Column(DECIMAL(20, 4))
    group3_30d_stddev_min = Column(DECIMAL(20, 4))
    
    other_dynamic_limits = Column(JSONB)
    
    # Icebox parameters V6
    icebox_ohlc_l_threshold_pct = Column(DECIMAL(5, 4))
    icebox_ohlc_l_days_threshold = Column(Integer)
    icebox_ohlc_c_threshold_pct = Column(DECIMAL(5, 4))
    icebox_ohlc_c_days_threshold = Column(Integer)
    icebox_recovery_l_days_threshold = Column(Integer)
    icebox_recovery_c_days_threshold = Column(Integer)
    
    # V25 enhancements
    projected_apy = Column(DECIMAL(20, 4))
    transaction_costs = Column(DECIMAL(20, 2))
    transaction_sequence = Column(JSONB)


class DefaultAllocationParameters(Base):
    __tablename__ = 'default_allocation_parameters'

    id = Column(Integer, primary_key=True)
    parameter_name = Column(String(255), unique=True, nullable=False)
    parameter_value = Column(JSONB, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())
