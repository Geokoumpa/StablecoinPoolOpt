
from sqlalchemy import Column, String, Integer, DateTime, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from database.models.base import Base

class RawDefiLlamaPool(Base):
    __tablename__ = 'raw_defillama_pools'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    raw_json_data = Column(JSONB)
    insertion_timestamp = Column(DateTime(timezone=True), server_default=func.now())


class RawDefiLlamaPoolHistory(Base):
    __tablename__ = 'raw_defillama_pool_history'

    id = Column(Integer, primary_key=True)
    pool_id = Column(String(255), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    raw_json_data = Column(JSONB)
    insertion_timestamp = Column(DateTime(timezone=True), server_default=func.now())


class RawEthGasTrackerHourlyGasData(Base):
    __tablename__ = 'raw_ethgastracker_hourly_gas_data'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    raw_json_data = Column(JSONB)
    insertion_timestamp = Column(DateTime(timezone=True), server_default=func.now())


class RawEtherscanAccountTransaction(Base):
    __tablename__ = 'raw_etherscan_account_transactions'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    raw_json_data = Column(JSONB)
    insertion_timestamp = Column(DateTime(timezone=True), server_default=func.now())










class RawEthplorerAccountTransaction(Base):
    __tablename__ = 'raw_ethplorer_account_transactions'

    id = Column(Integer, primary_key=True)
    raw_json_data = Column(JSONB, nullable=False)
    insertion_timestamp = Column(DateTime(timezone=True), server_default=func.now())



