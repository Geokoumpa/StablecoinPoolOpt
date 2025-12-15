
from sqlalchemy import Column, Integer, DECIMAL, Date, DateTime
from database.models.base import Base




class GasFeesDaily(Base):
    __tablename__ = 'gas_fees_daily'

    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False, unique=True)
    actual_avg_gas_gwei = Column(DECIMAL(10, 2))
    forecasted_avg_gas_gwei = Column(DECIMAL(10, 2))
    actual_max_gas_gwei = Column(DECIMAL(10, 2))
    forecasted_max_gas_gwei = Column(DECIMAL(10, 2))
    eth_open = Column(DECIMAL(20, 8))
    btc_open = Column(DECIMAL(20, 8))
