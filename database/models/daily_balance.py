
from sqlalchemy import Column, String, Integer, DECIMAL, Date
from database.models.base import Base

class DailyBalance(Base):
    __tablename__ = 'daily_balances'

    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    token_symbol = Column(String(255), nullable=False)
    wallet_address = Column(String(255))
    unallocated_balance = Column(DECIMAL(20, 10))
    allocated_balance = Column(DECIMAL(20, 10))
    pool_id = Column(String(255))
