
from sqlalchemy import Column, String, Integer, DECIMAL, Date, DateTime, Text, UniqueConstraint
from sqlalchemy.sql import func
from database.models.base import Base

class MacroeconomicData(Base):
    __tablename__ = 'macroeconomic_data'

    id = Column(Integer, primary_key=True)
    series_id = Column(String(50), nullable=False)
    series_name = Column(String(255), nullable=False)
    frequency = Column(String(20), nullable=False)
    date = Column(Date, nullable=False)
    value = Column(DECIMAL(20, 8))
    unit = Column(String(50))
    description = Column(Text)
    insertion_timestamp = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint('series_id', 'date', name='macroeconomic_data_series_id_date_key'),
    )
