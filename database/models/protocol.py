
from sqlalchemy import Column, String, Integer, DateTime
from sqlalchemy.sql import func
from database.models.base import Base

class ApprovedProtocol(Base):
    __tablename__ = 'approved_protocols'

    id = Column(Integer, primary_key=True)
    protocol_name = Column(String(255), unique=True, nullable=False)
    added_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    removed_timestamp = Column(DateTime(timezone=True))
