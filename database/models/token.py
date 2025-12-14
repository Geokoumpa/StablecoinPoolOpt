
from sqlalchemy import Column, String, Integer, DateTime, Text, UniqueConstraint
from sqlalchemy.sql import func
from database.models.base import Base

class ApprovedToken(Base):
    __tablename__ = 'approved_tokens'

    id = Column(Integer, primary_key=True)
    token_symbol = Column(String(255), unique=True, nullable=False)
    added_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    removed_timestamp = Column(DateTime(timezone=True))
    token_address = Column(String(42), unique=True)


class BlacklistedToken(Base):
    __tablename__ = 'blacklisted_tokens'

    id = Column(Integer, primary_key=True)
    token_symbol = Column(String(255), unique=True, nullable=False)
    added_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    removed_timestamp = Column(DateTime(timezone=True))


class IceboxToken(Base):
    __tablename__ = 'icebox_tokens'

    id = Column(Integer, primary_key=True)
    token_symbol = Column(String(255), unique=True, nullable=False)
    added_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    removed_timestamp = Column(DateTime(timezone=True))
    reason = Column(Text)
