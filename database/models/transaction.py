
from sqlalchemy import Column, String, Integer, NUMERIC, DateTime, Boolean, Text, ForeignKey, BigInteger, PrimaryKeyConstraint
from sqlalchemy.orm import relationship
from database.models.base import Base

class Transaction(Base):
    __tablename__ = 'account_transactions'

    # id is still there but not PK anymore
    id = Column(Integer, autoincrement=True) 
    
    transaction_hash = Column(String(255), nullable=False)
    operation_index = Column(Integer, nullable=False, default=0)
    
    timestamp = Column(DateTime(timezone=True), nullable=False)
    from_address = Column(String(255))
    to_address = Column(String(255))
    value = Column(NUMERIC)
    token_symbol = Column(String(50))
    token_decimals = Column(Integer)
    
    raw_transaction_id = Column(Integer, ForeignKey('raw_ethplorer_account_transactions.id'))
    
    insertion_timestamp = Column(DateTime(timezone=True))
    
    block_number = Column(Integer)
    confirmations = Column(Integer)
    success = Column(Boolean)
    transaction_index = Column(Integer)
    nonce = Column(BigInteger)
    raw_value = Column(NUMERIC)
    input_data = Column(Text)
    gas_limit = Column(NUMERIC)
    gas_price = Column(NUMERIC)
    gas_used = Column(NUMERIC)
    method_id = Column(String(20))
    function_name = Column(Text)
    creates = Column(String(255))
    
    operation_type = Column(String(255))
    operation_priority = Column(Integer)
    token_address = Column(String(255))
    token_name = Column(String(255))
    token_price_rate = Column(NUMERIC)
    token_price_currency = Column(String(10))

    __table_args__ = (
        PrimaryKeyConstraint('transaction_hash', 'operation_index'),
    )

    # Relationships need the raw table model to be defined or we can use string
    # raw_transaction = relationship("RawEthplorerAccountTransaction")
