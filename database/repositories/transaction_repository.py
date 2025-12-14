
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy import select, and_, func, text
from database.models.transaction import Transaction
from database.repositories.base_repository import BaseRepository

class TransactionRepository(BaseRepository[Transaction]):
    """
    Repository for Transaction entity operations.
    """
    def __init__(self):
        super().__init__(model_class=Transaction)

    def bulk_insert_transactions(self, transactions_data: List[Dict[str, Any]]) -> None:
        """
        Bulk insert transactions. Ignore duplicates.
        """
        if not transactions_data:
            return

        sql = """
            INSERT INTO account_transactions (
                transaction_hash, operation_index, timestamp, from_address, to_address,
                value, token_symbol, token_decimals, raw_transaction_id, insertion_timestamp,
                block_number, confirmations, success, transaction_index, nonce, raw_value,
                input_data, gas_limit, gas_price, gas_used, method_id, function_name,
                creates, operation_type, operation_priority, token_address, token_name,
                token_price_rate, token_price_currency
            ) VALUES %s
            ON CONFLICT (transaction_hash, operation_index) DO NOTHING
        """
        
        values = [
            (
                t['transaction_hash'], t.get('operation_index', 0), t['timestamp'], 
                t.get('from_address'), t.get('to_address'), t.get('value'), t.get('token_symbol'), 
                t.get('token_decimals'), t.get('raw_transaction_id'), t.get('insertion_timestamp'), 
                t.get('block_number'), t.get('confirmations'), t.get('success'), t.get('transaction_index'), 
                t.get('nonce'), t.get('raw_value'), t.get('input_data'), t.get('gas_limit'), 
                t.get('gas_price'), t.get('gas_used'), t.get('method_id'), t.get('function_name'), 
                t.get('creates'), t.get('operation_type'), t.get('operation_priority'), 
                t.get('token_address'), t.get('token_name'), t.get('token_price_rate'), 
                t.get('token_price_currency')
            )
            for t in transactions_data
        ]
        
        self.execute_bulk_values(sql, values)

    def get_transactions_in_range(self, start_time: datetime, end_time: datetime) -> List[Transaction]:
        """Get transactions within a time range."""
        with self.session() as session:
            stmt = select(Transaction).where(
                and_(Transaction.timestamp >= start_time, Transaction.timestamp <= end_time)
            ).order_by(Transaction.timestamp)
            return session.execute(stmt).scalars().all()

    def get_latest_transaction_timestamp(self) -> Optional[datetime]:
        """Get the timestamp of the latest transaction."""
        with self.session() as session:
            stmt = select(func.max(Transaction.timestamp))
            return session.execute(stmt).scalar()
            
    def get_all_ordered(self) -> List[Transaction]:
        """Get all transactions ordered by timestamp."""
        with self.session() as session:
            stmt = select(Transaction).order_by(Transaction.timestamp, Transaction.operation_index)
            return session.execute(stmt).scalars().all()
    
    # Placeholder for specific unmatched logic if needed later
    def get_unmatched_transactions(self) -> List[Transaction]:
        """
        Get all transactions ordered by timestamp.
        Specific logic for 'unmatched' might be handled by the caller filtering against balances.
        """
        return self.get_all_ordered()
