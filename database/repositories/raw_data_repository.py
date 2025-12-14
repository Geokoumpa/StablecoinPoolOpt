
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy import select
from database.models.raw_data import (
    RawDefiLlamaPool, RawDefiLlamaPoolHistory, RawEthGasTrackerHourlyGasData,
    RawEtherscanAccountTransaction, RawEtherscanAccountBalance,
    RawCoinMarketCapOHLCV, RawEthplorerAccountTransaction, 
    RawEthplorerAccountTransactionDetail
)
from database.repositories.base_repository import BaseRepository

class RawDataRepository(BaseRepository[RawDefiLlamaPool]):
    """
    Repository for raw data storage.
    Handles multiple raw data models.
    """
    def __init__(self):
        super().__init__(model_class=RawDefiLlamaPool)

    def insert_raw_pools(self, data: List[Dict[str, Any]]) -> None:
        """Insert raw DeFiLlama pool data."""
        self._bulk_insert(RawDefiLlamaPool, data)

    def insert_raw_pool_history(self, data: List[Dict[str, Any]]) -> None:
        """Insert raw DeFiLlama pool history."""
        self._bulk_insert(RawDefiLlamaPoolHistory, data)
        
    def insert_raw_gas_data(self, data: List[Dict[str, Any]]) -> None:
        """Insert raw gas data."""
        self._bulk_insert(RawEthGasTrackerHourlyGasData, data)
        
    def insert_raw_transactions(self, data: List[Dict[str, Any]], source: str = 'etherscan') -> None:
        """Insert raw transaction data."""
        if source == 'etherscan':
            self._bulk_insert(RawEtherscanAccountTransaction, data)
        elif source == 'ethplorer':
            self._bulk_insert(RawEthplorerAccountTransaction, data)
            
    def insert_raw_balances(self, data: List[Dict[str, Any]]) -> None:
        """Insert raw account balance data."""
        self._bulk_insert(RawEtherscanAccountBalance, data)
        
    def insert_ohlcv_data(self, data: List[Dict[str, Any]]) -> None:
        """
        Insert OHLCV data with conflict handling (duplicates ignored).
        """
        if not data:
            return
            
        sql = """
            INSERT INTO raw_coinmarketcap_ohlcv (data_timestamp, symbol, raw_json_data, insertion_timestamp)
            VALUES %s
            ON CONFLICT (data_timestamp, symbol) DO NOTHING
        """
        
        now_val = datetime.now()
        
        values = [
            (d['data_timestamp'], d['symbol'], d.get('raw_json_data'), now_val)
            for d in data
        ]
        
        self.execute_bulk_values(sql, values)
        
    def insert_raw_tx_details(self, data: List[Dict[str, Any]]) -> None:
        """Insert raw transaction details (upsert)."""
        if not data:
            return
            
        sql = """
            INSERT INTO raw_ethplorer_account_transaction_details (transaction_hash, raw_json, fetched_at)
            VALUES %s
            ON CONFLICT (transaction_hash) DO UPDATE SET
                raw_json = EXCLUDED.raw_json,
                fetched_at = EXCLUDED.fetched_at
        """
        
        now_val = datetime.now()
        values = [
            (d['transaction_hash'], d.get('raw_json'), now_val)
            for d in data
        ]
        
        self.execute_bulk_values(sql, values)

    def _bulk_insert(self, model_class, data: List[Dict[str, Any]]) -> None:
        """Helper to bulk insert data for a specific model."""
        if not data:
            return
        
        with self.session() as session:
            session.bulk_insert_mappings(model_class, data)
