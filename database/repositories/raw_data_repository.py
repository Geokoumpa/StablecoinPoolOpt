from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy import select, func, text
from database.models.raw_data import (
    RawDefiLlamaPool, RawDefiLlamaPoolHistory, RawEthGasTrackerHourlyGasData,
    RawEtherscanAccountTransaction,
    RawEthplorerAccountTransaction
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

    def has_raw_pool_data_for_date(self, date_val) -> bool:
        """Check if raw pool data exists for the given date."""
        with self.session() as session:
            stmt = select(func.count()).select_from(RawDefiLlamaPool).where(
                func.date(RawDefiLlamaPool.insertion_timestamp) == date_val
            )
            count = session.execute(stmt).scalar()
            return count > 0


    def insert_raw_pool_history(self, data: List[Dict[str, Any]]) -> None:
        """Insert raw DeFiLlama pool history."""
        self._bulk_insert(RawDefiLlamaPoolHistory, data)
        
    def insert_raw_gas_data(self, data: List[Dict[str, Any]]) -> None:
        """Insert raw gas data."""
        self._bulk_insert(RawEthGasTrackerHourlyGasData, data)

    def has_raw_gas_data_for_date(self, date_val) -> bool:
        """Check if raw gas data exists for the given date."""
        with self.session() as session:
            stmt = select(func.count()).select_from(RawEthGasTrackerHourlyGasData).where(
                func.date(RawEthGasTrackerHourlyGasData.insertion_timestamp) == date_val
            )
            count = session.execute(stmt).scalar()
            return count > 0



    def has_raw_ethplorer_transactions_for_date(self, date_val) -> bool:
        """Check if raw ethplorer transactions exist for the given date."""
        with self.session() as session:
            stmt = select(func.count()).select_from(RawEthplorerAccountTransaction).where(
                func.date(RawEthplorerAccountTransaction.insertion_timestamp) == date_val
            )
            count = session.execute(stmt).scalar()
            return count > 0
        
    def insert_raw_transactions(self, data: List[Dict[str, Any]], source: str = 'etherscan') -> None:
        """Insert raw transaction data."""
        if source == 'etherscan':
            self._bulk_insert(RawEtherscanAccountTransaction, data)
        elif source == 'ethplorer':
            self._bulk_insert(RawEthplorerAccountTransaction, data)
            

        
    def insert_ohlcv_data(self, data: List[Dict[str, Any]]) -> None:
        """
        Insert OHLCV data with conflict handling (duplicates ignored).
        """
        if not data:
            return
            
        sql = """
            INSERT INTO raw_coinmarketcap_ohlcv (data_timestamp, symbol, raw_json_data, insertion_timestamp)
            VALUES %s
            ON CONFLICT (data_timestamp, symbol) DO UPDATE SET
                raw_json_data = EXCLUDED.raw_json_data,
                insertion_timestamp = EXCLUDED.insertion_timestamp
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

    def get_crypto_open_prices(self, symbol: str) -> List[Any]:
        """
        Get daily open prices for a symbol using raw OHLCV data.
        Returns list of (date, open_price) tuples.
        """
        sql = text("""
            WITH daily_data AS (
                SELECT
                    data_timestamp,
                    (raw_json_data->'USD'->>'close')::numeric AS close_price
                FROM
                    raw_coinmarketcap_ohlcv
                WHERE
                    symbol = :symbol
                UNION ALL
                -- Ensure today's date is included
                SELECT CAST(CURRENT_DATE AS TIMESTAMP), NULL
            )
            SELECT
                data_timestamp AS date,
                LAG(close_price, 1) OVER (ORDER BY data_timestamp) AS open_price
            FROM
                daily_data
            ORDER BY
                data_timestamp;
        """)
        
        with self.session() as session:
            result = session.execute(sql, {"symbol": symbol})
            return result.fetchall()

    def get_raw_history_for_active_pools(self) -> List[Any]:
        """
        Get historical raw pool data for all active pools, with date series filling.
        Returns list of (pool_id, date, timestamp, apy, tvl_usd).
        """
        sql = text("""
            WITH date_range AS (
                SELECT generate_series(
                    date(MIN(timestamp)),
                    date(MAX(timestamp)),
                    '1 day'::interval
                )::date AS date
                FROM raw_defillama_pool_history
            ),
            unique_pools AS (
                SELECT DISTINCT h.pool_id
                FROM raw_defillama_pool_history h
                JOIN pools p ON h.pool_id = p.pool_id
                WHERE p.is_active = TRUE
            ),
            daily_data AS (
                SELECT
                    p.pool_id,
                    d.date,
                    h.timestamp,
                    (h.raw_json_data->>'apy')::numeric AS apy,
                    (h.raw_json_data->>'tvlUsd')::numeric AS tvl_usd
                FROM unique_pools p
                CROSS JOIN date_range d
                LEFT JOIN raw_defillama_pool_history h
                    ON p.pool_id = h.pool_id
                    AND date(h.timestamp) = d.date
            )
            SELECT * FROM daily_data
            ORDER BY pool_id, date;
        """)
        with self.session() as session:
            return session.execute(sql).fetchall()

    def get_raw_history_for_pool(self, pool_id: str, start_date: Any) -> List[Any]:
        """
        Get raw history for a specific pool since start_date.
        """
        sql = text("""
            SELECT
                DATE(h.timestamp) as date,
                h.timestamp,
                (h.raw_json_data->>'apy')::numeric AS apy,
                (h.raw_json_data->>'tvlUsd')::numeric AS tvl_usd
            FROM raw_defillama_pool_history h
            WHERE h.pool_id = :pool_id
              AND DATE(h.timestamp) >= :start_date
            ORDER BY DATE(h.timestamp), h.timestamp DESC;
        """)
        with self.session() as session:
            return session.execute(sql, {'pool_id': pool_id, 'start_date': start_date}).fetchall()

    def get_latest_token_metrics(self, tokens: List[str]) -> List[Any]:
        """
        Get latest OHLCV metrics for specified tokens.
        """
        sql = text("""
            SELECT DISTINCT ON (symbol)
                    symbol AS token_symbol,
                    (raw_json_data->>'quote')::jsonb->'USD'->>'price' AS current_price,
                    (raw_json_data->>'quote')::jsonb->'USD'->>'market_cap' AS market_cap,
                    (raw_json_data->>'quote')::jsonb->'USD'->>'volume_24h' AS volume_24h,
                    (raw_json_data->>'quote')::jsonb->'USD'->>'percent_change_1h' AS percent_change_1h,
                    (raw_json_data->>'quote')::jsonb->'USD'->>'percent_change_24h' AS percent_change_24h,
                    (raw_json_data->>'quote')::jsonb->'USD'->>'percent_change_7d' AS percent_change_7d,
                    (raw_json_data->>'quote')::jsonb->'USD'->>'percent_change_30d' AS percent_change_30d,
                    data_timestamp
            FROM raw_coinmarketcap_ohlcv
            WHERE symbol = ANY(:tokens)
            ORDER BY symbol, data_timestamp DESC;
        """)
        with self.session() as session:
            return session.execute(sql, {"tokens": tokens}).fetchall()

    def get_token_min_price(self, token_symbol: str, days: int) -> float:
        """Get minimum price for token in last X days."""
        sql = text("""
            SELECT MIN((raw_json_data->>'quote')::jsonb->'USD'->>'price')
            FROM raw_coinmarketcap_ohlcv
            WHERE symbol = :token_symbol
            AND data_timestamp >= NOW() - make_interval(days => :days)
        """)
        with self.session() as session:
            result = session.execute(sql, {"token_symbol": token_symbol, "days": days}).scalar()
            return float(result) if result else None

    def get_token_price_at_interval(self, token_symbol: str, days: int) -> float:
        """Get price X days ago (closest available record)."""
        sql = text("""
            SELECT (raw_json_data->>'quote')::jsonb->'USD'->>'price'
            FROM raw_coinmarketcap_ohlcv
            WHERE symbol = :token_symbol
            AND data_timestamp >= NOW() - make_interval(days => :days)
            ORDER BY data_timestamp ASC LIMIT 1;
        """)
        with self.session() as session:
            result = session.execute(sql, {"token_symbol": token_symbol, "days": days}).scalar()
            return float(result) if result else None

    def get_new_raw_ethplorer_transactions(self, last_processed_id: int) -> List[Any]:
        """Get new raw ethplorer transactions with ID > last_processed_id."""
        sql = text("""
            SELECT id, raw_json_data
            FROM raw_ethplorer_account_transactions
            WHERE id > :last_processed_id
            ORDER BY id
        """)
        with self.session() as session:
            return session.execute(sql, {"last_processed_id": last_processed_id}).fetchall()

    def clear_raw_pool_history(self) -> None:
        """Clear all data from raw_defillama_pool_history table."""
        with self.session() as session:
            session.execute(text("DELETE FROM raw_defillama_pool_history"))

    def get_latest_prices(self, tokens: List[str]) -> Dict[str, float]:
        """
        Get latest prices for specified tokens from raw_coinmarketcap_ohlcv.
        Handles case insensitivity and multiple quote currencies (USD, USDT, BTC, ETH).
        Returns: Dict[original_token_symbol, price_usd]
        """
        if not tokens:
            return {}
            
        # Create a mapping from lowercase to original case for proper return values
        token_mapping = {token.lower(): token for token in tokens}
        tokens_lower = list(token_mapping.keys())
        
        sql = text("""
        WITH ranked_ohlcv AS (
            SELECT
                LOWER(symbol) as symbol_lower,
                CASE 
                    WHEN raw_json_data->'quote' ? 'USD' THEN (raw_json_data->'quote'->'USD'->>'close')::float
                    WHEN raw_json_data ? 'USD' THEN (raw_json_data->'USD'->>'close')::float
                    WHEN raw_json_data ? 'USDT' THEN (raw_json_data->'USDT'->>'close')::float
                    WHEN raw_json_data ? 'BTC' THEN (raw_json_data->'BTC'->>'close')::float
                    WHEN raw_json_data ? 'ETH' THEN (raw_json_data->'ETH'->>'close')::float
                    ELSE NULL
                END as close_price,
                data_timestamp as ts,
                ROW_NUMBER() OVER(
                    PARTITION BY LOWER(symbol) 
                    ORDER BY data_timestamp DESC
                ) as rn
            FROM raw_coinmarketcap_ohlcv
            WHERE LOWER(symbol) = ANY(:tokens_lower)
        )
        SELECT symbol_lower, close_price
        FROM ranked_ohlcv
        WHERE rn = 1;
        """)
        
        with self.session() as session:
            rows = session.execute(sql, {'tokens_lower': tokens_lower}).fetchall()
            
        prices = {}
        for row in rows:
            symbol_lower = row[0]
            price = row[1]
            if price is not None:
                original_token = token_mapping.get(symbol_lower)
                if original_token:
                    prices[original_token] = float(price)
                    
        return prices






