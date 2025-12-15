
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




            

    
    def get_last_processed_raw_id(self) -> int:
        """Get the maximum raw_transaction_id processed."""
        with self.session() as session:
            stmt = select(func.max(Transaction.raw_transaction_id))
            result = session.execute(stmt).scalar()
            return result if result is not None else 0

    def get_unenriched_transactions(self) -> List[Any]:
        """Get list of (transaction_hash, raw_transaction_id) for transactions pending enrichment."""
        with self.session() as session:
            stmt = select(Transaction.transaction_hash, Transaction.raw_transaction_id).where(Transaction.block_number.is_(None))
            return session.execute(stmt).fetchall()

    def delete_transaction(self, tx_hash: str, operation_index: int) -> None:
        """Delete a specific transaction record."""
        with self.session() as session:
            stmt = text("DELETE FROM account_transactions WHERE transaction_hash = :tx_hash AND operation_index = :op_index")
            session.execute(stmt, {"tx_hash": tx_hash, "op_index": operation_index})

    def update_transaction(self, tx_hash: str, operation_index: int, updates: Dict[str, Any]) -> None:
        """Update a specific transaction record."""
        if not updates:
            return
            
        # Construct update statement dynamically
        set_clauses = [f"{k} = :{k}" for k in updates.keys()]
        sql = text(f"UPDATE account_transactions SET {', '.join(set_clauses)} WHERE transaction_hash = :tx_hash AND operation_index = :op_index")
        
        params = updates.copy()
        params['tx_hash'] = tx_hash
        params['op_index'] = operation_index
        
        with self.session() as session:
            session.execute(sql, params)

    def get_unallocated_balance_rollup(self, cold_wallet_address: str, warm_wallet_address: str) -> List[Any]:
        """
        Calculates the unallocated balance for each token from the cold wallet address.
        Returns: token_symbol, net_balance
        """
        query = text("""
        SELECT
            at.token_symbol,
            SUM(CASE 
                WHEN LOWER(at.to_address) = LOWER(:cold_wallet_address) THEN at.value 
                WHEN LOWER(at.from_address) = LOWER(:cold_wallet_address) THEN -at.value 
                ELSE 0 
            END) / (10 ^ MAX(at.token_decimals)) as net_balance
        FROM account_transactions at
        WHERE (LOWER(at.from_address) = LOWER(:cold_wallet_address) OR LOWER(at.to_address) = LOWER(:cold_wallet_address))
          AND LOWER(at.to_address) NOT IN (SELECT LOWER(pool_address) FROM pools WHERE pool_address IS NOT NULL)
          AND LOWER(at.from_address) NOT IN (SELECT LOWER(pool_address) FROM pools WHERE pool_address IS NOT NULL)
          AND LOWER(at.to_address) != LOWER(:warm_wallet_address)
          AND LOWER(at.from_address) != LOWER(:warm_wallet_address)
        GROUP BY at.token_symbol
        HAVING SUM(CASE 
            WHEN LOWER(at.to_address) = LOWER(:cold_wallet_address) THEN at.value 
            WHEN LOWER(at.from_address) = LOWER(:cold_wallet_address) THEN -at.value 
            ELSE 0 
        END) != 0;
        """)
        with self.session() as session:
            return session.execute(query, {
                "cold_wallet_address": cold_wallet_address,
                "warm_wallet_address": warm_wallet_address
            }).fetchall()

    def get_allocated_balance_rollup(self, warm_wallet_address: str, target_date: Any) -> List[Any]:
        """
        Identifies assets currently allocated in pools.
        Returns: pool_id, token_symbol, final_balance
        """
        query = text("""
        WITH valid_pools AS (
            SELECT DISTINCT ON (LOWER(p.pool_address))
                p.pool_id,
                p.pool_address,
                p.underlying_tokens
            FROM pools p
            LEFT JOIN approved_tokens at ON LOWER(p.pool_address) = LOWER(at.token_address)
            LEFT JOIN pool_daily_metrics pdm ON p.pool_id = pdm.pool_id AND pdm.date = :target_date
            WHERE p.pool_address IS NOT NULL
              AND at.token_symbol IS NULL
            ORDER BY LOWER(p.pool_address), COALESCE(pdm.forecasted_apy, 0) DESC, p.pool_id
        ),
        receipt_token_flows AS (
            SELECT
                vp.pool_id,
                vp.underlying_tokens,
                at.token_symbol as receipt_token,
                at.timestamp,
                CASE
                    WHEN LOWER(at.from_address) = LOWER(vp.pool_address)
                         AND LOWER(at.to_address) = LOWER(:warm_wallet_address)
                    THEN at.value / (10 ^ at.token_decimals)
                    WHEN LOWER(at.to_address) = LOWER(vp.pool_address)
                         AND LOWER(at.from_address) = LOWER(:warm_wallet_address)
                    THEN -at.value / (10 ^ at.token_decimals)
                    ELSE 0
                END as flow_amount
            FROM account_transactions at
            JOIN valid_pools vp ON (
                LOWER(at.to_address) = LOWER(vp.pool_address)
                OR LOWER(at.from_address) = LOWER(vp.pool_address)
            )
            WHERE at.token_symbol IS NOT NULL
              AND at.value > 0
        ),
        minted_token_flows AS (
            SELECT
                vp.pool_id,
                vp.underlying_tokens,
                at_mint.token_symbol as receipt_token,
                at_mint.timestamp,
                CASE
                    WHEN LOWER(at_mint.to_address) = LOWER(:warm_wallet_address)
                    THEN at_mint.value / (10 ^ at_mint.token_decimals)
                    WHEN LOWER(at_mint.from_address) = LOWER(:warm_wallet_address)
                         AND LOWER(at_mint.to_address) = '0x0000000000000000000000000000000000000000'
                    THEN -at_mint.value / (10 ^ at_mint.token_decimals)
                    ELSE 0
                END as flow_amount
            FROM account_transactions at_mint
            JOIN valid_pools vp ON (
                LOWER(at_mint.token_address) = LOWER(vp.pool_address)
            )
            WHERE (
                LOWER(at_mint.from_address) = '0x0000000000000000000000000000000000000000'
                OR LOWER(at_mint.to_address) = '0x0000000000000000000000000000000000000000'
            )
              AND at_mint.token_symbol IS NOT NULL
              AND at_mint.value > 0
        ),
        all_flows AS (
            SELECT pool_id, underlying_tokens, receipt_token, timestamp, flow_amount
            FROM receipt_token_flows
            WHERE flow_amount != 0
            UNION ALL
            SELECT pool_id, underlying_tokens, receipt_token, timestamp, flow_amount
            FROM minted_token_flows
            WHERE flow_amount != 0
        ),
        final_balances AS (
            SELECT
                pool_id,
                underlying_tokens[1] as token_symbol,
                SUM(flow_amount) as final_balance
            FROM all_flows
            GROUP BY pool_id, underlying_tokens
        )
        SELECT
            pool_id,
            COALESCE(token_symbol, 'UNKNOWN') as token_symbol,
            final_balance
        FROM final_balances
        WHERE final_balance > 0
        ORDER BY pool_id, token_symbol;
        """)
        with self.session() as session:
            return session.execute(query, {
                "warm_wallet_address": warm_wallet_address,
                "target_date": target_date
            }).fetchall()

    def get_unmatched_outflows(self, warm_wallet_address: str, cold_wallet_address: str) -> List[Any]:
        """
        Identify transactions from Warm Wallet to addresses NOT in the pools table.
        Returns: to_address, token_symbol, total_sent, tx_count, last_tx_time
        """
        query = text("""
        SELECT
            at.to_address,
            at.token_symbol,
            SUM(at.value) / (10 ^ MAX(at.token_decimals)) as total_sent,
            COUNT(*) as tx_count,
            MAX(at.timestamp) as last_tx_time
        FROM account_transactions at
        WHERE LOWER(at.from_address) = LOWER(:warm_wallet_address)
          AND LOWER(at.to_address) NOT IN (SELECT LOWER(pool_address) FROM pools WHERE pool_address IS NOT NULL)
          AND LOWER(at.to_address) != LOWER(:cold_wallet_address)
        GROUP BY at.to_address, at.token_symbol
        HAVING SUM(at.value) > 0
        ORDER BY total_sent DESC
        LIMIT 10;
        """)
        with self.session() as session:
            return session.execute(query, {
                "warm_wallet_address": warm_wallet_address,
                "cold_wallet_address": cold_wallet_address
            }).fetchall()


