import os
import sys
import logging
from datetime import datetime, timezone
from decimal import Decimal

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_utils import get_db_connection
from config import COLD_WALLET_ADDRESS, WARM_WALLET_ADDRESS

logger = logging.getLogger(__name__)

from sqlalchemy import text

class LedgerManager:
    def __init__(self):
        self.engine = get_db_connection()
        self.conn = None

    def __enter__(self):
        self.conn = self.engine.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
        if self.engine:
            self.engine.dispose()

    def get_unallocated_balance(self, cold_wallet_address):
        """
        Calculates the unallocated balance for each token from the cold wallet address.
        Returns the net balance (incoming - outgoing) for tokens currently in the cold wallet.
        This excludes funds that have been transferred to the warm wallet.
        """
        query = """
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
        """
        result = self.conn.execute(text(query), {
            "cold_wallet_address": cold_wallet_address,
            "warm_wallet_address": WARM_WALLET_ADDRESS
        })
        balances = {}
        for row in result.fetchall():
            token_symbol = row[0]
            net_balance = row[1]
            balances[token_symbol] = net_balance
            logger.debug(f"Unallocated balance for {token_symbol}: {net_balance}")
        
        logger.info(f"Found {len(balances)} tokens with unallocated balances in cold wallet")
        return balances

    def get_allocated_balances(self):
        """
        Identifies assets currently allocated in pools.
        Returns net allocated balance (deposits - withdrawals) for each pool/token combination.
        Tracks transactions from warm wallet to pools (since funds flow: cold → warm → pools).
        """
        query = """
        SELECT
            p.pool_id,
            at.token_symbol,
            SUM(CASE 
                WHEN LOWER(at.from_address) = LOWER(:warm_wallet_address) AND LOWER(at.to_address) = LOWER(p.pool_address) THEN at.value
                WHEN LOWER(at.from_address) = LOWER(p.pool_address) AND LOWER(at.to_address) = LOWER(:warm_wallet_address) THEN -at.value
                ELSE 0
            END) / (10 ^ MAX(at.token_decimals)) as net_allocated
        FROM account_transactions at
        JOIN pools p ON (LOWER(at.to_address) = LOWER(p.pool_address) OR LOWER(at.from_address) = LOWER(p.pool_address))
        WHERE (LOWER(at.from_address) = LOWER(:warm_wallet_address) OR LOWER(at.to_address) = LOWER(:warm_wallet_address))
          AND p.pool_address IS NOT NULL
        GROUP BY p.pool_id, at.token_symbol
        HAVING SUM(CASE 
            WHEN LOWER(at.from_address) = LOWER(:warm_wallet_address) AND LOWER(at.to_address) = LOWER(p.pool_address) THEN at.value
            WHEN LOWER(at.from_address) = LOWER(p.pool_address) AND LOWER(at.to_address) = LOWER(:warm_wallet_address) THEN -at.value
            ELSE 0
        END) != 0;
        """
        result = self.conn.execute(text(query), {"warm_wallet_address": WARM_WALLET_ADDRESS})
        allocations = []
        for row in result.fetchall():
            pool_id = row[0]
            token_symbol = row[1]
            net_allocated = row[2]
            allocations.append((pool_id, token_symbol, net_allocated))
            logger.debug(f"Allocated balance for pool {pool_id}, token {token_symbol}: {net_allocated}")
        
        logger.info(f"Found {len(allocations)} pool/token allocations")
        return allocations

    def update_filtered_out_pools(self):
        """
        Updates the currently_filtered_out flag for pools that are filtered out during pre-filtering.
        """
        # This function will be called from the pre-filtering script.
        # For now, we'll just have a placeholder.
        pass

    def record_daily_balance(self, date, token_symbol, wallet_address, unallocated_balance, allocated_balance, pool_id):
        """Records the daily balance entry for a specific token using upsert logic."""
        # Check if record exists
        check_query = """
        SELECT id FROM daily_balances
        WHERE date = :date AND token_symbol = :token_symbol AND wallet_address = :wallet_address AND pool_id = :pool_id;
        """
        result = self.conn.execute(text(check_query), {
            "date": date,
            "token_symbol": token_symbol,
            "wallet_address": wallet_address,
            "pool_id": pool_id
        })
        existing_record = result.fetchone()
        
        if existing_record:
            # Update existing record
            update_query = """
            UPDATE daily_balances SET
                unallocated_balance = :unallocated_balance,
                allocated_balance = :allocated_balance
            WHERE id = :id;
            """
            self.conn.execute(text(update_query), {
                "id": existing_record[0],
                "unallocated_balance": unallocated_balance,
                "allocated_balance": allocated_balance
            })
        else:
            # Insert new record
            insert_query = """
            INSERT INTO daily_balances (
                date,
                token_symbol,
                wallet_address,
                unallocated_balance,
                allocated_balance,
                pool_id
            ) VALUES (:date, :token_symbol, :wallet_address, :unallocated_balance, :allocated_balance, :pool_id);
            """
            self.conn.execute(text(insert_query), {
                "date": date,
                "token_symbol": token_symbol,
                "wallet_address": wallet_address,
                "unallocated_balance": unallocated_balance,
                "allocated_balance": allocated_balance,
                "pool_id": pool_id
            })
        
        self.conn.commit()
        logger.info(f"Daily balance record upserted for {token_symbol} on {date}")

    def clear_daily_records(self, date, wallet_address):
        """
        Clears existing daily balance records for the given date and wallet to avoid duplicates.
        """
        delete_query = """
        DELETE FROM daily_balances
        WHERE date = :date AND wallet_address = :wallet_address;
        """
        result = self.conn.execute(text(delete_query), {
            "date": date,
            "wallet_address": wallet_address
        })
        deleted_count = result.rowcount
        if deleted_count > 0:
            logger.info(f"Cleared {deleted_count} existing records for {date}")
        self.conn.commit()

    def log_ledger_summary(self, date, cold_wallet_address, unallocated_balances, allocated_balances):
        """
        Logs a comprehensive summary of the ledger management process.
        """
        logger.info("\n" + "="*60)
        logger.info("📊 DAILY BALANCE MANAGEMENT SUMMARY")
        logger.info("="*60)
        logger.info(f"📅 Date processed: {date}")
        logger.info(f"🔒 Cold wallet address: {cold_wallet_address}")
        
        # Calculate totals
        total_unallocated = sum(float(balance) for balance in unallocated_balances.values())
        total_allocated = sum(float(amount) for _, _, amount in allocated_balances)
        
        logger.info(f"💰 Unallocated balances: {len(unallocated_balances)} tokens, Total: ${total_unallocated:,.2f}")
        for token, balance in unallocated_balances.items():
            logger.info(f"   - {token}: ${float(balance):,.2f}")
        
        logger.info(f"💼 Allocated balances: {len(allocated_balances)} positions, Total: ${total_allocated:,.2f}")
        for pool_id, token_symbol, amount in allocated_balances:
            logger.info(f"   - Pool {pool_id[:8]}... ({token_symbol}): ${float(amount):,.2f}")
        
        logger.info(f"💾 Total assets tracked: ${total_unallocated + total_allocated:,.2f}")
        logger.info(f"💾 Data stored in: daily_balances table")
        logger.info("="*60)

    def manage_ledger(self):
        today = datetime.now(timezone.utc).date()
        cold_wallet_address = COLD_WALLET_ADDRESS

        logger.info(f"Processing ledger for {today}...")
        logger.info(f"Cold wallet address: {cold_wallet_address}")

        # Get current balances
        unallocated_balances = self.get_unallocated_balance(cold_wallet_address)
        allocated_balances = self.get_allocated_balances()

        # Clear existing records for today to avoid duplicates
        self.clear_daily_records(today, cold_wallet_address)

        # Record unallocated balances
        logger.info(f"Recording {len(unallocated_balances)} unallocated token balances...")
        for token, balance in unallocated_balances.items():
            logger.debug(f"Recording unallocated balance: {token} = {balance}")
            self.record_daily_balance(
                date=today,
                token_symbol=token,
                wallet_address=cold_wallet_address,
                unallocated_balance=Decimal(str(balance)),
                allocated_balance=Decimal('0'),
                pool_id=None
            )

        # Record allocated balances
        logger.info(f"Recording {len(allocated_balances)} pool allocations...")
        for pool_id, token_symbol, amount in allocated_balances:
            logger.debug(f"Recording allocated balance: pool {pool_id}, token {token_symbol} = {amount}")
            self.record_daily_balance(
                date=today,
                token_symbol=token_symbol,
                wallet_address=cold_wallet_address,
                unallocated_balance=Decimal('0'),
                allocated_balance=Decimal(str(amount)),
                pool_id=pool_id
            )

        # Log summary
        self.log_ledger_summary(today, cold_wallet_address, unallocated_balances, allocated_balances)

if __name__ == "__main__":
    with LedgerManager() as manager:
        manager.manage_ledger()