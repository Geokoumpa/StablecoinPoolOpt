import os
import sys
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Dict, Any

from database.repositories.transaction_repository import TransactionRepository
from database.repositories.daily_balance_repository import DailyBalanceRepository
from config import COLD_WALLET_ADDRESS, MAIN_ASSET_HOLDING_ADDRESS

logger = logging.getLogger(__name__)

class LedgerManager:
    def __init__(self):
        self.tx_repo = TransactionRepository()
        self.balance_repo = DailyBalanceRepository()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def manage_ledger(self):
        today = datetime.now(timezone.utc).date()
        cold_wallet_address = COLD_WALLET_ADDRESS
        warm_wallet_address = MAIN_ASSET_HOLDING_ADDRESS

        logger.info(f"Processing ledger for {today}...")
        logger.info(f"Cold wallet address: {cold_wallet_address}")
        logger.info(f"Warm wallet address: {warm_wallet_address}")

        # 1. Get current balances
        # Unallocated (Cold Wallet)
        unallocated_rows = self.tx_repo.get_unallocated_balance_rollup(cold_wallet_address, warm_wallet_address)
        unallocated_balances = {row[0]: row[1] for row in unallocated_rows}
        
        # Allocated (Pools) - Passing today as date for pool metrics lookup
        allocated_rows = self.tx_repo.get_allocated_balance_rollup(warm_wallet_address, today)
        # allocated_rows is list of (pool_id, token_symbol, final_balance)
        
        # 2. DIAGNOSTIC: Check for missing pools
        self._analyze_unmatched_transactions(warm_wallet_address, cold_wallet_address)

        # 3. Prepare data for bulk upsert
        balances_data = []

        # Prepare unallocated balances
        logger.info(f"Preparing {len(unallocated_balances)} unallocated token balances...")
        for token, balance in unallocated_balances.items():
            balances_data.append({
                'date': today,
                'token_symbol': token,
                'wallet_address': cold_wallet_address,
                'unallocated_balance': Decimal(str(balance)),
                'allocated_balance': Decimal('0'),
                'pool_id': None
            })

        # Prepare allocated balances
        logger.info(f"Preparing {len(allocated_rows)} pool allocations...")
        for row in allocated_rows:
            pool_id = row[0]
            token_symbol = row[1]
            amount = row[2]
            balances_data.append({
                'date': today,
                'token_symbol': token_symbol,
                'wallet_address': None,  # Pool allocations are not held in cold wallet directly in this schema
                'unallocated_balance': Decimal('0'),
                'allocated_balance': Decimal(str(amount)),
                'pool_id': pool_id
            })

        # 4. Bulk Upsert (Clear existing for today and insert new)
        logger.info(f"Bulk upserting {len(balances_data)} balance records...")
        self.balance_repo.bulk_upsert_balances(balances_data, ensure_unique_by_date=True)

        # 5. Log summary
        self._log_ledger_summary(today, cold_wallet_address, unallocated_balances, allocated_rows)

    def _analyze_unmatched_transactions(self, warm_wallet_address, cold_wallet_address):
        """
        Diagnostic method to identify transactions from Warm Wallet to addresses NOT in the pools table.
        """
        logger.info("🔍 Analyzing unmatched transactions from Warm Wallet...")
        unmatched = self.tx_repo.get_unmatched_outflows(warm_wallet_address, cold_wallet_address)
        
        if unmatched:
            logger.warning(f"⚠️ Found {len(unmatched)} addresses receiving funds that are NOT in the 'pools' table:")
            for row in unmatched:
                # to_address, token_symbol, total_sent, tx_count, last_tx_time
                logger.warning(f"   - To: {row[0]} | Token: {row[1]} | Total: {row[2]:,.2f} | Txs: {row[3]} | Last: {row[4]}")
        else:
            logger.info("✅ No significant unmatched outflows found.")

    def _log_ledger_summary(self, date, cold_wallet_address, unallocated_balances, allocated_rows):
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
        total_allocated = sum(float(row[2]) for row in allocated_rows)
        
        logger.info(f"💰 Unallocated balances: {len(unallocated_balances)} tokens, Total: ${total_unallocated:,.2f}")
        for token, balance in unallocated_balances.items():
            logger.info(f"   - {token}: ${float(balance):,.2f}")
        
        logger.info(f"💼 Allocated balances: {len(allocated_rows)} positions, Total: ${total_allocated:,.2f}")
        for row in allocated_rows:
            pool_id = row[0]
            token_symbol = row[1]
            amount = row[2]
            logger.info(f"   - Pool {str(pool_id)[:8]}... ({token_symbol}): ${float(amount):,.2f}")
        
        logger.info(f"💾 Total assets tracked: ${total_unallocated + total_allocated:,.2f}")
        logger.info(f"💾 Data stored in: daily_balances table")
        logger.info("="*60)

def manage_ledger():
    """Module-level function for pipeline execution."""
    with LedgerManager() as manager:
        manager.manage_ledger()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    manage_ledger()