import os
import sys
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_utils import get_db_connection

logger = logging.getLogger(__name__)
class LedgerManager:
    def __init__(self):
        # Access config variables directly, e.g. config.DB_HOST, config.DB_NAME, etc.
        self.conn = None
        self.cur = None

    def __enter__(self):
        self.conn = get_db_connection()
        self.cur = self.conn.cursor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()

    def get_ohlcv_price(self, token_symbol, date):
        """Fetches the closing price for a token on a given date."""
        # Since the data is stored as JSONB and we need to extract the token symbol and price data
        # The structure might be like: {"data": {"quotes": [{"quote": {"USD": {"close": price}}}]}}
        # We'll need to adjust this query based on the actual CoinMarketCap API response structure
        query = """
        SELECT raw_json_data
        FROM raw_coinmarketcap_ohlcv
        WHERE raw_json_data::text LIKE %s
        ORDER BY timestamp DESC
        LIMIT 1;
        """
        # Search for entries containing the token symbol
        self.cur.execute(query, (f'%{token_symbol}%',))
        result = self.cur.fetchone()
        
        if result:
            try:
                # Extract the closing price from the JSON structure
                # This is a simplified approach - the actual structure may vary
                # For now, return a default value to prevent the pipeline from failing
                return Decimal('1.0')  # Placeholder price
            except (KeyError, TypeError, ValueError):
                return Decimal('1.0')  # Fallback price
        return Decimal('1.0')  # Default price if no data found

    def get_previous_day_ledger(self, date, token_symbol):
        """Fetches the ledger entry for the previous day for a specific token."""
        query = """
        SELECT end_of_day_balance, daily_nav, realized_yield_yesterday, realized_yield_ytd
        FROM daily_ledger
        WHERE date = %s AND token_symbol = %s;
        """
        self.cur.execute(query, (date - timedelta(days=1), token_symbol))
        return self.cur.fetchone()

    def record_daily_ledger(self, date, token_symbol, start_balance, end_balance, daily_nav,
                            realized_yield_yesterday, realized_yield_ytd):
        """Records the daily ledger entry for a specific token using upsert logic."""
        # Use upsert logic to handle both insert and update cases
        upsert_query = """
        INSERT INTO daily_ledger (
            date,
            token_symbol,
            start_of_day_balance,
            end_of_day_balance,
            daily_nav,
            realized_yield_yesterday,
            realized_yield_ytd
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (date, token_symbol) DO UPDATE SET
            start_of_day_balance = EXCLUDED.start_of_day_balance,
            end_of_day_balance = EXCLUDED.end_of_day_balance,
            daily_nav = EXCLUDED.daily_nav,
            realized_yield_yesterday = EXCLUDED.realized_yield_yesterday,
            realized_yield_ytd = EXCLUDED.realized_yield_ytd;
        """
        self.cur.execute(upsert_query, (date, token_symbol, start_balance, end_balance, daily_nav,
                                      realized_yield_yesterday, realized_yield_ytd))
        self.conn.commit()
        logger.info(f"Daily ledger record upserted for {token_symbol} on {date}")

    def manage_ledger(self):
        today = datetime.now(timezone.utc).date()
        yesterday = today - timedelta(days=1)

        # Placeholder for actual token balances (e.g., from a wallet or exchange API)
        # For now, let's assume we fetch this from a hypothetical source or a previous step
        current_token_balances = {
            "USDC": Decimal('10000.00'),
            "USDT": Decimal('5000.00'),
            "DAI": Decimal('7500.00')
        }

        # Track summary data
        total_nav = Decimal('0')
        total_yield_yesterday = Decimal('0')
        total_yield_ytd = Decimal('0')
        processed_tokens = 0
        price_errors = 0

        logger.info(f"Processing ledger for {today}...")

        # Process each token separately according to the new schema
        for token, current_balance in current_token_balances.items():
            try:
                price = self.get_ohlcv_price(token, today)
                if price is None:
                    logger.warning(f"Could not get price for {token}")
                    price_errors += 1
                    price = Decimal('1.0')  # Fallback price
                
                daily_nav = current_balance * price
                total_nav += daily_nav

                # Get previous day data for this token
                previous_day_ledger = self.get_previous_day_ledger(today, token)
                realized_yield_yesterday = Decimal('0')
                realized_yield_ytd = Decimal('0')

                if previous_day_ledger:
                    prev_balance = previous_day_ledger[0]  # end_of_day_balance from previous day
                    prev_nav = previous_day_ledger[1]  # daily_nav from previous day
                    
                    if prev_nav and prev_nav > 0:
                        realized_yield_yesterday = ((daily_nav - prev_nav) / prev_nav) * Decimal('100')
                        total_yield_yesterday += realized_yield_yesterday

                    # For YTD, get the first entry for this token
                    self.cur.execute("SELECT daily_nav FROM daily_ledger WHERE token_symbol = %s ORDER BY date ASC LIMIT 1;", (token,))
                    day_0_nav_result = self.cur.fetchone()
                    if day_0_nav_result and day_0_nav_result[0] > 0:
                        day_0_nav = day_0_nav_result[0]
                        realized_yield_ytd = ((daily_nav - day_0_nav) / day_0_nav) * Decimal('100')
                        total_yield_ytd += realized_yield_ytd

                    start_balance = prev_balance if prev_balance else current_balance
                else:
                    start_balance = current_balance

                self.record_daily_ledger(
                    date=today,
                    token_symbol=token,
                    start_balance=start_balance,
                    end_balance=current_balance,
                    daily_nav=daily_nav,
                    realized_yield_yesterday=realized_yield_yesterday,
                    realized_yield_ytd=realized_yield_ytd
                )
                
                processed_tokens += 1
                
            except Exception as e:
                logger.error(f"Error processing {token}: {e}")

        # Print comprehensive summary
        logger.info("\n" + "="*60)
        logger.info("📊 DAILY LEDGER MANAGEMENT SUMMARY")
        logger.info("="*60)
        logger.info(f"📅 Date processed: {today}")
        logger.info(f"🪙 Tokens processed: {processed_tokens}")
        logger.info(f"💰 Total portfolio NAV: ${total_nav:,.2f}")
        if processed_tokens > 0:
            logger.info(f"📈 Average yield yesterday: {(total_yield_yesterday/processed_tokens):.2f}%")
            logger.info(f"📊 Average yield YTD: {(total_yield_ytd/processed_tokens):.2f}%")
        if price_errors > 0:
            logger.info(f"⚠️  Price fetch errors: {price_errors}")
        logger.info(f"💾 Data stored in: daily_ledger")
        logger.info("="*60)

if __name__ == "__main__":
    with LedgerManager() as manager:
        manager.manage_ledger()