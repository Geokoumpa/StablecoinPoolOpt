import logging
from database.db_utils import get_db_connection
from sqlalchemy import text

logger = logging.getLogger(__name__)

def process_account_transactions():
    """
    Process raw Ethplorer transaction data and insert unique transactions into the account_transactions table.
    """
    engine = None
    try:
        engine = get_db_connection()
        if not engine:
            logger.error("Could not establish database connection. Exiting.")
            return

        with engine.connect() as conn:
            with conn.begin():
                # Get the last processed raw transaction id
                last_processed_id_result = conn.execute(text("SELECT MAX(raw_transaction_id) FROM account_transactions")).scalar_one_or_none()
                last_processed_id = last_processed_id_result or 0

                # Fetch new raw transactions
                raw_transactions_result = conn.execute(
                    text("SELECT id, raw_json_data FROM raw_ethplorer_account_transactions WHERE id > :last_processed_id ORDER BY id"),
                    {"last_processed_id": last_processed_id}
                )

                for raw_id, raw_json in raw_transactions_result:
                    for tx in raw_json:
                        try:
                            # Extract relevant data
                            tx_hash = tx.get('transactionHash')
                            timestamp = tx.get('timestamp')
                            from_address = tx.get('from')
                            to_address = tx.get('to')
                            value = tx.get('value')
                            token_info = tx.get('tokenInfo', {})
                            token_symbol = token_info.get('symbol')
                            token_decimals = token_info.get('decimals')

                            # Insert into processed table, duplicates will be ignored due to UNIQUE constraint
                            insert_query = text("""
                                INSERT INTO account_transactions (transaction_hash, timestamp, from_address, to_address, value, token_symbol, token_decimals, raw_transaction_id)
                                VALUES (:transaction_hash, to_timestamp(:timestamp), :from_address, :to_address, :value, :token_symbol, :token_decimals, :raw_transaction_id)
                                ON CONFLICT (transaction_hash) DO NOTHING
                            """)
                            conn.execute(insert_query, {
                                "transaction_hash": tx_hash,
                                "timestamp": timestamp,
                                "from_address": from_address,
                                "to_address": to_address,
                                "value": value,
                                "token_symbol": token_symbol,
                                "token_decimals": token_decimals,
                                "raw_transaction_id": raw_id
                            })
                        except Exception as e:
                            logger.error(f"Error processing transaction {tx.get('transactionHash')}: {e}")
                logger.info("Successfully processed account transactions.")

    except Exception as e:
        logger.error(f"Error processing account transactions: {e}")
    finally:
        if engine:
            engine.dispose()

if __name__ == "__main__":
    process_account_transactions()