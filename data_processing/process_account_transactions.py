import logging
import json
from decimal import Decimal
from database.db_utils import get_db_connection
from sqlalchemy import text
from api_clients.ethplorer_client import get_tx_info
from config import MAIN_ASSET_HOLDING_ADDRESS

logger = logging.getLogger(__name__)

def process_account_transactions():
    """
    Process raw Ethplorer transaction data and insert unique transactions into the account_transactions table.
    For each newly inserted transaction, fetch detailed tx info from Ethplorer via api_clients.ethplorer_client.get_tx_info
    and persist the response to raw_ethplorer_account_transaction_details.
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

                            if not tx_hash:
                                logger.debug("Skipping transaction without transactionHash.")
                                continue

                            # Insert into processed table, duplicates will be ignored due to UNIQUE constraint
                            insert_query = text("""
                                INSERT INTO account_transactions (transaction_hash, timestamp, from_address, to_address, value, token_symbol, token_decimals, raw_transaction_id)
                                VALUES (:transaction_hash, to_timestamp(:timestamp), :from_address, :to_address, :value, :token_symbol, :token_decimals, :raw_transaction_id)
                                ON CONFLICT (transaction_hash) DO NOTHING
                            """)
                            result = conn.execute(insert_query, {
                                "transaction_hash": tx_hash,
                                "timestamp": timestamp,
                                "from_address": from_address,
                                "to_address": to_address,
                                "value": value,
                                "token_symbol": token_symbol,
                                "token_decimals": token_decimals,
                                "raw_transaction_id": raw_id
                            })

                            # If a new row was inserted, fetch tx info and persist it
                            try:
                                rowcount = result.rowcount if result is not None else 0
                            except Exception:
                                rowcount = 0

                            if rowcount:
                                try:
                                    tx_info = get_tx_info(tx_hash)
                                    if tx_info is not None:
                                        # persist raw details table
                                        insert_details = text("""
                                            INSERT INTO raw_ethplorer_account_transaction_details (transaction_hash, raw_json, fetched_at)
                                            VALUES (:transaction_hash, CAST(:raw_json AS JSONB), now())
                                            ON CONFLICT (transaction_hash) DO UPDATE
                                            SET raw_json = CAST(:raw_json AS JSONB), fetched_at = now()
                                        """)
                                        conn.execute(insert_details, {"transaction_hash": tx_hash, "raw_json": json.dumps(tx_info)})

                                        # Map Ethplorer response fields to account_transactions columns and update the row
                                        update_query = text("""
                                            UPDATE account_transactions SET
                                                block_number = :block_number,
                                                confirmations = :confirmations,
                                                success = :success,
                                                transaction_index = :transaction_index,
                                                nonce = :nonce,
                                                raw_value = :raw_value,
                                                input_data = :input_data,
                                                gas_limit = :gas_limit,
                                                gas_price = :gas_price,
                                                gas_used = :gas_used,
                                                method_id = :method_id,
                                                function_name = :function_name,
                                                creates = :creates
                                            WHERE transaction_hash = :transaction_hash
                                        """)

                                        # Extract values safely from tx_info
                                        block_number = tx_info.get('blockNumber')
                                        confirmations = tx_info.get('confirmations')
                                        success = tx_info.get('success')
                                        transaction_index = tx_info.get('transactionIndex')
                                        nonce = tx_info.get('nonce')

                                        raw_value = tx_info.get('rawValue')
                                        try:
                                            raw_value_num = Decimal(raw_value) if raw_value is not None else None
                                        except Exception:
                                            raw_value_num = None

                                        input_data = tx_info.get('input')
                                        gas_limit = tx_info.get('gasLimit')
                                        gas_price = tx_info.get('gasPrice')
                                        gas_used = tx_info.get('gasUsed')
                                        method_id = tx_info.get('methodId')
                                        function_name = tx_info.get('functionName')
                                        creates = tx_info.get('creates')

                                        conn.execute(update_query, {
                                            "block_number": block_number,
                                            "confirmations": confirmations,
                                            "success": success,
                                            "transaction_index": transaction_index,
                                            "nonce": nonce,
                                            "raw_value": raw_value_num,
                                            "input_data": input_data,
                                            "gas_limit": gas_limit,
                                            "gas_price": gas_price,
                                            "gas_used": gas_used,
                                            "method_id": method_id,
                                            "function_name": function_name,
                                            "creates": creates,
                                            "transaction_hash": tx_hash
                                        })
                                except Exception as e:
                                    logger.error(f"Error fetching/persisting tx info for {tx_hash}: {e}")
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