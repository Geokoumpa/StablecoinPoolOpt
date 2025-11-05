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

                # Step 1: Insert basic transaction info first
                for raw_id, raw_json in raw_transactions_result:
                    for tx in raw_json:
                        try:
                            tx_hash = tx.get('transactionHash')
                            if not tx_hash:
                                logger.debug("Skipping transaction without transactionHash.")
                                continue

                            insert_query = text("""
                                INSERT INTO account_transactions (transaction_hash, timestamp, from_address, to_address, value, token_symbol, token_decimals, raw_transaction_id, operation_index)
                                VALUES (:transaction_hash, to_timestamp(:timestamp), :from_address, :to_address, :value, :token_symbol, :token_decimals, :raw_transaction_id, 0)
                                ON CONFLICT (transaction_hash, operation_index) DO NOTHING
                            """)
                            conn.execute(insert_query, {
                                "transaction_hash": tx_hash,
                                "timestamp": tx.get('timestamp'),
                                "from_address": tx.get('from'),
                                "to_address": tx.get('to'),
                                "value": tx.get('value'),
                                "token_symbol": tx.get('tokenInfo', {}).get('symbol'),
                                "token_decimals": tx.get('tokenInfo', {}).get('decimals'),
                                "raw_transaction_id": raw_id
                            })
                        except Exception as e:
                            logger.error(f"Error inserting basic transaction info for {tx.get('transactionHash')}: {e}")

                # Step 2: Fetch and process detailed info for transactions that haven't been enriched yet
                unenriched_txs_result = conn.execute(text("SELECT transaction_hash, raw_transaction_id FROM account_transactions WHERE block_number IS NULL"))
                for tx_hash, raw_id in unenriched_txs_result:
                    try:
                        tx_info = get_tx_info(tx_hash)
                        
                        # Check if tx_info is a valid dictionary
                        if not isinstance(tx_info, dict):
                            logger.warning(f"Invalid tx_info type for {tx_hash}: {type(tx_info)}. Skipping.")
                            continue
                            
                        # Persist raw details
                        insert_details = text("""
                            INSERT INTO raw_ethplorer_account_transaction_details (transaction_hash, raw_json, fetched_at)
                            VALUES (:transaction_hash, CAST(:raw_json AS JSONB), now())
                            ON CONFLICT (transaction_hash) DO UPDATE SET raw_json = CAST(:raw_json AS JSONB), fetched_at = now()
                        """)
                        conn.execute(insert_details, {"transaction_hash": tx_hash, "raw_json": json.dumps(tx_info)})

                        # Extract common values
                        block_number = tx_info.get('blockNumber')
                        confirmations = tx_info.get('confirmations')
                        success = tx_info.get('success')
                        transaction_index = tx_info.get('transactionIndex')
                        nonce = tx_info.get('nonce')
                        raw_value = tx_info.get('rawValue')
                        raw_value_num = Decimal(raw_value) if raw_value is not None else None
                        input_data = tx_info.get('input')
                        gas_limit = tx_info.get('gasLimit')
                        gas_price = tx_info.get('gasPrice')
                        gas_used = tx_info.get('gasUsed')
                        method_id = tx_info.get('methodId')
                        function_name = tx_info.get('functionName')
                        creates = tx_info.get('creates')

                        # Process operations if they exist
                        if 'operations' in tx_info and tx_info['operations']:
                            # Delete the placeholder record
                            conn.execute(text("DELETE FROM account_transactions WHERE transaction_hash = :tx_hash AND operation_index = 0"), {"tx_hash": tx_hash})

                            for i, op in enumerate(tx_info['operations']):
                                op_token_info = op.get('tokenInfo', {})
                                # Ensure op_token_info is a dictionary
                                if not isinstance(op_token_info, dict):
                                    logger.warning(f"Invalid op_token_info type for {tx_hash}, operation {i}: {type(op_token_info)}. Using empty dict.")
                                    op_token_info = {}
                                
                                # Safely get price info, ensuring it's a dictionary
                                price_info = op_token_info.get('price')
                                if isinstance(price_info, dict):
                                    op_price_info = price_info
                                else:
                                    op_price_info = {}
                                
                                insert_op_query = text("""
                                    INSERT INTO account_transactions (
                                        transaction_hash, timestamp, from_address, to_address, value, token_symbol, token_decimals,
                                        raw_transaction_id, operation_index, block_number, confirmations, success,
                                        transaction_index, nonce, raw_value, input_data, gas_limit, gas_price,
                                        gas_used, method_id, function_name, creates, operation_type, operation_priority,
                                        token_address, token_name, token_price_rate, token_price_currency
                                    ) VALUES (
                                        :transaction_hash, to_timestamp(:timestamp), :from_address, :to_address, :value, :token_symbol, :token_decimals,
                                        :raw_transaction_id, :operation_index, :block_number, :confirmations, :success,
                                        :transaction_index, :nonce, :raw_value, :input_data, :gas_limit, :gas_price,
                                        :gas_used, :method_id, :function_name, :creates, :operation_type, :operation_priority,
                                        :token_address, :token_name, :token_price_rate, :token_price_currency
                                    ) ON CONFLICT (transaction_hash, operation_index) DO NOTHING
                                """)
                                conn.execute(insert_op_query, {
                                    "transaction_hash": tx_hash, "timestamp": op.get('timestamp'), "from_address": op.get('from'),
                                    "to_address": op.get('to'), "value": op.get('value'), "token_symbol": op_token_info.get('symbol'),
                                    "token_decimals": op_token_info.get('decimals'), "raw_transaction_id": raw_id, "operation_index": i,
                                    "block_number": block_number, "confirmations": confirmations, "success": success,
                                    "transaction_index": transaction_index, "nonce": nonce, "raw_value": raw_value_num,
                                    "input_data": input_data, "gas_limit": gas_limit, "gas_price": gas_price, "gas_used": gas_used,
                                    "method_id": method_id, "function_name": function_name, "creates": creates,
                                    "operation_type": op.get('type'), "operation_priority": op.get('priority'),
                                    "token_address": op_token_info.get('address'), "token_name": op_token_info.get('name'),
                                    "token_price_rate": op_price_info.get('rate'), "token_price_currency": op_price_info.get('currency')
                                })
                        else:
                            # Update existing record if no operations
                            update_query = text("""
                                UPDATE account_transactions SET
                                    block_number = :block_number, confirmations = :confirmations, success = :success,
                                    transaction_index = :transaction_index, nonce = :nonce, raw_value = :raw_value,
                                    input_data = :input_data, gas_limit = :gas_limit, gas_price = :gas_price,
                                    gas_used = :gas_used, method_id = :method_id, function_name = :function_name, creates = :creates
                                WHERE transaction_hash = :transaction_hash AND operation_index = 0
                            """)
                            conn.execute(update_query, {
                                "transaction_hash": tx_hash, "block_number": block_number, "confirmations": confirmations,
                                "success": success, "transaction_index": transaction_index, "nonce": nonce,
                                "raw_value": raw_value_num, "input_data": input_data, "gas_limit": gas_limit,
                                "gas_price": gas_price, "gas_used": gas_used, "method_id": method_id,
                                "function_name": function_name, "creates": creates
                            })
                    except Exception as e:
                        logger.error(f"Error fetching/persisting tx info for {tx_hash}: {e}")
                logger.info("Successfully processed account transactions.")

    except Exception as e:
        logger.error(f"Error processing account transactions: {e}")
    finally:
        if engine:
            engine.dispose()

if __name__ == "__main__":
    process_account_transactions()