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

                            try:
                                tx_info = get_tx_info(tx_hash)
                                if isinstance(tx_info, dict):
                                    # persist raw details table
                                    insert_details = text("""
                                        INSERT INTO raw_ethplorer_account_transaction_details (transaction_hash, raw_json, fetched_at)
                                        VALUES (:transaction_hash, CAST(:raw_json AS JSONB), now())
                                        ON CONFLICT (transaction_hash) DO UPDATE
                                        SET raw_json = CAST(:raw_json AS JSONB), fetched_at = now()
                                    """)
                                    conn.execute(insert_details, {"transaction_hash": tx_hash, "raw_json": json.dumps(tx_info)})

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

                                    # If there are operations, process them
                                    if 'operations' in tx_info and tx_info['operations']:
                                        for i, op in enumerate(tx_info['operations']):
                                            op_from = op.get('from')
                                            op_to = op.get('to')
                                            op_value = op.get('value')
                                            op_token_info = op.get('tokenInfo', {})
                                            op_token_symbol = op_token_info.get('symbol')
                                            op_token_decimals = op_token_info.get('decimals')
                                            op_token_address = op_token_info.get('address')
                                            op_token_name = op_token_info.get('name')
                                            op_price_info = op_token_info.get('price', {})
                                            op_token_price_rate = op_price_info.get('rate')
                                            op_token_price_currency = op_price_info.get('currency')
                                            op_type = op.get('type')
                                            op_priority = op.get('priority')

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
                                                "transaction_hash": tx_hash,
                                                "timestamp": op.get('timestamp'),
                                                "from_address": op_from,
                                                "to_address": op_to,
                                                "value": op_value,
                                                "token_symbol": op_token_symbol,
                                                "token_decimals": op_token_decimals,
                                                "raw_transaction_id": raw_id,
                                                "operation_index": i,
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
                                                "operation_type": op_type,
                                                "operation_priority": op_priority,
                                                "token_address": op_token_address,
                                                "token_name": op_token_name,
                                                "token_price_rate": op_token_price_rate,
                                                "token_price_currency": op_token_price_currency
                                            })
                                    else:
                                        # Insert a single record if no operations
                                        insert_query = text("""
                                            INSERT INTO account_transactions (
                                                transaction_hash, timestamp, from_address, to_address, value, token_symbol, token_decimals,
                                                raw_transaction_id, operation_index, block_number, confirmations, success,
                                                transaction_index, nonce, raw_value, input_data, gas_limit, gas_price,
                                                gas_used, method_id, function_name, creates, operation_type, operation_priority,
                                                token_address, token_name, token_price_rate, token_price_currency
                                            ) VALUES (
                                                :transaction_hash, to_timestamp(:timestamp), :from_address, :to_address, :value, :token_symbol, :token_decimals,
                                                :raw_transaction_id, 0, :block_number, :confirmations, :success,
                                                :transaction_index, :nonce, :raw_value, :input_data, :gas_limit, :gas_price,
                                                :gas_used, :method_id, :function_name, :creates, NULL, NULL,
                                                NULL, NULL, NULL, NULL
                                            ) ON CONFLICT (transaction_hash, operation_index) DO NOTHING
                                        """)
                                        conn.execute(insert_query, {
                                            "transaction_hash": tx_hash,
                                            "timestamp": timestamp,
                                            "from_address": from_address,
                                            "to_address": to_address,
                                            "value": value,
                                            "token_symbol": token_symbol,
                                            "token_decimals": token_decimals,
                                            "raw_transaction_id": raw_id,
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
                                            "creates": creates
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