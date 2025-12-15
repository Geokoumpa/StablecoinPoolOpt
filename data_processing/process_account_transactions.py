import logging
import json
from decimal import Decimal
from datetime import datetime, timezone
from api_clients.ethplorer_client import get_tx_info
from database.repositories.transaction_repository import TransactionRepository
from database.repositories.raw_data_repository import RawDataRepository

# Configure module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def process_account_transactions():
    """
    Process raw Ethplorer transaction data and insert unique transactions into the account_transactions table.
    For each newly inserted transaction, fetch detailed tx info and enrich the data.
    """
    logger.info("Starting account transaction processing...")
    
    # Initialize repositories
    tx_repo = TransactionRepository()
    raw_repo = RawDataRepository()

    try:
        # Get the last processed raw transaction id
        last_processed_id = tx_repo.get_last_processed_raw_id()
        logger.info(f"Last processed raw transaction ID: {last_processed_id}")

        # Fetch new raw transactions
        raw_transactions = raw_repo.get_new_raw_ethplorer_transactions(last_processed_id)
        
        # Step 1: Insert basic transaction info
        if raw_transactions:
            logger.info(f"Processing {len(raw_transactions)} new raw transactions...")
            transactions_to_insert = []
            
            for raw_id, raw_json in raw_transactions:
                operations = raw_json if isinstance(raw_json, list) else raw_json.get('operations', [])
                for tx in operations:
                    tx_hash = tx.get('transactionHash')
                    if not tx_hash:
                        continue
                    
                    transactions_to_insert.append({
                        'transaction_hash': tx_hash,
                        'operation_index': 0, # Initial placeholder
                        'timestamp':  datetime.fromtimestamp(tx.get('timestamp'), timezone.utc) if tx.get('timestamp') else None,
                        'from_address': tx.get('from'),
                        'to_address': tx.get('to'),
                        'value': tx.get('value'),
                        'token_symbol': tx.get('tokenInfo', {}).get('symbol'),
                        'token_decimals': tx.get('tokenInfo', {}).get('decimals'),
                        'raw_transaction_id': raw_id,
                        # Other fields will be null initially
                        'insertion_timestamp': None # Let DB handle or adding now? Repo helper uses missing.
                    })

            if transactions_to_insert:
               tx_repo.bulk_insert_transactions(transactions_to_insert)
               logger.info(f"Inserted/Skipped {len(transactions_to_insert)} basic transaction records.")

        # Step 2: Fetch and process detailed info
        unenriched_txs = tx_repo.get_unenriched_transactions()
        logger.info(f"Found {len(unenriched_txs)} transactions pending enrichment.")

        for tx_hash, raw_id in unenriched_txs:
            try:
                tx_info = get_tx_info(tx_hash)
                
                if not isinstance(tx_info, dict):
                    logger.warning(f"Invalid tx_info for {tx_hash}. Skipping.")
                    continue
                    
                # Persist raw details
                raw_repo.insert_raw_tx_details([{
                    'transaction_hash': tx_hash,
                    'raw_json': json.dumps(tx_info)
                }])

                # Extract common values
                block_number = tx_info.get('blockNumber')
                confirmations = tx_info.get('confirmations')
                success = tx_info.get('success')
                transaction_index = tx_info.get('transactionIndex')
                nonce = tx_info.get('nonce')
                raw_value = tx_info.get('rawValue')
                if raw_value is not None:
                     # raw_value in repository insert is handled? 
                     # The repository maps 'raw_value' to table column.
                     # Table column is Numeric/Decimal?
                     # Model Transaction.raw_value is Numeric.
                     # In python usually Decimal.
                     pass 
                
                input_data = tx_info.get('input')
                gas_limit = tx_info.get('gasLimit')
                gas_price = tx_info.get('gasPrice')
                gas_used = tx_info.get('gasUsed')
                method_id = tx_info.get('methodId')
                function_name = tx_info.get('functionName')
                creates = tx_info.get('creates')

                # Process operations if they exist
                if 'operations' in tx_info and tx_info['operations']:
                    # Delete the placeholder record (operation_index=0)
                    tx_repo.delete_transaction(tx_hash, 0)

                    ops_to_insert = []
                    for i, op in enumerate(tx_info['operations']):
                        op_token_info = op.get('tokenInfo', {}) if isinstance(op.get('tokenInfo'), dict) else {}
                        price_info = op_token_info.get('price', {}) if isinstance(op_token_info.get('price'), dict) else {}

                        ops_to_insert.append({
                            'transaction_hash': tx_hash,
                            'timestamp': datetime.fromtimestamp(op.get('timestamp'), timezone.utc) if op.get('timestamp') else None,
                            'from_address': op.get('from'),
                            'to_address': op.get('to'),
                            'value': op.get('value'),
                            'token_symbol': op_token_info.get('symbol'),
                            'token_decimals': op_token_info.get('decimals'),
                            'raw_transaction_id': raw_id,
                            'operation_index': i,
                            'block_number': block_number,
                            'confirmations': confirmations,
                            'success': success,
                            'transaction_index': transaction_index,
                            'nonce': nonce,
                            'raw_value': raw_value,
                            'input_data': input_data,
                            'gas_limit': gas_limit,
                            'gas_price': gas_price,
                            'gas_used': gas_used,
                            'method_id': method_id,
                            'function_name': function_name,
                            'creates': creates,
                            'operation_type': op.get('type'),
                            'operation_priority': op.get('priority'),
                            'token_address': op_token_info.get('address'),
                            'token_name': op_token_info.get('name'),
                            'token_price_rate': price_info.get('rate'),
                            'token_price_currency': price_info.get('currency')
                        })
                    
                    tx_repo.bulk_insert_transactions(ops_to_insert)

                else:
                    # Update existing record if no operations
                    updates = {
                        "block_number": block_number, "confirmations": confirmations, "success": success,
                        "transaction_index": transaction_index, "nonce": nonce, "raw_value": raw_value,
                        "input_data": input_data, "gas_limit": gas_limit, "gas_price": gas_price,
                        "gas_used": gas_used, "method_id": method_id, "function_name": function_name, "creates": creates
                    }
                    tx_repo.update_transaction(tx_hash, 0, updates)
            
            except Exception as e:
                logger.error(f"Error fetching/persisting tx info for {tx_hash}: {e}")

        logger.info("Successfully processed account transactions.")

    except Exception as e:
        logger.error(f"Error processing account transactions: {e}")
        raise

if __name__ == "__main__":
    process_account_transactions()