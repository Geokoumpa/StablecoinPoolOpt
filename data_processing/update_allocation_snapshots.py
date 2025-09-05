import logging
import json
from datetime import datetime
from database.db_utils import get_db_connection
import pandas as pd
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_dynamic_lists(engine):
    """Fetches current dynamic lists from individual tables."""
    approved_tokens = pd.read_sql("SELECT token_symbol FROM approved_tokens;", engine)['token_symbol'].tolist()
    blacklisted_tokens = pd.read_sql("SELECT token_symbol FROM blacklisted_tokens;", engine)['token_symbol'].tolist()
    approved_protocols = pd.read_sql("SELECT protocol_name FROM approved_protocols;", engine)['protocol_name'].tolist()
    icebox_tokens = pd.read_sql("SELECT token_symbol FROM icebox_tokens WHERE removed_timestamp IS NULL;", engine)['token_symbol'].tolist()

    return {
        "approved_tokens": [{"token_symbol": token} for token in approved_tokens],
        "blacklisted_tokens": [{"token_symbol": token} for token in blacklisted_tokens],
        "approved_protocols": [{"protocol_name": protocol} for protocol in approved_protocols],
        "icebox_tokens": [{"token_symbol": token} for token in icebox_tokens],
    }

def update_allocation_snapshots():
    """
    Updates the snapshots in the latest allocation_parameters entry with current dynamic lists.
    This is called after icebox logic has updated the icebox_tokens table.
    """
    logging.info("Updating allocation parameter snapshots...")
    engine = None
    try:
        engine = get_db_connection()
        if not engine:
            raise Exception("Failed to establish database connection")

        # Fetch current dynamic lists from individual tables
        dynamic_lists = fetch_dynamic_lists(engine)

        # Update the latest allocation_parameters entry with new snapshots
        with engine.connect() as conn:
            with conn.begin():
                conn.execute(text("""
                    UPDATE allocation_parameters
                    SET
                        approved_tokens_snapshot = :approved_tokens,
                        blacklisted_tokens_snapshot = :blacklisted_tokens,
                        approved_protocols_snapshot = :approved_protocols,
                        icebox_tokens_snapshot = :icebox_tokens,
                        timestamp = :timestamp
                    WHERE run_id = (
                        SELECT run_id FROM allocation_parameters
                        ORDER BY timestamp DESC LIMIT 1
                    );
                """), {
                    "approved_tokens": json.dumps(dynamic_lists['approved_tokens']),
                    "blacklisted_tokens": json.dumps(dynamic_lists['blacklisted_tokens']),
                    "approved_protocols": json.dumps(dynamic_lists['approved_protocols']),
                    "icebox_tokens": json.dumps(dynamic_lists['icebox_tokens']),
                    "timestamp": datetime.now()
                })

            logging.info("Updated allocation parameter snapshots with latest dynamic lists")
            logging.info(f"Approved protocols: {len(dynamic_lists['approved_protocols'])}")
            logging.info(f"Approved tokens: {len(dynamic_lists['approved_tokens'])}")
            logging.info(f"Blacklisted tokens: {len(dynamic_lists['blacklisted_tokens'])}")
            logging.info(f"Icebox tokens: {len(dynamic_lists['icebox_tokens'])}")

    except Exception as e:
        logging.error(f"Error updating allocation snapshots: {e}")
        raise

if __name__ == "__main__":
    update_allocation_snapshots()