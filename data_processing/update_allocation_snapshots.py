import logging
import json
from datetime import datetime
from typing import Dict, List

from database.repositories.parameter_repository import ParameterRepository
from database.repositories.token_repository import TokenRepository
from database.repositories.protocol_repository import ProtocolRepository

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

def fetch_dynamic_lists() -> Dict[str, List[Dict[str, str]]]:
    """Fetches current dynamic lists using repositories."""
    token_repo = TokenRepository()
    protocol_repo = ProtocolRepository()
    
    # Approved tokens
    approved_tokens = token_repo.get_approved_tokens()
    approved_tokens_list = [{"token_symbol": t.token_symbol} for t in approved_tokens]
    
    # Blacklisted tokens
    blacklisted_tokens = token_repo.get_blacklisted_tokens()
    blacklisted_tokens_list = [{"token_symbol": t.token_symbol} for t in blacklisted_tokens]
    
    # Icebox tokens
    icebox_tokens = token_repo.get_icebox_tokens()
    icebox_tokens_list = [{"token_symbol": t.token_symbol} for t in icebox_tokens]
    
    # Approved protocols
    approved_protocols = protocol_repo.get_approved_protocols()
    approved_protocols_list = [{"protocol_name": p.protocol_name} for p in approved_protocols]
    
    return {
        "approved_tokens": approved_tokens_list,
        "blacklisted_tokens": blacklisted_tokens_list,
        "approved_protocols": approved_protocols_list,
        "icebox_tokens": icebox_tokens_list,
    }

def update_allocation_snapshots():
    """
    Updates the snapshots in the latest allocation_parameters entry with current dynamic lists.
    This is called after icebox logic has updated the icebox_tokens table.
    """
    logger.info("Updating allocation parameter snapshots...")
    try:
        # Fetch current dynamic lists
        dynamic_lists = fetch_dynamic_lists()

        # Update the latest allocation_parameters entry
        repo = ParameterRepository()
        repo.update_snapshots(
            approved_tokens=dynamic_lists['approved_tokens'],
            blacklisted_tokens=dynamic_lists['blacklisted_tokens'],
            approved_protocols=dynamic_lists['approved_protocols'],
            icebox_tokens=dynamic_lists['icebox_tokens']
        )

        logger.info("Updated allocation parameter snapshots with latest dynamic lists")
        logger.info(f"Approved protocols: {len(dynamic_lists['approved_protocols'])}")
        logger.info(f"Approved tokens: {len(dynamic_lists['approved_tokens'])}")
        logger.info(f"Blacklisted tokens: {len(dynamic_lists['blacklisted_tokens'])}")
        logger.info(f"Icebox tokens: {len(dynamic_lists['icebox_tokens'])}")

    except Exception as e:
        logger.error(f"Error updating allocation snapshots: {e}")
        raise

if __name__ == "__main__":
    update_allocation_snapshots()