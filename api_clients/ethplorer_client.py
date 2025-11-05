import logging
import requests
from typing import Optional, Dict
from config import ETHPLORER_API_KEY

logger = logging.getLogger(__name__)

BASE_URL = "https://api.ethplorer.io"


def _get(endpoint: str, params: Optional[Dict] = None, timeout: int = 10) -> Optional[Dict]:
    """
    Internal helper to perform GET requests against Ethplorer API.
    Returns parsed JSON on success, None on failure.
    """
    if params is None:
        params = {}
    api_key = params.get("apiKey") or ETHPLORER_API_KEY
    if api_key:
        params["apiKey"] = api_key
    else:
        logger.warning("ETHPLORER_API_KEY is not configured; cannot call Ethplorer API.")
        return None

    try:
        resp = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=timeout)
        resp.raise_for_status()
        json_response = resp.json()
        if isinstance(json_response, dict):
            return json_response
        else:
            logger.warning(f"Ethplorer returned non-dict response for {endpoint}: {json_response}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Ethplorer request failed for endpoint {endpoint}: {e}")
    except ValueError as e:
        logger.error(f"Failed to decode JSON response for endpoint {endpoint}: {e}")
    return None


def get_tx_info(tx_hash: str, api_key: Optional[str] = None, timeout: int = 10) -> Optional[Dict]:
    """
    Fetch transaction details from Ethplorer /getTxInfo/{txHash}.
    """
    params = {"apiKey": api_key} if api_key else None
    result = _get(f"/getTxInfo/{tx_hash}", params=params, timeout=timeout)
    
    # Ensure we always return None or a dict
    if result is None or isinstance(result, dict):
        return result
    else:
        logger.warning(f"get_tx_info returned unexpected type for {tx_hash}: {type(result)}")
        return None


def get_address_history(address: str, api_key: Optional[str] = None, limit: int = 100, timeout: int = 10) -> Optional[Dict]:
    """
    Fetch address history via /getAddressHistory/{address}.
    Returns parsed JSON on success or None on failure.
    """
    params = {"limit": limit}
    if api_key:
        params["apiKey"] = api_key
    result = _get(f"/getAddressHistory/{address}", params=params, timeout=timeout)
    
    # Ensure we always return None or a dict
    if result is None or isinstance(result, dict):
        return result
    else:
        logger.warning(f"get_address_history returned unexpected type for {address}: {type(result)}")
        return None