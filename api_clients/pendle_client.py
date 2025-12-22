
import logging
import requests
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

BASE_URL = "https://api-v2.pendle.finance/core"

def get_sdk_tokens(market_address: str, chain_id: int = 1, timeout: int = 10) -> Optional[Dict]:
    """
    Fetch tokens info for a market from Pendle SDK endpoint:
    /v1/sdk/{chainId}/markets/{marketAddress}/tokens
    
    Returns a dict containing 'tokensIn', 'tokensMintSy', 'tokensRedeemSy' etc.
    """
    url = f"{BASE_URL}/v1/sdk/{chain_id}/markets/{market_address}/tokens"
    
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Pendle SDK API request failed for market {market_address}: {e}")
        return None
    except ValueError as e:
        logger.error(f"Failed to decode Pendle SDK API response for market {market_address}: {e}")
        return None

def get_all_markets(chain_id: int = 1, timeout: int = 10) -> Optional[Dict[str, str]]:
    """
    Fetch all markets for a chain and return a map of {PoolAddress: UnderlyingAssetAddress}.
    Endpoint: /v1/markets/all?chainId={chain_id}
    """
    url = f"{BASE_URL}/v1/markets/all"
    params = {'chainId': chain_id}
    
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        
        markets = []
        if isinstance(data, dict) and 'markets' in data:
             markets = data['markets']
        elif isinstance(data, list):
             markets = data
        else:
             logger.warning(f"Unexpected response format from Pendle markets/all: {type(data)}")
             return None
        
        market_map = {}
        for m in markets:
            addr = m.get('address')
            ua = m.get('underlyingAsset')
            
            if addr and ua:
                # Clean prefix "1-0x..." -> "0x..."
                if '-' in ua:
                    ua = ua.split('-')[-1]
                
                market_map[addr.lower()] = ua.lower()
                
        logger.info(f"Fetched {len(market_map)} markets from Pendle API.")
        return market_map

    except requests.exceptions.RequestException as e:
        logger.error(f"Pendle Markets API request failed: {e}")
        return None
    except ValueError as e:
        logger.error(f"Failed to decode Pendle Markets API response: {e}")
        return None
